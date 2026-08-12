"""Batched decode engine -- the piece that makes time-between-tokens (TBT) real.

Prefill answers "how fast is the first token" (TTFT); this module answers "how fast
is every *subsequent* token" (TBT). A serving instance decodes a **batch** of
requests together: each decode *step* emits one token for every request in the
batch, and the step's wall time is the TBT every batched request sees for that
token. Step time rises with batch size -- the accelerator says by how much -- so
TBT degrades as an instance fills up, which is exactly the tension a TBT SLO
bounds.

Two levers the design cares about are modelled here:

* **VRAM cap** (``max_batch``): a batch grows only to the cap; requests over it
  queue (``pending``) and their wait counts against their TBT.
* **Prefill/decode coupling**: this host's compute timeline
  (:class:`~kvcache_sim.data._compute.ComputeTimeline`) is owned by the *host*,
  because it is the host's accelerator, and handed to whichever engines run on it.
  A host with a :class:`~kvcache_sim.data._prefill.PrefillEngine` too shares one
  between them, so a long prefill delays the next decode step and spikes that
  token's TBT -- and a host with only this engine shares it with nothing. That is
  what coupling *is*, so there is no flag for it. ``on_compute_busy`` reports each
  step's end back out so the control plane's *predicted* prefill queue can be
  corrected.

One engine is **one host's** decode side: one batch, one compute timeline, one
step loop. It used to be all of them at once, keyed by instance id, which is a
shape no deployment has -- a host has its own VRAM and its own GPU, and knows
nothing of another's batch. Everything that used to take an ``inst`` argument
therefore takes none.

Async & deterministic: decode runs as a coroutine on the shared run's event loop,
one step at a time, each booked on and awaited through the accelerator. Clocks are
the loop's virtual time; no wall-clock, no randomness.

Admission answers with a completion
-----------------------------------
The step loop is a task, so it outlives the call that fed it -- but that is a fact
about *how* the batch is driven, not a licence for the request to outlive its
caller. :meth:`DecodeEngine.admit` hands back a future resolved when that request
emits its last token, so the caller can wait for the thing it asked for. It used to
return at admission and nothing else waited either, which cost two things: the run
needed a drain hook purely to keep the loop alive for a tail nobody was holding,
and no coroutine was still on the request when its last token landed, so nothing
was in a position to measure arrival-to-last-token. Both are gone: the drain hook
is deleted and the client stamps the latency
(:mod:`kvcache_sim.workload._serving`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..control.request import Request
from ._compute import Accelerator

__all__ = ["DecodeEngine"]


@dataclass
class _Active:
    """One request currently occupying (or queued for) a decode batch slot."""

    request: Request
    remaining: int              # decode tokens still to generate (output - first)
    last_token_time: float      # sim time of its previous token (init: join time)
    #: Resolved when this request's last token lands. Whoever admitted the request
    #: is parked on it, which is why it is a field of the slot rather than a
    #: registry beside the batch: a request cannot be in the batch without one, and
    #: cannot leave the batch without it being resolved.
    done: "asyncio.Future"
    tbt_max: float = 0.0        # worst inter-token gap observed so far
    tokens: int = 0             # decode tokens emitted so far


class DecodeEngine:
    """Drives batched, stepped decode for **one host**.

    The clock and task creation come from the running event loop (under
    simulation its clock is virtual, so a step costs no wall time).

    Args:
        compute: the host's :class:`~kvcache_sim.data._compute.Accelerator` -- what
            a step costs and what makes it take that long. The *same object* as its
            prefill engine's when the two are modelled as contending.
        max_batch: VRAM cap on this host's decode batch.
        on_finish: called ``(request, tbt_max)`` when a request emits its last
            token. Telemetry, not control flow: the *caller* learns that its
            request finished from the future :meth:`admit` gave it, and this tells
            the owning host what the gaps were so it can write its half of the
            row. One listener, no return value, and it runs first (see
            :meth:`_finish`).
        on_compute_busy: called ``(until)`` every time a step occupies this
            host's compute timeline. The host passes this only when it is
            **coupled**, i.e. prefill shares that timeline; a disaggregated host
            leaves it ``None``.
        on_state: called ``(finishes)`` whenever the batch changes, with one
            estimated finish time per request decoding or queued here. The host
            forwards it to the coordinator, which is how control knows the decode
            load without holding this object.

    Both callbacks are the plane's, not control's: this engine reports to its
    owner on the same host, and the owner decides what to send onward.
    """

    def __init__(
        self,
        compute: Accelerator,
        *,
        max_batch: int,
        on_finish: Optional[Callable[[Request, float], None]] = None,
        on_compute_busy: Optional[Callable[[float], None]] = None,
        on_state: Optional[Callable[[List[float]], None]] = None,
    ) -> None:
        self.max_batch = max_batch
        # Batch=1 baseline step time for the decode-load prediction (each remaining
        # token is assumed to cost ~one uncontended step). Asked of the accelerator,
        # so it is this host's answer rather than a constant.
        self._base_step = compute.step_cost(1)
        # The ACTUAL compute timeline of this host, shared with whatever else runs
        # on it. The control plane keeps its own predicted queue and hears about
        # this one through ``on_compute_busy``.
        self.compute = compute
        self.on_finish = on_finish
        self.on_compute_busy = on_compute_busy
        self.on_state = on_state
        self.batch: List[_Active] = []
        self.pending: List[_Active] = []
        self._step_task: Optional["asyncio.Task"] = None

    # -- what we report about ourselves ------------------------------------ #
    def _report(self) -> None:
        """Push this host's batch state to whoever is listening (the host).

        One estimated finish time per request decoding or queued here, under the
        uniform per-token assumption (each remaining token ~ one uncontended
        step). The count is the occupancy and the values answer "still decoding
        at ``t``?", which is everything control asks about decode -- so control
        can answer both from this list instead of holding this object. Sent on
        every batch change, because that is when either answer moves.
        """
        if self.on_state is None:
            return
        self.on_state([
            a.last_token_time + a.remaining * self._base_step
            for a in self.batch + self.pending
        ])

    # -- lifecycle -------------------------------------------------------- #
    def admit(self, request: Request) -> "asyncio.Future":
        """Enter ``request`` into this host's decode batch; answer when it is done.

        The first token was produced by prefill (TTFT); decode generates the
        remaining ``output_tokens - 1``. A request with <= 1 output token needs no
        decode and finishes immediately.

        The answer is a future resolved when this request emits its **last**
        token, and it is the reason admission is not fire-and-forget any more.
        Decode used to outlive its caller: ``admit`` returned as soon as the
        request had a slot, the step loop ran on as a separate task, and the run
        needed a drain hook to keep the loop alive for the tail nobody was waiting
        on. Nobody was waiting on it because nobody *could*: the request's caller
        had already returned, so no coroutine was in a position to stamp how long
        the whole thing took. Handing back the completion is what lets the caller
        be that coroutine, and the drain hook was deleted in the same change.

        Not a callback, though this class has two of those. ``on_finish`` and
        ``on_state`` are *telemetry* -- one listener, the owning host, told about
        an event it did not initiate. This is the answer to a specific request the
        caller made, one per admission, and a caller that had to filter a
        broadcast for its own id would be reassembling a return value by hand.

        Both ways out of here resolve it exactly once, and both do it *after*
        ``on_finish`` has run -- see :meth:`_finish`. A step task that dies
        resolves every outstanding one with the exception rather than leaving a
        caller parked forever (:meth:`_abandon`).
        """
        loop = asyncio.get_running_loop()
        done: "asyncio.Future" = loop.create_future()
        remaining = max(0, request.output_tokens - 1)
        if remaining == 0:
            # Never in the batch, so never in a step: retired here, on the same
            # clock instant it arrived. The caller still gets a future rather than
            # a special case, and awaiting an already-resolved one costs nothing.
            self._finish(request, 0.0, done)
            return done
        a = _Active(
            request=request,
            remaining=remaining,
            last_token_time=loop.time(),
            done=done,
        )
        if len(self.batch) < self.max_batch:
            self.batch.append(a)
        else:
            self.pending.append(a)  # VRAM full: wait counts against TBT
        self._report()
        self._ensure_stepping()
        return done

    def _finish(self, request: Request, tbt: float, done: "asyncio.Future") -> None:
        """Retire one request: report it, then release whoever admitted it.

        The order is the whole reason this is a method and not two lines at each
        of its two call sites. ``on_finish`` is how this host's half of the
        request's row reaches the run's ledger (the inter-token gaps); the future
        is what the caller is parked on, and a caller released first could read
        that row before the half it just spent the whole decode waiting for was
        written into it. Nothing today reads the row that early, which is exactly
        why the ordering has to be stated somewhere rather than held by luck.

        ``set_result`` is unguarded on purpose: a second retirement of the same
        request raises :class:`asyncio.InvalidStateError` here rather than
        silently overwriting a measurement, and that is a batch-accounting bug
        worth a traceback.
        """
        if self.on_finish is not None:
            self.on_finish(request, tbt)
        done.set_result(None)

    def _ensure_stepping(self) -> None:
        """Start the decode-step loop unless one is already running."""
        task = self._step_task
        if (task is not None and not task.done()) or not self.batch:
            return
        self._step_task = asyncio.get_running_loop().create_task(self._run_steps())

    async def _run_steps(self) -> None:
        """Emit tokens one step at a time until the batch drains."""
        try:
            while self.batch:
                members = list(self.batch)             # frozen for this step
                # Booked before it is waited for, so the end can be announced to a
                # prefill sharing this accelerator while there is still time to
                # schedule around it.
                step_end = self.compute.claim_step(len(members))
                if self.on_compute_busy is not None:
                    self.on_compute_busy(step_end)
                await self.compute.wait_until(step_end)
                self._step_complete(members, step_end)
        except BaseException as exc:      # noqa: BLE001 -- re-raised below
            self._abandon(exc)
            raise

    def _abandon(self, exc: BaseException) -> None:
        """Fail everything this engine will now never finish.

        Nothing in the step loop is expected to raise, and if that stays true this
        never runs. It exists because of what the alternative failure mode is:
        this task is the only thing that resolves an admitted request's future,
        and a caller parked on a future nobody will ever resolve does not fail --
        it hangs, and takes the whole run with it, with the original exception
        swallowed into a task nobody joins. An error that surfaces as "the
        simulation never terminated" is the worst shape a bug can have here.

        So a dying loop hands its exception to every caller waiting on it, which
        turns a hang into a traceback at the point that cares. The batch is
        cleared with it: those requests are not decoding, and leaving them listed
        would have the next :meth:`admit` start a step loop over slots whose
        futures are already resolved.
        """
        for a in self.batch + self.pending:
            if not a.done.done():
                a.done.set_exception(exc)
        self.batch.clear()
        self.pending.clear()

    def _step_complete(self, members: List[_Active], step_end: float) -> None:
        """Emit one token for each member; retire finished; promote pending."""
        for a in members:
            if a not in self.batch:
                continue  # defensive; members never leave mid-step in this model
            gap = step_end - a.last_token_time
            a.tbt_max = max(a.tbt_max, gap)
            a.last_token_time = step_end
            a.remaining -= 1
            a.tokens += 1
            if a.remaining == 0:
                self.batch.remove(a)
                self._finish(a.request, a.tbt_max, a.done)
        # A freed VRAM slot admits the next queued request (still counting its wait).
        while self.pending and len(self.batch) < self.max_batch:
            self.batch.append(self.pending.pop(0))
        self._report()
