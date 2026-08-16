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
* **Prefill/decode coupling**: this host's compute
  (:class:`~kvcache_sim.data._compute.Accelerator`) is owned by the *host* and handed
  to whichever engines run on it. A host with a
  :class:`~kvcache_sim.data._prefill.PrefillEngine` too shares one between them, so a
  long prefill delays the next decode step and spikes that token's TBT. That is what
  coupling *is*, so there is no flag for it. ``on_compute_busy`` reports each step's
  end back out so control's *predicted* prefill queue can be corrected.

One engine is **one host's** decode side: one batch, one compute timeline, one step
loop, and nothing here takes an ``inst`` argument.

Async & deterministic: decode runs as a coroutine on the shared run's event loop, one
step at a time, each booked on and awaited through the accelerator. Clocks are the
loop's virtual time; no wall-clock, no randomness.

What admission answers with
---------------------------
The step loop is a task and outlives the call that fed it, but
:meth:`DecodeEngine.admit` returns only when its request emits its last token, so the
caller can stamp arrival-to-last-token (:mod:`kvcache_sim.workload._serving`).

It answers with a :class:`Generated`: **the tokens this engine produced**, accumulated
per batch member as the steps land, and **the KV those tokens left behind**, in whole
blocks (:meth:`kvcache_sim.data._compute.Accelerator.generated_kv`). The host publishes
that KV, not this engine, which makes no store calls at all. Unpublished, a volume
would hold only what a prefill put there and a bounded cache could be decoded into
forever without once being pressured.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import torch

from ..control.request import Request
from ._compute import Accelerator

__all__ = ["DecodeEngine", "Generated"]


@dataclass(eq=False)
class Generated:
    """What a decode batch produced for one request: its tokens and its KV.

    A named pair rather than a 2-tuple of ``List[torch.Tensor]``: unpacked the wrong
    way round it would publish tokens under block keys and answer the client with KV,
    and both halves would typecheck all the way down.

    ``eq=False``, as :attr:`_Active.tokens` carries ``compare=False``: a generated
    ``__eq__`` compares tensors elementwise and answers with a tensor, so the first
    ``==`` anywhere would raise "Boolean value of Tensor is ambiguous".
    """

    #: One token per decode step this request was in, in order: its output minus the
    #: first, which prefill produced.
    tokens: List[torch.Tensor]
    #: The KV those tokens left on the decode host, in whole blocks. Empty when no
    #: step ran.
    kv: List[torch.Tensor]


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
    #: What this slot has generated so far, one token per step it has been in,
    #: handed over by :meth:`DecodeEngine._finish`.
    #:
    #: ``compare=False``: ``==`` between two slots would compare these lists element
    #: by element, and comparing two tensors answers with a tensor rather than a
    #: bool. The batch is searched by identity (``a not in self.batch``
    #: short-circuits on ``is``), so nothing needs the comparison anyway.
    tokens: List[torch.Tensor] = field(default_factory=list, compare=False)


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
            token. Telemetry, not control flow: the *caller* learns its request
            finished by :meth:`admit` returning, while this tells the owning host
            what the gaps were so it can write its half of the row. It runs first
            (see :meth:`_finish`).
        on_compute_busy: awaited ``(until)`` every time a step occupies this
            host's compute timeline. The host passes this only when it is
            **coupled**, i.e. prefill shares that timeline; a disaggregated host
            leaves it ``None``.
        on_state: awaited ``(finishes)`` whenever the batch changes, with one
            estimated finish time per request decoding or queued here. The host
            forwards it to control's model, which is how control knows the decode
            load without holding this object.

    Both callbacks are the plane's, not control's: this engine reports to its owner on
    the same host and the owner decides what to send onward. Awaited, because the owner
    sends them over a hop and a report fired off as a task would reorder the run.
    """

    def __init__(
        self,
        compute: Accelerator,
        *,
        max_batch: int,
        on_finish: Optional[Callable[[Request, float], None]] = None,
        on_compute_busy: Optional[Callable[[float], Awaitable[None]]] = None,
        on_state: Optional[Callable[[List[float]], Awaitable[None]]] = None,
    ) -> None:
        self.max_batch = max_batch
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
    async def _report(self) -> None:
        """Push this host's batch state to whoever is listening (the host).

        One estimated finish time per request decoding or queued here, under the
        uniform per-token assumption (each remaining token ~ one uncontended step).
        The count is the occupancy and the values answer "still decoding at ``t``?",
        which is everything control asks about decode. Sent on every batch change,
        because that is when either answer moves.
        """
        if self.on_state is None:
            return
        # Asked of the accelerator, so the baseline step is this host's answer rather
        # than a constant; hoisted because every active request is priced against it.
        base_step = self.compute.step_cost(1)
        await self.on_state([
            a.last_token_time + a.remaining * base_step
            for a in self.batch + self.pending
        ])

    # -- lifecycle -------------------------------------------------------- #
    async def admit(self, request: Request) -> Generated:
        """Enter ``request`` into this host's decode batch; answer with its tokens.

        Returns when its **last token** lands, with a :class:`Generated`. Prefill
        produced the first token, so decode generates ``output_tokens - 1``; a request
        with <= 1 of them finishes immediately with an empty :class:`Generated`.

        The wait is on the slot's own future, so the step loop retires a member without
        knowing who admitted it. Both ways out resolve that future exactly once, and
        both after ``on_finish`` has run (:meth:`_finish`); a step task that dies
        resolves every outstanding one with the exception rather than leaving a caller
        parked forever (:meth:`_abandon`).
        """
        loop = asyncio.get_running_loop()
        done: "asyncio.Future" = loop.create_future()
        remaining = max(0, request.output_tokens - 1)
        if remaining == 0:
            # Never in the batch, so never in a step: retired on the clock instant
            # it arrived, and with no step there is no position of KV either.
            # Awaiting an already-resolved future costs nothing.
            self._finish(request, 0.0, done, [])
            return await done
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
        await self._report()
        self._ensure_stepping()
        return await done

    def _finish(
        self,
        request: Request,
        tbt: float,
        done: "asyncio.Future",
        tokens: List[torch.Tensor],
    ) -> None:
        """Retire one request: report it, then hand over what it produced.

        ``on_finish`` runs first. It writes this host's half of the request's row (the
        inter-token gaps), and a caller released before that could read the row without
        the half it spent the whole decode waiting for.

        ``tokens`` is passed in rather than read off a slot, because a request with
        nothing to decode is retired from :meth:`admit` and has none. The KV is derived
        from ``len(tokens)``, so a single count stays in play.

        ``set_result`` is unguarded on purpose: a second retirement of the same request
        raises :class:`asyncio.InvalidStateError` rather than silently overwriting a
        measurement.
        """
        if self.on_finish is not None:
            self.on_finish(request, tbt)
        done.set_result(
            Generated(tokens=tokens, kv=self.compute.generated_kv(len(tokens)))
        )

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
                    await self.on_compute_busy(step_end)
                await self.compute.wait_until(step_end)
                await self._step_complete(members, step_end)
        except BaseException as exc:      # noqa: BLE001 -- re-raised below
            self._abandon(exc)
            raise

    def _abandon(self, exc: BaseException) -> None:
        """Fail everything this engine will now never finish.

        The step loop is the only thing that resolves an admitted request's future, and
        a caller parked on a future nobody resolves does not fail: it hangs and takes
        the run with it, the original exception swallowed into a task nobody joins. So
        a dying loop hands its exception to every caller waiting on it.

        The batch is cleared with it, or the next :meth:`admit` would start a step loop
        over slots whose futures are already resolved.
        """
        for a in self.batch + self.pending:
            if not a.done.done():
                a.done.set_exception(exc)
        self.batch.clear()
        self.pending.clear()

    async def _step_complete(self, members: List[_Active], step_end: float) -> None:
        """Emit one token for each member; retire finished; promote pending.

        One token per member, in the order the members were frozen in, so a slot
        accumulates the token the step produced *for it*.
        """
        emitted = self.compute.step_tokens(len(members))
        for a, token in zip(members, emitted):
            if a not in self.batch:
                continue  # defensive; members never leave mid-step in this model
            gap = step_end - a.last_token_time
            a.tbt_max = max(a.tbt_max, gap)
            a.last_token_time = step_end
            a.remaining -= 1
            a.tokens.append(token)
            if a.remaining == 0:
                self.batch.remove(a)
                self._finish(a.request, a.tbt_max, a.done, a.tokens)
        # A freed VRAM slot admits the next queued request (still counting its wait).
        while self.pending and len(self.batch) < self.max_batch:
            self.batch.append(self.pending.pop(0))
        await self._report()
