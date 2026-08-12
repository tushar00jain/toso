"""Batched decode engine -- the piece that makes time-between-tokens (TBT) real.

Prefill answers "how fast is the first token" (TTFT); this module answers "how fast
is every *subsequent* token" (TBT). A serving instance decodes a **batch** of
requests together: each decode *step* emits one token for every request in the
batch, and the step's wall time is the TBT every batched request sees for that
token. Step time rises with batch size (:func:`~domain.llm.decode_step_time`,
charged on the GPU roofline), so TBT degrades as an instance fills up -- exactly
the tension a TBT SLO bounds.

Two levers the design cares about are modelled here:

* **VRAM cap** (``max_batch``): a batch grows only to the cap; requests over it
  queue (``pending``) and their wait counts against their TBT.
* **Prefill/decode coupling**: this host's compute timeline
  (:attr:`DecodeEngine.compute_busy`) is owned *here*, in the data plane, because
  it is the physical resource. On a **coupled** host prefill runs on that same
  timeline, so a long prefill delays the next decode step and spikes that token's
  TBT. That is expressed by two explicit calls rather than a dict shared with the
  scheduler: :meth:`DecodeEngine.reserve` applies a prefill reservation to the
  timeline, and ``on_compute_busy`` reports each step's end back out so the
  control plane's *predicted* prefill queue can be corrected. A disaggregated host
  simply never makes either call, so decode is never disturbed by prefill.

One engine is **one host's** decode side: one batch, one compute timeline, one
step loop. It used to be all of them at once, keyed by instance id, which is a
shape no deployment has -- a host has its own VRAM and its own GPU, and knows
nothing of another's batch. Everything that used to take an ``inst`` argument
therefore takes none.

Async & deterministic: decode runs as a coroutine on the shared run's event loop
-- one step per ``await asyncio.sleep(step_time)``. Recency/clocks are the loop's
virtual time; no wall-clock, no randomness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, List, Optional

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, Model,
)

from ..control.request import Request

__all__ = ["DecodeEngine"]


@dataclass
class _Active:
    """One request currently occupying (or queued for) a decode batch slot."""

    request: Request
    remaining: int              # decode tokens still to generate (output - first)
    last_token_time: float      # sim time of its previous token (init: join time)
    tbt_max: float = 0.0        # worst inter-token gap observed so far
    tokens: int = 0             # decode tokens emitted so far


class DecodeEngine:
    """Drives batched, stepped decode for **one host**.

    The clock and task creation come from the running event loop (under
    simulation its clock is virtual, so a step costs no wall time).

    Args:
        max_batch: VRAM cap on this host's decode batch.
        profile: target-machine profile driving :func:`decode_step_time`.
        model: served-model profile supplying the decode flop term.
        on_finish: called ``(request, tbt_max)`` when a request emits its last
            token.
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
        *,
        max_batch: int,
        profile=DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        on_finish: Optional[Callable[[Request, float], None]] = None,
        on_compute_busy: Optional[Callable[[float], None]] = None,
        on_state: Optional[Callable[[List[float]], None]] = None,
    ) -> None:
        self.max_batch = max_batch
        self.profile = profile
        self.model = model
        # Batch=1 baseline step time for the decode-load prediction (each remaining
        # token is assumed to cost ~one uncontended step). Derived per engine from
        # this run's profiles, not once at import from the defaults.
        self._base_step = decode_step_time(1, profile, model)
        # The ACTUAL compute timeline of this host. Owned here; the control plane
        # keeps its own predicted queue and hears about this one through
        # ``on_compute_busy`` when the two are the same physical resource.
        self.compute_busy: float = 0.0
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

    # -- the coupled-host timeline ----------------------------------------- #
    def reserve(self, until: float) -> None:
        """Occupy this host's compute timeline until ``until`` (a prefill).

        Called by the host only when its prefill and decode share the compute. It
        is the actuation half of the coupling; ``on_compute_busy`` is the
        observation half.
        """
        self.compute_busy = until

    # -- lifecycle -------------------------------------------------------- #
    def admit(self, request: Request) -> None:
        """Enter ``request`` into this host's decode batch at the current sim time.

        The first token was produced by prefill (TTFT); decode generates the
        remaining ``output_tokens - 1``. A request with <= 1 output token needs no
        decode and finishes immediately.
        """
        remaining = max(0, request.output_tokens - 1)
        if remaining == 0:
            if self.on_finish is not None:
                self.on_finish(request, 0.0)
            return
        a = _Active(
            request=request,
            remaining=remaining,
            last_token_time=asyncio.get_running_loop().time(),
        )
        if len(self.batch) < self.max_batch:
            self.batch.append(a)
        else:
            self.pending.append(a)  # VRAM full: wait counts against TBT
        self._report()
        self._ensure_stepping()

    def _ensure_stepping(self) -> None:
        """Start the decode-step loop unless one is already running."""
        task = self._step_task
        if (task is not None and not task.done()) or not self.batch:
            return
        self._step_task = asyncio.get_running_loop().create_task(self._run_steps())

    async def _run_steps(self) -> None:
        """Emit tokens one step at a time until the batch drains."""
        loop = asyncio.get_running_loop()
        while self.batch:
            members = list(self.batch)             # frozen for this step
            dt = decode_step_time(len(members), self.profile, self.model)
            start = max(loop.time(), self.compute_busy)
            step_end = start + dt
            self.compute_busy = step_end
            if self.on_compute_busy is not None:
                self.on_compute_busy(step_end)
            await asyncio.sleep(step_end - loop.time())
            self._step_complete(members, step_end)

    async def drain(self) -> None:
        """Await all in-flight decode steps so every request finalizes.

        The request coroutines finish at prefill completion, but decode continues
        afterwards on its own step task; the host calls this so the loop keeps
        running until the last token of the last request is emitted.
        """
        while True:
            task = self._step_task
            if task is None or task.done():
                return
            await task

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
                if self.on_finish is not None:
                    self.on_finish(a.request, a.tbt_max)
        # A freed VRAM slot admits the next queued request (still counting its wait).
        while self.pending and len(self.batch) < self.max_batch:
            self.batch.append(self.pending.pop(0))
        self._report()
