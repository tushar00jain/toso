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
* **Prefill/decode coupling**: an instance's compute timeline
  (:attr:`DecodeEngine.compute_busy`) is owned *here*, in the data plane, because
  it is the physical resource. On a **coupled** instance prefill runs on that same
  timeline, so a long prefill delays the next decode step and spikes that token's
  TBT. That is expressed by two explicit calls rather than a dict shared with the
  scheduler: :meth:`DecodeEngine.reserve` applies a prefill reservation to the
  timeline, and ``on_compute_busy`` reports each step's end back out so the
  control plane's *predicted* prefill queue can be corrected. A disaggregated pool
  simply never makes either call, so decode is never disturbed by prefill.

Async & deterministic: each instance's decode runs as a coroutine on the shared
run's event loop -- one step per
``await asyncio.sleep(step_time)``. Recency/clocks are the loop's virtual time; no
wall-clock, no randomness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, Model,
)

from ..workload.request import Request


@dataclass
class _Active:
    """One request currently occupying (or queued for) a decode batch slot."""

    request: Request
    remaining: int              # decode tokens still to generate (output - first)
    last_token_time: float      # sim time of its previous token (init: join time)
    tbt_max: float = 0.0        # worst inter-token gap observed so far
    tokens: int = 0             # decode tokens emitted so far


class DecodeEngine:
    """Drives batched, stepped decode for a set of decode instances.

    Args:
        loop: the run's event loop (for the clock and task creation). Under
            simulation its clock is virtual, so a step costs no wall time.
        decode_ids: the decode instance ids.
        max_batch: VRAM cap on a decode batch.
        profile: target-machine profile driving :func:`decode_step_time`.
        model: served-model profile supplying the decode flop term.
        on_finish: called ``(request, tbt_max)`` when a request emits its last
            token.
        on_compute_busy: called ``(instance, until)`` every time a step occupies
            an instance's compute timeline. The serving plane passes this only for
            a **coupled** instance, where prefill shares that timeline; a
            disaggregated pool leaves it ``None``.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        decode_ids: List[str],
        *,
        max_batch: int,
        profile=DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        on_finish: Optional[Callable[[Request, float], None]] = None,
        on_compute_busy: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.loop = loop
        self.ids = sorted(decode_ids)
        self.max_batch = max_batch
        self.profile = profile
        self.model = model
        # Batch=1 baseline step time for the decode-load prediction (each remaining
        # token is assumed to cost ~one uncontended step). Derived per engine from
        # this run's profiles, not once at import from the defaults.
        self._base_step = decode_step_time(1, profile, model)
        # The ACTUAL per-instance compute timeline. Owned here; the control plane
        # keeps its own predicted queue and hears about this one through
        # ``on_compute_busy`` when the two are the same physical resource.
        self.compute_busy: Dict[str, float] = {i: 0.0 for i in self.ids}
        self.on_finish = on_finish
        self.on_compute_busy = on_compute_busy
        self.batch: Dict[str, List[_Active]] = {i: [] for i in self.ids}
        self.pending: Dict[str, List[_Active]] = {i: [] for i in self.ids}
        self._step_task: Dict[str, Optional["asyncio.Task"]] = {
            i: None for i in self.ids
        }

    # -- queries the scheduler observes us through (control.DecodeLoad) ---- #
    def occupancy(self, inst: str) -> int:
        """Requests currently decoding or queued on ``inst`` (the live batch load)."""
        return len(self.batch[inst]) + len(self.pending[inst])

    def predict_occupancy(self, inst: str, at_t: float) -> int:
        """Estimate how many current occupants are *still* decoding at ``at_t``.

        Uses the uniform per-token assumption (each remaining token ~ one
        uncontended step), matching the decode-load prediction: a request is still
        resident at ``at_t`` if its estimated finish time is past ``at_t``.
        """
        n = 0
        for a in self.batch[inst] + self.pending[inst]:
            finish = a.last_token_time + a.remaining * self._base_step
            if finish > at_t:
                n += 1
        return n

    # -- the coupled-instance timeline ------------------------------------- #
    def reserve(self, inst: str, until: float) -> None:
        """Occupy ``inst``'s compute timeline until ``until`` (a prefill).

        Called by the serving plane only when prefill and decode share the
        instance. It is the actuation half of the coupling; ``on_compute_busy`` is
        the observation half.
        """
        if inst in self.compute_busy:
            self.compute_busy[inst] = until

    # -- lifecycle -------------------------------------------------------- #
    def admit(self, request: Request, inst: str) -> None:
        """Enter ``request`` into ``inst``'s decode batch at the current sim time.

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
            request=request, remaining=remaining, last_token_time=self.loop.time()
        )
        if len(self.batch[inst]) < self.max_batch:
            self.batch[inst].append(a)
        else:
            self.pending[inst].append(a)  # VRAM full: wait counts against TBT
        self._ensure_stepping(inst)

    def _ensure_stepping(self, inst: str) -> None:
        """Start ``inst``'s decode-step loop unless one is already running."""
        task = self._step_task[inst]
        if (task is not None and not task.done()) or not self.batch[inst]:
            return
        self._step_task[inst] = self.loop.create_task(self._run_steps(inst))

    async def _run_steps(self, inst: str) -> None:
        """Emit tokens one step at a time until ``inst``'s batch drains."""
        while self.batch[inst]:
            members = list(self.batch[inst])       # frozen for this step
            dt = decode_step_time(len(members), self.profile, self.model)
            start = max(self.loop.time(), self.compute_busy[inst])
            step_end = start + dt
            self.compute_busy[inst] = step_end
            if self.on_compute_busy is not None:
                self.on_compute_busy(inst, step_end)
            await asyncio.sleep(step_end - self.loop.time())
            self._step_complete(inst, members, step_end)

    async def drain(self) -> None:
        """Await all in-flight decode steps so every request finalizes.

        The request coroutines finish at prefill completion, but decode continues
        afterwards on its own step tasks; the runner calls this so the loop keeps
        running until the last token of the last request is emitted.
        """
        while True:
            tasks = [
                t for t in self._step_task.values() if t is not None and not t.done()
            ]
            if not tasks:
                return
            await asyncio.gather(*tasks)

    def _step_complete(
        self, inst: str, members: List[_Active], step_end: float
    ) -> None:
        """Emit one token for each member; retire finished; promote pending."""
        for a in members:
            if a not in self.batch[inst]:
                continue  # defensive; members never leave mid-step in this model
            gap = step_end - a.last_token_time
            a.tbt_max = max(a.tbt_max, gap)
            a.last_token_time = step_end
            a.remaining -= 1
            a.tokens += 1
            if a.remaining == 0:
                self.batch[inst].remove(a)
                if self.on_finish is not None:
                    self.on_finish(a.request, a.tbt_max)
        # A freed VRAM slot admits the next queued request (still counting its wait).
        while self.pending[inst] and len(self.batch[inst]) < self.max_batch:
            self.batch[inst].append(self.pending[inst].pop(0))
