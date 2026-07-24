"""Batched decode engine -- the piece that makes time-between-tokens (TBT) real.

Prefill answers "how fast is the first token" (TTFT); this module answers "how fast
is every *subsequent* token" (TBT). A serving instance decodes a **batch** of
requests together: each decode *step* emits one token for every request in the
batch, and the step's wall time is the TBT every batched request sees for that
token. Step time rises with batch size (:func:`~kvcache_sim.sim.cost.decode_step_time`,
charged on the GPU roofline), so TBT degrades as an instance fills up -- exactly
the tension a TBT SLO bounds.

Two levers the design cares about are modelled here:

* **VRAM cap** (``max_batch``): a batch grows only to the cap; requests over it
  queue (``pending``) and their wait counts against their TBT.
* **Prefill/decode coupling**: when prefill and decode share an instance's compute
  timeline (``compute_busy`` aliased to the scheduler's prefill ``busy_until``), a
  long prefill delays the next decode step, spiking that token's TBT.
  Disaggregation gives decode its own ``compute_busy`` so prefill never stalls it.

Async & deterministic: each instance's decode runs as a coroutine on the shared
:class:`~sim_common.async_engine.AsyncEngine` virtual clock -- one step per
``await asyncio.sleep(step_time)``. Recency/clocks are the loop's virtual time; no
wall-clock, no randomness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .cost import decode_step_time, PROFILE
from .model import Request

# Batch=1 baseline step time, used by the decode-load prediction (each remaining
# token is assumed to cost ~one uncontended step).
_BASE_STEP = decode_step_time(1, PROFILE)


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

    ``compute_busy`` is the per-instance compute timeline. Pass the scheduler's
    prefill ``busy_until`` dict to model a **coupled** instance (prefill delays
    decode); pass a private dict for a **disaggregated** decode pool (no
    interference). ``on_finish(request, tbt_max)`` is called when a request emits
    its last token.

    Args:
        loop: the :class:`~sim_common.async_engine.AsyncEngine` (for the virtual
            clock and task creation).
        decode_ids: the decode instance ids.
        max_batch: VRAM cap on a decode batch.
        profile: target-machine profile driving :func:`decode_step_time`.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        decode_ids: List[str],
        *,
        max_batch: int,
        profile=PROFILE,
        compute_busy: Optional[Dict[str, float]] = None,
        on_finish: Optional[Callable[[Request, float], None]] = None,
    ) -> None:
        self.loop = loop
        self.ids = sorted(decode_ids)
        self.max_batch = max_batch
        self.profile = profile
        self.compute_busy: Dict[str, float] = (
            compute_busy if compute_busy is not None else {i: 0.0 for i in self.ids}
        )
        for i in self.ids:
            self.compute_busy.setdefault(i, 0.0)
        self.on_finish = on_finish
        self.batch: Dict[str, List[_Active]] = {i: [] for i in self.ids}
        self.pending: Dict[str, List[_Active]] = {i: [] for i in self.ids}
        self._step_task: Dict[str, Optional["asyncio.Task"]] = {
            i: None for i in self.ids
        }

    # -- queries used by the scheduler for TBT admission ------------------- #
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
            finish = a.last_token_time + a.remaining * _BASE_STEP
            if finish > at_t:
                n += 1
        return n

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
            dt = decode_step_time(len(members), self.profile)
            start = max(self.loop.time(), self.compute_busy[inst])
            step_end = start + dt
            self.compute_busy[inst] = step_end
            await asyncio.sleep(step_end - self.loop.time())
            self._step_complete(inst, members, step_end)

    async def drain(self) -> None:
        """Await all in-flight decode steps so every request finalizes.

        The request coroutines finish at prefill completion, but decode continues
        afterwards on its own step tasks; the driver calls this so the loop keeps
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
