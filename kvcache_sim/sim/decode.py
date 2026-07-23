"""Batched decode engine (K6) -- the piece that makes TBT real.

The prefill path answers "how fast is the first token" (TTFT); this module answers
"how fast is every *subsequent* token" (TBT, time-between-tokens). A serving
instance decodes a **batch** of requests together: each decode *step* emits one
token for every request in the batch, and the step's wall time is the TBT every
batched request sees for that token. Step time rises with batch size
(:func:`sim.cost.decode_step_time`), so TBT degrades as an instance fills up --
exactly the tension a TBT SLO bounds (Mooncake §4.2).

Two levers the design cares about are modelled here:

* **VRAM cap** (``max_batch``): aggregated KVCache is bounded, so a batch can only
  grow so far. Requests over the cap queue (``pending``) and their wait counts
  against their TBT -- i.e. VRAM pressure shows up as TBT violations, which is why
  the scheduler must admit against a *predicted* batch, not just accept blindly.
* **Prefill/decode coupling**: when prefill and decode share an instance's compute
  (``compute_busy`` aliased to the scheduler's prefill timeline), a long prefill
  delays the next decode step, spiking that token's TBT. Disaggregation gives the
  decode engine its own ``compute_busy`` so prefill can never stall decode -- the
  central Mooncake result.

Single-threaded and deterministic: steps are ``Sim`` events ordered by
``(time, seq)``; recency/clocks are integer counters. No wall-clock, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from sim_common.engine import Sim
from .cost import decode_step_time, TBT_BASE
from .model import Request


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
    """

    def __init__(self, sim: Sim, decode_ids: List[str], *, max_batch: int,
                 compute_busy: Optional[Dict[str, float]] = None,
                 on_finish: Optional[Callable[[Request, float], None]] = None
                 ) -> None:
        self.sim = sim
        self.ids = sorted(decode_ids)
        self.max_batch = max_batch
        self.compute_busy: Dict[str, float] = (
            compute_busy if compute_busy is not None
            else {i: 0.0 for i in self.ids}
        )
        for i in self.ids:
            self.compute_busy.setdefault(i, 0.0)
        self.on_finish = on_finish
        self.batch: Dict[str, List[_Active]] = {i: [] for i in self.ids}
        self.pending: Dict[str, List[_Active]] = {i: [] for i in self.ids}
        self._stepping: Dict[str, bool] = {i: False for i in self.ids}

    # -- queries used by the scheduler for TBT admission ------------------- #
    def occupancy(self, inst: str) -> int:
        """Requests currently decoding or queued on ``inst`` (the live batch load)."""
        return len(self.batch[inst]) + len(self.pending[inst])

    def predict_occupancy(self, inst: str, at_t: float) -> int:
        """Estimate how many current occupants are *still* decoding at ``at_t``.

        Uses the uniform per-token assumption (each remaining token ~ ``TBT_BASE``),
        matching Mooncake's decode-load prediction: a request is still resident at
        ``at_t`` if its estimated finish time is past ``at_t``.
        """
        n = 0
        for a in self.batch[inst] + self.pending[inst]:
            finish = a.last_token_time + a.remaining * TBT_BASE
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
        a = _Active(request=request, remaining=remaining,
                    last_token_time=self.sim.now)
        if len(self.batch[inst]) < self.max_batch:
            self.batch[inst].append(a)
        else:
            self.pending[inst].append(a)  # VRAM full: wait counts against TBT
        self._schedule_step(inst)

    def _schedule_step(self, inst: str) -> None:
        """Schedule the next decode step on ``inst`` unless one is already pending."""
        if self._stepping[inst] or not self.batch[inst]:
            return
        members = list(self.batch[inst])          # frozen for this step
        dt = decode_step_time(len(members))
        start = max(self.sim.now, self.compute_busy[inst])
        step_end = start + dt
        self.compute_busy[inst] = step_end
        self._stepping[inst] = True
        self.sim.schedule(
            step_end - self.sim.now,
            lambda: self._step_complete(inst, members, step_end),
            label=f"decode_step:{inst}",
        )

    def _step_complete(self, inst: str, members: List[_Active],
                       step_end: float) -> None:
        """Emit one token for each member; retire finished; promote pending."""
        self._stepping[inst] = False
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
        self._schedule_step(inst)
