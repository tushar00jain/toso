"""What this plane answers with: :class:`Plan`, :class:`Response`, and the candidate
pair a ranking over either is built out of.

Values only, so each crosses a service boundary unchanged, and the layer under both
the rankings that sort them (:mod:`kvcache_sim.control._selector`) and the plane that
prices and answers with them (:mod:`kvcache_sim.control.scheduler`) -- which is what
lets a selector be typed on a plan without a cycle back into the plane that builds
one. What is decided *about* is :mod:`kvcache_sim.control.request`; the facts a host
reports are with the sensor they write (:mod:`kvcache_sim.control._sensor`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from proposed import VolumeId

__all__ = ["Plan", "Response", "Priced", "Batched"]


@dataclass
class Plan:
    """What prefilling one request on one instance was priced at.

    One candidate's price and nothing else: which instance this is, and which one
    decodes, are the two selections' winners and live on the :class:`Response`. Every
    field here is about the prefill, so a losing candidate is a complete value too.
    """

    match_blocks: int            # reused prefix length (blocks)
    cached_tokens: int
    uncached_tokens: int
    reuse_source: Optional[str]  # remote instance a prefix gap is pulled from
    transfer_bytes: int
    queue_wait: float
    ttft: float                  # time-to-first-token (queue + transfer + prefill)
    done_time: float             # absolute sim time prefill completes
    prefill_t: float = 0.0       # prefill compute duration
    transfer_t: float = 0.0      # predicted remote-pull fetch duration
    pull_keys: List[str] = field(default_factory=list)  # gap blocks to fetch

    @property
    def local_blocks(self) -> int:
        """Blocks the prefill host already held: the match, minus what it pulls.

        Derived rather than a field: the data plane needs it three times over (reuse
        to report, suffix to publish, prefix to fall back on when a planned pull is
        gone) and all three have to agree.
        """
        return self.match_blocks - len(self.pull_keys)


@dataclass(frozen=True)
class Response:
    """Where one request runs: the winner of each selection, and the price of one.

    What :meth:`~kvcache_sim.control.scheduler._Scheduler.decide` answers and the only
    part of a decision that travels. The rankings behind it stay inside the scheduler:
    nothing outside asks what lost.

    Args:
        prefill / decode: the two instances chosen, one from each selection.
        plan: what prefilling on ``prefill`` was priced at.
        pred_batch / pred_tbt: the decode batch this request was predicted to meet
            and the inter-token gap that implies. What the TBT SLO is applied to
            (:meth:`~kvcache_sim.control.scheduler._Scheduler._admit`), which is why
            they are here and not on the plan -- they are the decode side's.
    """

    prefill: VolumeId
    decode: VolumeId
    plan: Plan
    pred_batch: int = 0
    pred_tbt: float = 0.0


#: One candidate as the pair a :class:`~proposed.selector.Selection` is built out of:
#: the instance, and what the scheduler priced it at. A :class:`Plan` when the choice
#: is which host prefills, a predicted batch size when it is which host decodes.
Priced = Tuple[VolumeId, Plan]
Batched = Tuple[VolumeId, int]
