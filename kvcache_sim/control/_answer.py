"""What this plane answers with: :class:`Plan`, :class:`Response`, and the pair a
ranking over decode hosts is built out of.

Values only, so each crosses a service boundary unchanged. The layer under both the
ranking over decode hosts (:mod:`kvcache_sim.control._selector`) and the plane that
prices and answers with them (:mod:`kvcache_sim.control.scheduler`), so a selector can
be typed on this plane's own values without a cycle back into it. What is decided
*about* is :mod:`kvcache_sim.control.request`; the facts a host reports are with the
sensor they write (:mod:`kvcache_sim.control._sensor`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from proposed import VolumeId

__all__ = ["Plan", "Response"]


@dataclass
class Plan:
    """What prefilling one request on one instance was priced at.

    No order of its own: which of its numbers orders a pool of plans is named where the
    chain is (:meth:`~kvcache_sim.control.scheduler._Scheduler.attach`), so nothing
    compares two of these by accident.

    Which instance this is, and which one decodes, are the two selections' winners and
    live on the :class:`Response`. Every field here is about the prefill, so a losing
    candidate is a complete value too.
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
    pull_keys: List[str] = field(default_factory=list)  # gap blocks to fetch

    @property
    def local_blocks(self) -> int:
        """Blocks the prefill host already held: the match, minus what it pulls.

        Derived, not a field: the data plane reads it three times (reuse to report,
        suffix to publish, prefix to fall back on) and all three must agree.
        """
        return self.match_blocks - len(self.pull_keys)


@dataclass(frozen=True)
class Response:
    """Where one request runs: the winner of each selection, and the price of one.

    The only part of a decision that travels; the rankings behind it stay inside the
    scheduler.

    Args:
        prefill / decode: the two instances chosen, one from each selection.
        plan: what prefilling on ``prefill`` was priced at.
        pred_tbt: the inter-token gap implied by the decode batch this request was
            predicted to meet -- the decode side's, so not on the plan. What the TBT
            SLO bounds (:func:`~kvcache_sim.control.scheduler._tbt_at_most`).
    """

    prefill: VolumeId
    decode: VolumeId
    plan: Plan
    pred_tbt: float = 0.0


