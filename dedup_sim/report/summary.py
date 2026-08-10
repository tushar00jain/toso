"""Render the dedup-vs-baseline fabric comparison.

The measurements themselves are a shared
:class:`sim_common.report.Ledger`, filled by the mesh's transfer accounting --
this capability defines none of its own (contrast ``kvcache_sim.report.metrics``,
which owns a whole per-request outcome model). All this module does is turn two
:class:`~realsim.scenarios.put_get.BurstResult` objects into the side-by-side
story: how many times the payload crossed the fabric under each policy, and who
served whom.
"""

from __future__ import annotations

from realsim.scenarios.put_get import BurstResult
from sim_common.report import render_tree

__all__ = ["render_dedup_summary", "render_baseline_summary"]


def render_dedup_summary(dedup: BurstResult, naive: BurstResult, cap: int) -> str:
    """Render the dedup-vs-baseline fabric summary + the source->dest tree."""
    payload = dedup.payload_bytes
    union = payload  # 1x target: the key crosses the fabric once
    dedup_x = dedup.ledger.origin_bytes / union if union else 0.0
    naive_x = naive.ledger.origin_bytes / union if union else 0.0
    saved = naive.ledger.origin_bytes - dedup.ledger.origin_bytes
    topo = "chain" if cap == 1 else "tree"
    lines = [
        f"readers: {dedup.num_readers}   payload: {payload}B   "
        f"1x-union target: {union}B   fanout_cap: {cap} ({topo})",
        f"fabric(origin->readers): dedup={dedup.ledger.origin_bytes}B "
        f"({dedup_x:.1f}x)   naive={naive.ledger.origin_bytes}B ({naive_x:.1f}x)   "
        f"saved {saved}B",
        f"total delivered (both): dedup={dedup.ledger.transfer_bytes}B   "
        f"naive={naive.ledger.transfer_bytes}B",
        f"wallclock: dedup={dedup.ledger.wallclock:.4f}   "
        f"naive={naive.ledger.wallclock:.4f}   "
        f"(the {topo} trades wallclock for {dedup_x:.0f}x fabric)",
        "source->dest (dedup routes each reader to a peer, not the origin):",
    ]
    for line in render_tree(dedup.ledger.edges):
        lines.append("    " + line)
    return "\n".join(lines)


def render_baseline_summary(naive: BurstResult, num_readers: int) -> str:
    """Render the unrouted baseline's own fabric summary.

    The counterpart to :func:`render_dedup_summary`: what the same burst costs
    with no policy installed, which is the number dedup is measured against.
    """
    payload = naive.payload_bytes
    return "\n".join([
        f"fabric(origin->readers): naive={naive.ledger.origin_bytes}B "
        f"({naive.ledger.origin_bytes / payload:.1f}x)   "
        f"wallclock={naive.ledger.wallclock:.4f}",
        f"every reader pulls the full payload cross-node -> m x fabric; "
        f"concurrent so it wins wallclock, but pays {num_readers}x the bytes.",
    ])
