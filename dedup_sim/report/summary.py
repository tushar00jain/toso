"""Render the dedup-vs-naive fabric comparison.

The metrics themselves are realsim's
:class:`~realsim.coordinator.model.BurstMetrics`, filled by the coordinator's
transfer accounting -- this capability defines none of its own (contrast
``kvcache_sim.report.metrics``, which owns a whole per-request outcome model). All
this module does is turn two :class:`~realsim.scenarios.burst_get.BurstResult`
objects into the side-by-side story: how many times the payload crossed the
fabric under each policy, and who served whom.
"""

from __future__ import annotations

from realsim.scenarios.burst_get import BurstResult
from sim_common.report import render_tree

__all__ = ["render_dedup_summary"]


def render_dedup_summary(dedup: BurstResult, naive: BurstResult, cap: int) -> str:
    """Render the dedup-vs-naive fabric summary + the source->dest tree."""
    payload = dedup.expected.numel() * dedup.expected.element_size()
    union = payload  # 1x target: the key crosses the fabric once
    dedup_x = dedup.metrics.fabric_bytes / union if union else 0.0
    naive_x = naive.metrics.fabric_bytes / union if union else 0.0
    saved = naive.metrics.fabric_bytes - dedup.metrics.fabric_bytes
    topo = "chain" if cap == 1 else "tree"
    lines = [
        f"readers: {dedup.num_readers}   payload: {payload}B   "
        f"1x-union target: {union}B   fanout_cap: {cap} ({topo})",
        f"fabric(origin->readers): dedup={dedup.metrics.fabric_bytes}B "
        f"({dedup_x:.1f}x)   naive={naive.metrics.fabric_bytes}B ({naive_x:.1f}x)   "
        f"saved {saved}B",
        f"total delivered (both): dedup={dedup.metrics.total_get_bytes}B   "
        f"naive={naive.metrics.total_get_bytes}B",
        f"wallclock: dedup={dedup.metrics.wallclock:.4f}   "
        f"naive={naive.metrics.wallclock:.4f}   "
        f"(the {topo} trades wallclock for {dedup_x:.0f}x fabric)",
        "source->dest (dedup routes each reader to a peer, not the origin):",
    ]
    for line in render_tree(dedup.metrics.edges):
        lines.append("    " + line)
    return "\n".join(lines)
