"""Render the dedup-vs-baseline fabric comparison.

The measurements are a shared :class:`sim_common.report.Ledger` filled by the
mesh's transfer accounting; this capability defines none of its own. All these
reports do is turn the runs' :class:`~realsim.run.Result` objects into the
side-by-side story: how many times the payload crossed the fabric under each
selector, and who served whom.
"""

from __future__ import annotations

from realsim.run import Report, Result
from sim_common.report import render_tree

__all__ = ["DedupReport", "BaselineReport"]


class DedupReport(Report):
    """One routed configuration against the unrouted baseline."""

    def __init__(self, dedup: Result, naive: Result, cap: int) -> None:
        self.dedup = dedup
        self.naive = naive
        self.cap = cap

    def render(self) -> str:
        dedup, naive, cap = self.dedup, self.naive, self.cap
        payload = dedup.workload.payload_bytes
        union = payload  # 1x target: the key crosses the fabric once
        dedup_x = dedup.ledger.origin_bytes / union if union else 0.0
        naive_x = naive.ledger.origin_bytes / union if union else 0.0
        saved = naive.ledger.origin_bytes - dedup.ledger.origin_bytes
        topo = "chain" if cap == 1 else "tree"
        lines = [
            f"readers: {dedup.workload.num_readers}   payload: {payload}B   "
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


class BaselineReport(Report):
    """The unrouted baseline's own fabric summary.

    The counterpart to :class:`DedupReport`: what the same burst costs with no
    selector installed, which is the number dedup is measured against.
    """

    def __init__(self, naive: Result) -> None:
        self.naive = naive

    def render(self) -> str:
        naive = self.naive
        payload = naive.workload.payload_bytes
        return "\n".join([
            f"fabric(origin->readers): naive={naive.ledger.origin_bytes}B "
            f"({naive.ledger.origin_bytes / payload:.1f}x)   "
            f"wallclock={naive.ledger.wallclock:.4f}",
            f"every reader pulls the full payload cross-node -> m x fabric; "
            f"concurrent so it wins wallclock, but pays "
            f"{naive.workload.num_readers}x the bytes.",
        ])
