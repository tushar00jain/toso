"""Rendering a burst outcome, like each other capability's ``report/``."""

from __future__ import annotations

from realsim.run import Report, Result
from sim_common.report import render_tree

__all__ = ["BurstReport"]


class BurstReport(Report):
    """The fabric/wallclock summary plus the source->dest tree.

    Reads the payload facts off the run's workload rather than off a result
    subclass: the burst already knows how big W is and how many readers wanted
    it, so nothing needs copying.
    """

    def __init__(self, result: Result) -> None:
        self.result = result

    def render(self) -> str:
        res, burst = self.result, self.result.workload
        ledger = res.ledger
        payload = burst.payload_bytes
        union = payload  # the 1x target: W crosses the fabric once
        fabric_x = ledger.origin_bytes / union if union else 0.0
        lines = [
            f"readers: {burst.num_readers}   payload(W): {payload}B   "
            f"1x-union target: {union}B",
            f"fabric(origin->readers): {ledger.origin_bytes}B ({fabric_x:.1f}x)   "
            f"total delivered: {ledger.transfer_bytes}B",
            f"wallclock: {ledger.wallclock:.4f}   "
            f"readers done: {ledger.items_done}/{ledger.items_total}",
            "source->dest (unrouted: every reader pulls the origin):",
        ]
        for line in render_tree(ledger.edges):
            lines.append("    " + line)
        return "\n".join(lines)
