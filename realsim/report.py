"""Rendering a burst outcome, like each capability's ``report/``."""

from __future__ import annotations

from sim_common.report import render_tree

from realsim.harness import BurstResult

__all__ = ["render_burst_summary"]


def render_burst_summary(res: BurstResult) -> str:
    """Render the fabric/wallclock summary + the source->dest tree."""
    payload = res.payload_bytes
    union = payload  # the 1x target: W crosses the fabric once
    fabric_x = res.ledger.origin_bytes / union if union else 0.0
    lines = [
        f"readers: {res.num_readers}   payload(W): {payload}B   "
        f"1x-union target: {union}B",
        f"fabric(origin->readers): {res.ledger.origin_bytes}B ({fabric_x:.1f}x)   "
        f"total delivered: {res.ledger.transfer_bytes}B",
        f"wallclock: {res.ledger.wallclock:.4f}   "
        f"readers done: {res.ledger.items_done}/{res.ledger.items_total}",
        "source->dest (unrouted: every reader pulls the origin):",
    ]
    for line in render_tree(res.ledger.edges):
        lines.append("    " + line)
    return "\n".join(lines)
