"""The dedup scenarios: what each run simulates.

Mirrors :mod:`kvcache_sim.workload.scenarios` -- a scenario fixes the choices (how
many readers, which fan-out caps) and hands them to :func:`dedup_sim.harness.run`,
which does the assembling. Nothing here renders or wires; ``__main__`` renders,
the harness wires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from realsim.harness import BurstResult

from ..harness import run

__all__ = ["NUM_READERS", "FANOUT_CAPS", "Comparison", "run_dedup_vs_baseline"]

#: Readers in the burst. Three is enough to show a chain and a shallow tree.
NUM_READERS = 3
#: The routed configurations to compare: cap 1 is a chain, cap 2 a shallow tree.
FANOUT_CAPS = (1, 2)


@dataclass
class Comparison:
    """One unrouted baseline against one routed run per fan-out cap."""

    baseline: BurstResult
    routed: List[Tuple[int, BurstResult]]
    num_readers: int

    @property
    def payload_bytes(self) -> int:
        """Bytes of the payload -- the 1x union a routed run drives fabric toward."""
        return self.baseline.payload_bytes


def run_dedup_vs_baseline(
    num_readers: int = NUM_READERS, caps: Sequence[int] = FANOUT_CAPS
) -> Comparison:
    """Run the same burst unrouted, then once per fan-out cap with the policy."""
    return Comparison(
        baseline=run(num_readers=num_readers),
        routed=[(cap, run(num_readers=num_readers, fanout_cap=cap)) for cap in caps],
        num_readers=num_readers,
    )
