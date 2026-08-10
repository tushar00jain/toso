"""Running one kvcache configuration: :func:`run`.

The only place kvcache calls :func:`~realsim.entrypoint.run_simulation`. Every
scenario and the demo come through here. What to simulate is
:class:`~kvcache_sim.workload.serving.KVWorkload`; this only says which trace and
ledger a run reports into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from domain import DEFAULT_PROFILE
from proposed import Endpoint
from realsim.entrypoint import Result, run_simulation

from .report.metrics import Metrics, Trace
from .workload.serving import BLOCK_TOKENS, KVWorkload

__all__ = ["BLOCK_TOKENS", "run"]

def run(topology: Dict[str, Endpoint], requests, kind: str, **knobs) -> Result:
    """Run ``requests`` on ``topology`` under scheduler ``kind``."""
    return run_simulation(
        KVWorkload(topology, requests, kind, **knobs),
        profile=DEFAULT_PROFILE,
        trace=Trace(time_width=8, kind_width=7),
        ledger=Metrics(),
    )


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #

