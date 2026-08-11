"""What this sim runs: one unrouted burst.

The degenerate comparison -- a single :class:`~realsim.run.Run` with no policy
and no data plane. That absence is the content: it is the ``m x`` baseline
``dedup_sim`` measures its 1x against, and ``dedup_sim`` builds its own runs over
this same :class:`~putget_sim.workload.put_get.PutGetBurst`.
"""

from __future__ import annotations

from typing import List, Optional

from realsim.run import Run
from sim_common.cost_model import MachineProfile

from .put_get import DEFAULT_N, MODE_META, PutGetBurst

__all__ = ["NUM_READERS", "burst"]

#: Readers in the burst, when the CLI does not say otherwise.
NUM_READERS = 3


def burst(
    num_readers: int = NUM_READERS,
    *,
    n: int = DEFAULT_N,
    mode: str = MODE_META,
    profile: Optional[MachineProfile] = None,
) -> List[Run]:
    """The one run: ``num_readers`` readers get W, with nothing installed."""
    workload = PutGetBurst(num_readers, n=n, mode=mode, profile=profile)
    return [Run("unrouted", workload, profile=workload.profile)]
