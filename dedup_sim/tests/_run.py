"""One dedup configuration, for the tests that assert on a single run.

The demo compares a whole list of :class:`~realsim.run.Run` values at once
(:func:`dedup_sim.workload.scenarios.dedup_vs_baseline`); a test usually wants
exactly one -- the baseline, or one fan-out cap -- so this builds and executes
that one. Test scaffolding, not API.
"""

from __future__ import annotations

from typing import Optional

import torch

from putget_sim.workload.put_get import DEFAULT_N, MODE_META, PutGetBurst
from realsim.run import execute, Result

from ..workload.scenarios import dedup_vs_baseline

__all__ = ["run"]


def run(
    num_readers: int = 3,
    *,
    fanout_cap: Optional[int] = None,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    random_seed: Optional[int] = None,
    real_directory: Optional[bool] = None,
) -> Result:
    """Run one burst: the unrouted baseline, or dedup at ``fanout_cap``.

    The scenario builds both, so the two runs differ by exactly what the demo's
    comparison differs by -- nothing is re-wired here.
    """
    burst = PutGetBurst(num_readers, n=n, dtype=dtype, mode=mode)
    caps = () if fanout_cap is None else (fanout_cap,)
    runs = dedup_vs_baseline(num_readers, caps, burst=burst)
    chosen = runs[0] if fanout_cap is None else runs[1]
    return execute(chosen, random_seed=random_seed, real_directory=real_directory)
