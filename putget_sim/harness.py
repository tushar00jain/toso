"""Running one unrouted burst: :func:`run_burst`.

Mirrors :mod:`dedup_sim.harness` and :mod:`kvcache_sim.harness` -- the one place
this capability wires itself onto a stack. :mod:`putget_sim.workload.put_get`
says *what* is simulated (a topology, a payload, m readers), this says *how* a
run of it is assembled, and :mod:`putget_sim.report.summary` renders the outcome.

There is no policy and no data plane unless a caller passes one, so the default
run is the ``m x`` baseline ``dedup_sim`` measures against.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from proposed import Policy
from sim_common.cost_model import MachineProfile
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.entrypoint import run_simulation
from putget_sim.workload.put_get import (
    BurstResult,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_N,
    MakePlane,
    MODE_META,
    PutGetBurst,
)

__all__ = ["run_burst"]


def run_burst(
    num_readers: int = 3,
    *,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    device: str = "meta",
    profile: Optional[MachineProfile] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    policy: Optional[Policy] = None,
    make_plane: Optional[MakePlane] = None,
    trace: Optional[Trace] = None,
    ledger: Optional[Ledger] = None,
    random_seed: Optional[int] = None,
    real_directory: Optional[bool] = None,
) -> BurstResult:
    """Run one burst end-to-end on a fresh deterministic engine."""
    workload = PutGetBurst(
        num_readers,
        n=n,
        dtype=dtype,
        mode=mode,
        device=device,
        profile=profile,
        compute_device=compute_device,
        make_plane=make_plane,
    )
    return run_simulation(
        workload,
        policy=policy,
        profile=workload.profile,
        trace=trace,
        ledger=ledger,
        real_directory=real_directory,
        random_seed=random_seed,
    )
