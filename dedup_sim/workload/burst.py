"""The dedup burst: the same scenario as the baseline, with a policy installed.

This is the point of the whole exercise. The scenario is ``realsim``'s own
put/get fixture -- ordinary user code: seed ``W``, then a gather of
``client.get(W)``. Running it unchanged gives the ``m x`` baseline. Running it
with :class:`~dedup_sim.control.routing.DedupPolicy` and the read-through
:class:`~dedup_sim.data.read_through.ReadThroughPlane` gives 1x.

Same topology, same payload, same cost model, same client calls -- the *only*
difference between the two runs is the policy, which is what makes the
comparison mean something.
"""

from __future__ import annotations

from typing import Optional

import torch

from realsim.scenarios.put_get import (
    build_burst,
    BurstResult,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_N,
    MODE_META,
    MODE_METADATA,
    run_burst as run_naive_burst,
)
from sim_common.async_engine import run_sim
from sim_common.cost_model import MachineProfile
from sim_common.report import Ledger
from sim_common.trace import Trace

from dedup_sim.control.routing import DedupPolicy
from dedup_sim.data.read_through import make_plane

__all__ = [
    "DEFAULT_N",
    "MODE_META",
    "MODE_METADATA",
    "run_dedup_burst",
    "run_naive_burst",
]


def run_dedup_burst(
    num_readers: int = 3,
    *,
    fanout_cap: int = 1,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    device: str = "meta",
    profile: Optional[MachineProfile] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    random_seed: Optional[int] = None,
    real_directory: Optional[bool] = None,
) -> BurstResult:
    """Run one dedup burst end-to-end on a fresh deterministic engine.

    Returns a :class:`~realsim.scenarios.put_get.BurstResult` whose
    ``ledger.origin_bytes`` is the 1x union (each unique byte crosses the fabric
    once), versus ``m x`` for :func:`run_naive_burst`.
    """
    trace = Trace()
    sim, ctx = build_burst(
        num_readers,
        n=n,
        dtype=dtype,
        mode=mode,
        device=device,
        profile=profile,
        compute_device=compute_device,
        policy=DedupPolicy(fanout_cap=fanout_cap, trace=trace),
        make_plane=make_plane,
        trace=trace,
        random_seed=random_seed,
        real_directory=real_directory,
    )
    results = sim.run(ctx["items"], plane=ctx["plane"], before=ctx["seed"])
    return BurstResult(
        trace=sim.trace,
        ledger=sim.ledger,
        results=results,
        expected=ctx["expected"],
        origin_id=ctx["origin_id"],
        num_readers=num_readers,
    )
