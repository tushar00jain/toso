"""Running one dedup configuration: :func:`run`.

Mirrors :mod:`kvcache_sim.harness` -- the one place this capability wires itself
onto a stack. There is a single run function: with no ``fanout_cap`` it is the
unrouted baseline, with one it is dedup routing. Same scenario either way.

This is the point of the whole exercise. The scenario is ``putget_sim``'s
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

from realsim.entrypoint import run_simulation
from putget_sim.workload.put_get import (
    BurstResult,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_N,
    MODE_META,
    MODE_METADATA,
    PutGetBurst,
)
from sim_common.cost_model import MachineProfile
from sim_common.trace import Trace

from dedup_sim.control.routing import DedupPolicy
from dedup_sim.data.read_through import make_plane

__all__ = ["DEFAULT_N", "MODE_META", "MODE_METADATA", "run"]


def run(
    num_readers: int = 3,
    *,
    fanout_cap: Optional[int] = None,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    device: str = "meta",
    profile: Optional[MachineProfile] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    random_seed: Optional[int] = None,
    real_directory: Optional[bool] = None,
) -> BurstResult:
    """Run one burst end-to-end on a fresh deterministic engine.

    With no ``fanout_cap`` this is the unrouted baseline: every reader locates the
    origin and pulls from it, ``m x`` fabric. With one, the dedup policy is
    installed in the controller and each reader is routed to a peer instead, so
    ``ledger.origin_bytes`` is the 1x union. The scenario is identical either way
    -- the policy is the only difference, which is what makes the comparison mean
    something.
    """
    # A routed run installs the policy in the controller and gives the readers a
    # read-through plane; the baseline passes neither. Same workload either way.
    routed = fanout_cap is not None
    trace = Trace()
    workload = PutGetBurst(
        num_readers,
        n=n,
        dtype=dtype,
        mode=mode,
        device=device,
        profile=profile,
        compute_device=compute_device,
        make_plane=make_plane if routed else None,
    )
    return run_simulation(
        workload,
        policy=DedupPolicy(fanout_cap=fanout_cap, trace=trace) if routed else None,
        profile=workload.profile,
        trace=trace,
        real_directory=real_directory,
        random_seed=random_seed,
    )
