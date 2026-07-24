"""The dedup read-burst scenario on the real TorchStore directory.

This reuses realsim's real-object wiring wholesale -- the real ``LocalClient``
planning core, the real ``Controller`` directory, the real ``InMemoryStore``
behind the fake volume handles, the resource cost model, the deterministic async
engine, and the allocation-free meta/metadata carriers -- and only adds the two
dedup-specific pieces:

* it drives the burst with a :class:`~dedup_sim.policy.DedupPolicy` (a real
  ``ReadPolicy``) instead of realsim's ``NaivePolicy``, and
* it points each reader's real client at a routing handle so the policy can steer
  it to the chosen source (see :mod:`dedup_sim.policy`).

The naive ``m x`` baseline is realsim's own :func:`realsim.scenarios.burst_get.run_burst`
run unchanged, so dedup and naive are compared on byte-for-byte the same topology,
payload, and cost model -- the only difference is the routing policy.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from realsim.scenarios.burst_get import (
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
from sim_common.report import render_tree
from sim_common.trace import Trace

from dedup_sim.policy import DedupPolicy

__all__ = [
    "DEFAULT_N",
    "MODE_META",
    "MODE_METADATA",
    "run_dedup_burst",
    "run_naive_burst",
    "render_dedup_summary",
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

    Builds the real objects via realsim's :func:`~realsim.scenarios.burst_get.build_burst`
    (same wiring as the naive baseline), swaps in a :class:`~dedup_sim.policy.DedupPolicy`,
    and routes each reader's real client through the policy's per-reader directory
    view. Returns a :class:`~realsim.scenarios.burst_get.BurstResult` whose
    ``metrics.fabric_bytes`` is the 1x union (each unique byte crosses the fabric
    once), versus ``m x`` for the naive baseline.
    """
    trace = Trace()
    # put_value is filled in from the built context (ctx["expected"] is exactly the
    # payload carrier the producer put), then read by the policy's read-through.
    policy = DedupPolicy(put_value=None, fanout_cap=fanout_cap)

    scenario_coro, ctx = build_burst(
        num_readers,
        n=n,
        dtype=dtype,
        mode=mode,
        device=device,
        profile=profile,
        compute_device=compute_device,
        policy=policy,
        trace=trace,
        real_directory=real_directory,
    )
    metrics = ctx["metrics"]
    policy._put_value = ctx["expected"]
    # Point each reader's real client at a routing handle so the DedupPolicy can
    # scope its locate to the chosen source (the real directory is untouched).
    for reader in ctx["readers"]:
        policy.install_on(reader, ctx["controller"].handle)

    results, trace = run_sim(scenario_coro(), random_seed=random_seed, trace=trace)
    return BurstResult(
        trace=trace,
        metrics=metrics,
        results=results,
        expected=ctx["expected"],
        origin_id=ctx["origin_id"],
        num_readers=num_readers,
    )


def render_dedup_summary(dedup: BurstResult, naive: BurstResult, cap: int) -> str:
    """Render the dedup-vs-naive fabric summary + the source->dest tree."""
    payload = dedup.expected.numel() * dedup.expected.element_size()
    union = payload  # 1x target: the key crosses the fabric once
    dedup_x = dedup.metrics.fabric_bytes / union if union else 0.0
    naive_x = naive.metrics.fabric_bytes / union if union else 0.0
    saved = naive.metrics.fabric_bytes - dedup.metrics.fabric_bytes
    topo = "chain" if cap == 1 else "tree"
    lines = [
        f"readers: {dedup.num_readers}   payload: {payload}B   "
        f"1x-union target: {union}B   fanout_cap: {cap} ({topo})",
        f"fabric(origin->readers): dedup={dedup.metrics.fabric_bytes}B "
        f"({dedup_x:.1f}x)   naive={naive.metrics.fabric_bytes}B ({naive_x:.1f}x)   "
        f"saved {saved}B",
        f"total delivered (both): dedup={dedup.metrics.total_get_bytes}B   "
        f"naive={naive.metrics.total_get_bytes}B",
        f"wallclock: dedup={dedup.metrics.wallclock:.4f}   "
        f"naive={naive.metrics.wallclock:.4f}   "
        f"(the {topo} trades wallclock for {dedup_x:.0f}x fabric)",
        "source->dest (dedup routes each reader to a peer, not the origin):",
    ]
    for line in render_tree(dedup.metrics.edges):
        lines.append("    " + line)
    return "\n".join(lines)
