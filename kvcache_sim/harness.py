"""Running one kvcache configuration: :func:`run`.

The one place both planes are wired together, and the only place kvcache calls
:func:`~realsim.entrypoint.run_simulation`. It is separate from
:mod:`kvcache_sim.workload.scenarios` on purpose: a scenario says *what* to
simulate (a topology, a workload, a set of knobs); this says *how* a run is
assembled from it. Every scenario, and the demo, comes through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from domain import DEFAULT_PROFILE
from proposed import DataPlane, Endpoint
from realsim.entrypoint import run_simulation
from realsim.runner import WorkItem, Workload
from realsim.simulation import Simulation

from .control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from .control.view import KVView
from .data.serving import ServingPlane
from .report.metrics import Metrics, Trace
from .workload.deploy import make_store

__all__ = ["BLOCK_TOKENS", "Run", "run"]

BLOCK_TOKENS = 512


def make_topology(num: int, per_node: int = 2) -> Dict[str, Endpoint]:
    """Build ``num`` instances laid out ``per_node`` per node (distinct hosts).

    Same-node instances exchange KV over NVLink; cross-node over RDMA -- so *where*
    a reusable prefix lives changes the cost of pulling it. The instance id is also
    its storage-volume id in the real directory.
    """
    topo: Dict[str, Endpoint] = {}
    for i in range(num):
        node = f"N{i // per_node}"
        topo[f"s{i}"] = Endpoint(id=f"s{i}", host=f"h{i}", node=node)
    return topo


def subset(topology: Dict[str, Endpoint], ids: List[str]) -> Dict[str, Endpoint]:
    """Return the sub-topology for ``ids`` (order-stable)."""
    return {i: topology[i] for i in ids}


@dataclass
class Run:
    """Output of one scheduler run."""

    metrics: Metrics
    trace: Trace


def run(
    topology: Dict[str, Endpoint],
    requests,
    kind: str,
    *,
    capacity: Optional[int] = None,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    max_batch: int = 8,
    coupled: bool = False,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "off",
) -> Run:
    """Run ``requests`` on ``topology`` under scheduler ``kind``.

    ``kind`` is ``"cache_aware"`` (cache-aware coordinator) or ``"load_balance"``
    (baseline). Returns metrics + trace. The ``simulate_decode`` group of kwargs
    drives the batched-decode / TBT model; ``coupled`` says whether prefill shares
    the decode instances' compute, which is a data-plane fact and so is handed to
    the serving plane, not to the scheduler.
    """
    def build(sim: Simulation) -> Tuple[DataPlane, Workload]:
        """Build both planes onto the assembled stack."""
        store = make_store(sim, block_tokens=BLOCK_TOKENS)
        # Control senses the same real directory the data plane writes, but only
        # ever reads it.
        view = KVView(sim.view.directory, sim.topology)
        common = dict(
            transfer_cost=sim.transfer_cost,
            block_tokens=BLOCK_TOKENS,
            capacity=capacity,
            profile=DEFAULT_PROFILE,
            slo_ttft=slo_ttft,
            slo_tbt=slo_tbt,
            simulate_decode=simulate_decode,
            max_batch=max_batch,
            prefill_pool=prefill_pool,
            decode_pool=decode_pool,
            early_rejection=early_rejection,
        )
        if kind == "cache_aware":
            sched = CacheAwareScheduler(
                view, balance_threshold=balance_threshold,
                replicate=replicate, **common,
            )
        elif kind == "load_balance":
            sched = LoadBalanceScheduler(view, **common)
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown scheduler kind {kind!r}")

        plane = ServingPlane(
            sim.loop, store, sched, trace=sim.trace, metrics=sim.ledger,
            coupled=coupled, max_batch=max_batch,
        )
        return plane, Workload(
            items=[
                WorkItem(id=r.id, release_time=r.arrival, payload=r)
                for r in requests
            ]
        )

    result = run_simulation(
        topology,
        build,
        profile=DEFAULT_PROFILE,
        trace=Trace(time_width=8, kind_width=7),
        ledger=Metrics(),
    )
    return Run(metrics=result.ledger, trace=result.trace)


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #

