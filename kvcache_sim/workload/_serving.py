"""The kvcache workload, and the wiring a run installs around it.

Two separate things, deliberately:

* :class:`KVWorkload` is *the work* -- a request stream, one
  :class:`~realsim.runner.WorkItem` per request at its arrival time. It builds no
  store, no scheduler and no plane;
* :func:`serving_plane` is the *capability wiring* -- the store, the view, the
  scheduler and the :class:`~kvcache_sim.data.serving.ServingPlane` over them.
  It is a factory because a plane reaches for the clock, the mesh and the ledger,
  none of which exist before the stack does.

A scenario pairs them on a :class:`~realsim.run.Run`: same workload, different
wiring, which is exactly what "cache-aware vs load-balance" means.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from domain import DEFAULT_MODEL, DEFAULT_PROFILE, Model
from proposed import Endpoint
from realsim.runner import WorkItem
from realsim.seams.transport import TensorDescriptor
from realsim.simulation import Simulation
from realsim.workload import Workload

from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ..control.view import KVView
from ..data.serving import ServingPlane
from ..data.store import KVStore

#: Tokens per KV block. Fixed for every scenario so runs stay comparable.
BLOCK_TOKENS = 512

__all__ = ["BLOCK_TOKENS", "KVWorkload", "serving_plane", "sim_block_carrier"]


def sim_block_carrier(
    block_tokens: int = BLOCK_TOKENS, model: Model = DEFAULT_MODEL
):
    """What one KV block is stored as **under simulation**.

    A metadata-only carrier: a uint8 descriptor whose length *is* the block's
    modeled byte size, so the bytes the transport charges cannot drift from the
    bytes the scheduler predicted. Zero real storage.

    This is the one piece a real deployment chooses differently -- it stores the
    KV tensors -- which is why it lives with the run rather than in
    :mod:`kvcache_sim.data.store`.
    """
    return TensorDescriptor(
        shape=(model.block_bytes(1, block_tokens),), dtype=torch.uint8
    )


class KVWorkload(Workload):
    """A request stream over a set of serving instances.

    One work item per request, released at its arrival time. Which scheduler
    serves them is not this object's business -- see :func:`serving_plane`.
    """

    def __init__(self, topology: Dict[str, Endpoint], requests) -> None:
        super().__init__(topology)
        self.requests = requests

    def items(self, sim: Simulation) -> List[WorkItem]:
        """One item per request; the serving plane runs each one's lifecycle."""
        return [
            WorkItem(id=r.id, release_time=r.arrival, payload=r)
            for r in self.requests
        ]


def serving_plane(
    kind: str,
    *,
    coupled: bool = False,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    capacity: Optional[int] = None,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    max_batch: int = 8,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "off",
) -> Callable[[Simulation], ServingPlane]:
    """Build the factory that wires this capability onto an assembled stack.

    ``kind`` is ``"cache_aware"`` (the coordinator under test) or
    ``"load_balance"`` (the baseline). ``coupled`` says whether prefill shares the
    decode instances' compute -- a deployment fact, so it goes to the serving
    plane, not the scheduler.
    """
    if kind not in ("cache_aware", "load_balance"):
        raise ValueError(f"unknown scheduler kind {kind!r}")
    knobs = dict(
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

    def build(sim: Simulation) -> ServingPlane:
        # The simulation *is* the deployment: it vends the client for an instance
        # and holds the directory. All the run adds is the block carrier.
        store = KVStore.for_deployment(
            sim.mesh, block_tokens=BLOCK_TOKENS, carrier=sim_block_carrier()
        )
        # Control senses the same real directory the data plane writes, but only
        # ever reads it.
        view = KVView(sim.view.directory, sim.topology)
        common = dict(transfer_cost=sim.transfer_cost, **knobs)
        if kind == "cache_aware":
            sched = CacheAwareScheduler(
                view,
                balance_threshold=balance_threshold,
                replicate=replicate,
                **common,
            )
        else:
            sched = LoadBalanceScheduler(view, **common)
        return ServingPlane(
            sim.loop, store, sched, trace=sim.trace, metrics=sim.ledger,
            coupled=coupled, max_batch=max_batch,
        )

    return build
