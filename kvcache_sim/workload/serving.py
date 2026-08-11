"""The kvcache workload: a request stream served by one scheduler.

What is simulated, in the shape :func:`~realsim.entrypoint.run_simulation` takes.
:mod:`kvcache_sim.harness` runs it; :mod:`kvcache_sim.workload.scenarios` chooses
the knobs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from domain import DEFAULT_MODEL, DEFAULT_PROFILE, Model
from proposed import DataPlane, Endpoint
from realsim.entrypoint import Workload
from realsim.runner import WorkItem
from realsim.seams.transport import TensorDescriptor
from realsim.simulation import Simulation

from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ..control.view import KVView
from ..data.serving import ServingPlane
from ..data.store import KVStore

#: Tokens per KV block. Fixed for every scenario so runs stay comparable.
BLOCK_TOKENS = 512

__all__ = ["BLOCK_TOKENS", "KVWorkload", "sim_block_carrier"]


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
    """A request stream served by one scheduler over a set of instances.

    ``kind`` is ``"cache_aware"`` (the coordinator under test) or
    ``"load_balance"`` (the baseline). ``coupled`` says whether prefill shares the
    decode instances' compute -- a deployment fact, so it goes to the serving
    plane, not the scheduler.
    """

    def __init__(
        self,
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
    ) -> None:
        if kind not in ("cache_aware", "load_balance"):
            raise ValueError(f"unknown scheduler kind {kind!r}")
        self.topology = topology
        self.requests = requests
        self.kind = kind
        self.coupled = coupled
        self.max_batch = max_batch
        self.balance_threshold = balance_threshold
        self.replicate = replicate
        self.knobs = dict(
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

    def build(self, sim: Simulation) -> Tuple[DataPlane, List[WorkItem]]:
        """Build both planes onto the assembled stack."""
        # The simulation *is* the deployment: it vends the client for an instance
        # and holds the directory. All the run adds is the block carrier.
        store = KVStore.for_deployment(
            sim.mesh, block_tokens=BLOCK_TOKENS, carrier=sim_block_carrier()
        )
        # Control senses the same real directory the data plane writes, but only
        # ever reads it.
        view = KVView(sim.view.directory, sim.topology)
        common = dict(transfer_cost=sim.transfer_cost, **self.knobs)
        if self.kind == "cache_aware":
            sched = CacheAwareScheduler(
                view,
                balance_threshold=self.balance_threshold,
                replicate=self.replicate,
                **common,
            )
        else:
            sched = LoadBalanceScheduler(view, **common)

        plane = ServingPlane(
            sim.loop, store, sched, trace=sim.trace, metrics=sim.ledger,
            coupled=self.coupled, max_batch=self.max_batch,
        )
        items = [
            WorkItem(id=r.id, release_time=r.arrival, payload=r)
            for r in self.requests
        ]
        return plane, items
