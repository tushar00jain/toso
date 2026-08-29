"""Real-TorchStore simulation adapter and Qwen workload for precomputed routes."""

from __future__ import annotations

import asyncio
import math
from typing import List

import torch
from torchstore.routing.plan import RoutingPlan
from torchstore.routing.service import RoutingService as ProductionRoutingService
from torchstore.transport.types import TensorSlice

from putget_sim.workload.put_get import FLOPS_PER_ELEMENT, KEY, PutGetBurst
from dedup_sim.routing.client import SimulationRoutingClient
from dedup_sim.routing.plans import registrations
from realsim.run import Run, Workload
from realsim.runner import ItemDispatch, WorkItem
from realsim.seams.routing_handle import LocalRoutingServiceHandle
from realsim.seams.routing_service import RoutingService
from sim_common.cost_model import MachineProfile, compute_time
from sim_common.topology import Tier

from ._weight_sync import WeightSync

__all__ = []

_QWEN_BYTES = 55_600_000_000
_ELEMENT_SIZE = 2
_QWEN_ELEMENTS = _QWEN_BYTES // _ELEMENT_SIZE
_TP = 4
_DP = 2
_KEYS = tuple(f"W{rank}" for rank in range(_TP))
# Four weights making up the model, each split across the TP ranks.
_WEIGHT_ELEMENTS = _QWEN_ELEMENTS // len(_KEYS)
_SHARD_ELEMENTS = _WEIGHT_ELEMENTS // _TP


def _shard(generator_index: int, key: str) -> TensorSlice:
    """The piece of ``key`` held by one generator, by its TP and DP rank."""
    del key
    tp_rank, dp_rank = generator_index % _TP, generator_index // _TP
    return TensorSlice(
        offsets=(tp_rank * _SHARD_ELEMENTS,),
        coordinates=(tp_rank, dp_rank),
        global_shape=(_WEIGHT_ELEMENTS,),
        local_shape=(_SHARD_ELEMENTS,),
        mesh_shape=(_TP, _DP),
    )


def _profile() -> MachineProfile:
    # Storage/RAM are free here because 17.5 GB/s is already the measured
    # effective bandwidth including CPU staging. Charging them separately would
    # count the same staging twice.
    return MachineProfile(
        tiers={
            Tier.SHM: (0.0, math.inf),
            Tier.NVLINK: (0.0, 900.0e9),
            Tier.RDMA: (0.0, 17.5e9),
        },
        ram_bandwidth=math.inf,
        storage_read_bw=math.inf,
        storage_write_bw=math.inf,
    )


class _RoutingWeightSync(WeightSync):
    """Qwen3.6-27B as four weights, each TP-sharded four ways with two DP readers.

    A tensor-parallel generator holds a shard of *every* weight, not one whole
    weight, so publisher and requester name the same four keys and differ only
    in geometry. Per-generator bytes are unchanged: a quarter of each of four
    weights is the same quarter of the model.
    """

    tensor_parallel = _TP
    data_parallel = _DP

    def __init__(self) -> None:
        super().__init__(
            num_trainers=1,
            num_generators=_TP * _DP,
            n=_QWEN_ELEMENTS,
            dtype=torch.bfloat16,
            profile=_profile(),
        )
        self.snapshot = {
            key: torch.empty(_WEIGHT_ELEMENTS, dtype=torch.bfloat16, device="meta")
            for key in _KEYS
        }

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return tuple(self.topology[rank].id for rank in self.trainer_ids)

    def snapshots(self):
        return {rank: self.snapshot for rank in self.trainer_ids}

    def routing(self) -> RoutingPlan:
        full = TensorSlice(
            offsets=(0,),
            coordinates=(0,),
            global_shape=(_WEIGHT_ELEMENTS,),
            local_shape=(_WEIGHT_ELEMENTS,),
            mesh_shape=(1,),
        )
        element_sizes = {key: _ELEMENT_SIZE for key in _KEYS}
        publishers = {
            rank: {key: full for key in _KEYS} for rank in self.trainer_ids
        }
        requesters = {
            rank: {key: _shard(index, key) for key in _KEYS}
            for index, rank in enumerate(self.generator_ids)
        }
        return RoutingPlan.build(
            registrations(publishers, element_sizes),
            registrations(requesters, element_sizes),
        )

    def items(self, sim) -> List[WorkItem]:
        return [
            WorkItem(id=rank, release_time=0.0, payload=_KEYS)
            for rank in self.generator_ids
        ]

    async def prepare(self, sim) -> None:
        """Trainer publication is performed by the selected transfer path."""


def _configure_sim(sim, workload) -> None:
    sim.origins(*workload.origin_ids)
    # This workload models one shared trainer egress. Reuse the Mesh's
    # contention registry even when global fidelity is left at its default.
    if isinstance(workload, _RoutingWeightSync) and sim.mesh.registry.mode == "none":
        sim.mesh.registry.mode = "progressive"


class _CurrentWeightSync(_RoutingWeightSync):
    """The same workload using the existing controller-backed client."""

    def items(self, sim) -> List[WorkItem]:
        def _get(rank: str, index: int):
            async def call():
                sim.mesh.bind_source(rank)
                client = sim.mesh.client(rank)
                # Same shards the routed arm fetches, so the two are comparable.
                return [
                    await client.get(key, tensor_slice_spec=_shard(index, key))
                    for key in _KEYS
                ]

            return call

        return [
            WorkItem(id=rank, release_time=0.0, run=_get(rank, index))
            for index, rank in enumerate(self.generator_ids)
        ]

    async def prepare(self, sim) -> None:
        _configure_sim(sim, self)
        with sim.mesh.installed():
            for rank, snapshot in self.snapshots().items():
                sim.mesh.bind_source(rank)
                await sim.mesh.client(rank).put_batch(snapshot)


class _RoutingBurst(Workload):
    """The ordinary dedupe burst expressed as one precomputed routing plan."""

    def __init__(self, burst: PutGetBurst) -> None:
        super().__init__(burst.topology)
        self.source = burst
        self.profile = burst.profile
        self.num_readers = burst.num_readers
        self.reader_ids = tuple(burst.reader_ids)
        self.expected = (
            burst.expected
            if isinstance(burst.expected, torch.Tensor)
            else torch.empty(burst.descriptor.shape, dtype=burst.dtype, device="meta")
        )

    @property
    def payload_bytes(self) -> int:
        return self.source.payload_bytes

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return (self.source.origin_id,)

    def snapshots(self):
        return {"p": {KEY: self.expected}}

    def routing(self) -> RoutingPlan:
        shape = tuple(int(x) for x in self.source.descriptor.shape)
        full = TensorSlice(
            offsets=tuple(0 for _ in shape),
            coordinates=(0,),
            global_shape=shape,
            local_shape=shape,
            mesh_shape=(1,),
        )
        requesters = {
            rank: {
                KEY: TensorSlice(
                    offsets=full.offsets,
                    coordinates=(index,),
                    global_shape=shape,
                    local_shape=shape,
                    mesh_shape=(self.num_readers,),
                )
            }
            for index, rank in enumerate(self.reader_ids)
        }
        element_size = torch.empty(0, dtype=self.source.dtype).element_size()
        sizes = {KEY: element_size}
        return RoutingPlan.build(
            registrations({"p": {KEY: full}}, sizes),
            registrations(requesters, sizes),
        )

    def items(self, sim) -> List[WorkItem]:
        return [
            WorkItem(id=rank, release_time=0.0, payload=KEY) for rank in self.reader_ids
        ]

    async def prepare(self, sim) -> None:
        nbytes = self.source.descriptor.nbytes
        flops = FLOPS_PER_ELEMENT * self.source.descriptor.numel()
        dt = compute_time(
            flops,
            str(self.source.dtype).replace("torch.", ""),
            self.source.compute_device,
            self.profile,
            nbytes,
        )
        await asyncio.sleep(dt)
        sim.trace.record(
            asyncio.get_running_loop().time(),
            "compute",
            f"generate {KEY} flops={flops:g} {nbytes}B "
            f"dev={self.source.compute_device} cost={dt:.4f}",
        )


class _RoutingPlane:
    def __init__(self, sim, workload) -> None:
        self.workload = workload
        self.plan = workload.routing()
        service_handles = {
            rank: LocalRoutingServiceHandle(
                RoutingService(ProductionRoutingService(id_func=lambda r=rank: r))
            )
            for rank in self.plan.ranks
        }
        self.clients = {
            rank: SimulationRoutingClient(
                rank,
                self.plan.for_rank(rank),
                service_handles,
                sim.mesh,
            )
            for rank in self.plan.ranks
        }
        self._publish_task = None
        _configure_sim(sim, workload)

    async def _publish(self) -> None:
        for rank, snapshot in self.workload.snapshots().items():
            await self.clients[rank].put_batch(snapshot)

    async def execute(self, item: WorkItem) -> object:
        if self._publish_task is None:
            self._publish_task = asyncio.create_task(self._publish())
        await self._publish_task
        return await self.clients[item.id].get_batch(list(item.payload))


def _routing_data(workload: Workload):
    def attach(sim) -> ItemDispatch:
        return ItemDispatch(_RoutingPlane(sim, workload).execute)

    return attach


def _routing_runs() -> List[Run]:
    current = _CurrentWeightSync()
    routed = _RoutingWeightSync()
    return [
        Run(
            "direct current path",
            current,
            profile=current.profile,
        ),
        Run(
            "routing",
            routed,
            data=_routing_data(routed),
            profile=routed.profile,
        ),
    ]


def _dedup_routing_run(burst: PutGetBurst) -> Run:
    workload = _RoutingBurst(burst)
    return Run(
        "precomputed",
        workload,
        data=_routing_data(workload),
        profile=workload.profile,
    )
