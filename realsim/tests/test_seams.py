"""Smoke test: a real put + real full get + real sliced get round-trip.

The whole path is real TorchStore code driven off-actor via the in-process seams:

    producer.client.put(...)   -> real LocalClient.put_batch
                               -> real InMemoryStore.put (via InMemoryTransport)
                               -> real Controller._notify_put (via LocalControllerHandle)

    consumer.client.get(...)   -> real LocalClient._fetch / _build_volume_requests
                                  / _assemble_results / _apply_inplace
                               -> real Controller.locate_volumes body (mirrored)
                               -> real InMemoryStore.get (full + sliced)

A plain asyncio loop is fine here (the deterministic engine is a later task).
The producer lives on volume "1" and the consumer on volume "0", so every
consumer read is a cross-volume (RDMA) transfer with a non-zero cost.
"""

from __future__ import annotations

import asyncio

import torch

from realsim.adapters.real_client import RealClientAdapter
from realsim.adapters.real_controller import RealControllerAdapter
from realsim.seams.transport import Endpoint
from realsim.seams.volume_handle import FakeVolumeHandle
from sim_common.trace import Trace
from torchstore.transport import TensorSlice


def _build():
    """Build a controller, two volumes, and producer/consumer client adapters."""
    controller = RealControllerAdapter()

    volumes = {"0": FakeVolumeHandle(), "1": FakeVolumeHandle()}
    topology = {
        "0": Endpoint(id="vol0", host="hostA", node="nodeA"),  # consumer's volume
        "1": Endpoint(id="vol1", host="hostB", node="nodeB"),  # producer's volume
    }
    trace = Trace()

    producer = RealClientAdapter(
        controller.handle,
        volume_handles=volumes,
        client_volume_id="1",
        topology=topology,
        trace=trace,
    )
    consumer = RealClientAdapter(
        controller.handle,
        volume_handles=volumes,
        client_volume_id="0",
        topology=topology,
        trace=trace,
    )
    return controller, producer, consumer, trace


async def _roundtrip():
    controller, producer, consumer, trace = _build()

    full = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    # PUT on the producer (volume "1").
    with producer.installed():
        await producer.client.put("W", full)

    # Directory now knows the key, on the producer's volume only.
    assert await controller.handle.keys.call_one() == ["W"]
    located = await controller.handle.locate_volumes.call_one(["W"])
    assert set(located["W"]) == {"1"}

    # Full GET on the consumer (volume "0") -> cross-volume read.
    with consumer.installed():
        got_full = await consumer.client.get("W")

    # Sliced GET on the consumer: rows [1,3), cols [1,3).
    tensor_slice = TensorSlice(
        offsets=(1, 1),
        coordinates=(0,),
        global_shape=(4, 4),
        local_shape=(2, 2),
        mesh_shape=(1,),
    )
    with consumer.installed():
        got_slice = await consumer.client.get("W", tensor_slice_spec=tensor_slice)

    return full, got_full, got_slice, trace


def test_put_full_get_and_sliced_get_roundtrip():
    full, got_full, got_slice, trace = asyncio.run(_roundtrip())

    # Full get returns the whole tensor.
    assert torch.equal(got_full, full)

    # Sliced get returns exactly the requested sub-region.
    assert got_slice.shape == (2, 2)
    assert torch.equal(got_slice, full[1:3, 1:3])

    # The cross-volume consumer reads incurred a non-zero transfer cost, and the
    # transport recorded them into the shared trace.
    xfers = [e for e in trace.events if e[1] == "xfer"]
    get_xfers = [e for e in xfers if e[2].startswith("get")]
    assert len(get_xfers) >= 2  # full get + sliced get
    assert all("cost=" in e[2] for e in get_xfers)
    # vol0 (consumer) <- vol1 (producer): cross-node, so cost must be > 0.
    assert all("vol1->vol0" in e[2] for e in get_xfers)
