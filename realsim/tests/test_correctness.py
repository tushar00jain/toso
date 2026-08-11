"""Off-sim byte-level correctness baseline for the real store + client path.

The DES scenario runs allocation-free (meta tensors / descriptors), so it can no
longer assert exact reassembled *bytes* -- only shape/dtype/nbytes (see
``test_determinism.py``). This test restores the byte-equality guarantee using
**tiny real CPU tensors** driven through the exact same real TorchStore code the
sim uses (``LocalClient`` planning core + ``Controller`` directory +
``InMemoryStore``), but off the deterministic engine (a plain asyncio loop).

It is deliberately small (a 4x4 tensor) because it allocates real storage; its
job is to prove the real path reassembles the correct bytes for full and sliced
gets, so the sim can trust that path while carrying zero real data.
"""

from __future__ import annotations

import asyncio

import torch

from realsim.adapters.real_client import RealClientAdapter
from realsim.adapters.real_controller import RealControllerAdapter
from realsim.seams.transport import Endpoint
from realsim.seams.volume_handle import LocalVolumeHandle
from realsim.seams.volume_service import VolumeService
from torchstore.transport import TensorSlice


def _build():
    """A controller, two volumes, and producer(vol "1")/consumer(vol "0") clients."""
    controller = RealControllerAdapter()
    volumes = {
        "0": LocalVolumeHandle(VolumeService()),
        "1": LocalVolumeHandle(VolumeService()),
    }
    topology = {
        "0": Endpoint(id="vol0", host="hostA", node="nodeA"),  # consumer
        "1": Endpoint(id="vol1", host="hostB", node="nodeB"),  # producer
    }
    producer = RealClientAdapter(
        controller.handle,
        volume_handles=volumes,
        client_volume_id="1",
        topology=topology,
    )
    consumer = RealClientAdapter(
        controller.handle,
        volume_handles=volumes,
        client_volume_id="0",
        topology=topology,
    )
    return controller, producer, consumer


async def _put_full_get_and_sliced():
    controller, producer, consumer = _build()

    # A tiny REAL CPU tensor with distinct values so byte errors are visible.
    full = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    with producer.installed():
        await producer.client.put("W", full)

    # Full get on the consumer (cross-volume read).
    with consumer.installed():
        got_full = await consumer.client.get("W")

    # Sliced get: rows [1,3), cols [1,3).
    tensor_slice = TensorSlice(
        offsets=(1, 1),
        coordinates=(0,),
        global_shape=(4, 4),
        local_shape=(2, 2),
        mesh_shape=(1,),
    )
    with consumer.installed():
        got_slice = await consumer.client.get("W", tensor_slice_spec=tensor_slice)

    return full, got_full, got_slice


def test_full_get_returns_exact_bytes():
    full, got_full, _ = asyncio.run(_put_full_get_and_sliced())
    assert got_full.shape == full.shape
    assert got_full.dtype == full.dtype
    # Real values must match to the byte.
    assert torch.equal(got_full, full)


def test_sliced_get_returns_exact_bytes():
    full, _, got_slice = asyncio.run(_put_full_get_and_sliced())
    assert got_slice.shape == (2, 2)
    # Exact sub-region, byte for byte.
    assert torch.equal(got_slice, full[1:3, 1:3])
