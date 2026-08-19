"""Torchstore surfaces required by the simulator."""

from __future__ import annotations

import inspect

from proposed import Controller as ControllerProtocol
from realsim.adapters.real_controller import RealControllerAdapter
from torchstore.client import LocalClient
from torchstore.controller import Controller, ObjectType, StorageInfo
from torchstore.transport import Request, TensorSlice


def test_controller_exposes_publication_and_preference_parameters():
    locate = inspect.signature(Controller.locate_volumes._method)
    put = inspect.signature(Controller.notify_put_batch._method)
    delete = inspect.signature(Controller.notify_delete_batch._method)

    assert "prefer" in locate.parameters
    assert put.parameters["pending"].default is True
    assert delete.parameters["pub"].default is None
    assert hasattr(Controller, "serving_union")
    assert hasattr(Controller, "_locate")
    assert hasattr(Controller, "_notify_put")


def test_proposed_controller_declares_the_real_surface():
    required = {
        "_locate",
        "_notify_put",
        "keys",
        "locate_volumes",
        "notify_delete",
        "notify_delete_batch",
        "notify_put_batch",
        "serving_union",
    }
    assert required <= set(vars(ControllerProtocol))


def test_sliced_fetch_uses_the_first_replica_only():
    tensor_slice = TensorSlice([0], [0], [8], [4], [2])
    request = Request.from_any("K", None).meta_only()
    info = StorageInfo(ObjectType.TENSOR_SLICE, {tensor_slice})

    class _Buffer:
        supports_inplace_resharding = False

    planned, whole = LocalClient.__new__(LocalClient)._build_volume_requests(
        [request],
        {"K": {"v0": info, "v1": info}},
        {"v0": _Buffer(), "v1": _Buffer()},
    )

    assert list(planned) == ["v0"]
    assert len(planned["v0"]) == 1
    assert whole == set()


def test_tensor_slice_normalizes_every_hash_field():
    tensor_slice = TensorSlice([0], [0], [8], [4], [2])

    assert tensor_slice.offsets == (0,)
    assert tensor_slice.coordinates == (0,)
    assert tensor_slice.global_shape == (8,)
    assert tensor_slice.local_shape == (4,)
    assert tensor_slice.mesh_shape == (2,)
    assert hash(tensor_slice) == hash(
        TensorSlice((0,), (0,), (8,), (4,), (2,))
    )


def test_controller_service_and_handle_use_the_real_controller():
    adapter = RealControllerAdapter()
    request = Request.from_any("W", None).meta_only()
    adapter.service.notify_put_batch([request], "v0", pending=False)

    assert adapter.service._locate(["W"])["W"].keys() == {"v0"}
    assert adapter.handle.controller is adapter.controller
    assert hasattr(adapter.handle, "locate_volumes")
    assert hasattr(adapter.handle, "notify_put_batch")
