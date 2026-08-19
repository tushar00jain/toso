"""Pending publication behavior in the real controller."""

from __future__ import annotations

import pytest

from realsim.adapters.real_controller import RealControllerAdapter
from torchstore.controller import ObjectType, Pending
from torchstore.transport import Request, TensorSlice


def _tensor(key: str) -> Request:
    return Request.from_any(key, None).meta_only()


def _shard(key: str, coordinate: int) -> Request:
    return Request.from_tensor_slice(
        key,
        TensorSlice((coordinate * 4,), (coordinate,), (8,), (4,), (2,)),
    ).meta_only()


def test_pending_rows_are_hidden_from_locate_keys_and_commit_checks():
    service = RealControllerAdapter().service
    hidden = service.notify_put_batch([_tensor("W")], "v0")
    pub = service.notify_put_batch([_shard("D", 0)], "v0")
    service.notify_put_batch([_shard("D", 1)], "v1", pending=False)

    assert service._locate(["W"], missing_ok=True) == {}
    assert service.keys() == ["D"]
    raw = service.controller.keys_to_storage_volumes["D"]
    assert isinstance(raw["v0"], Pending)
    assert not service.controller._is_dtensor_fully_committed("D", raw)
    with pytest.raises(KeyError, match="partially committed"):
        service._locate(["D"])

    service.notify_delete_batch(pub=hidden)
    service.notify_delete_batch(pub=pub)
    with pytest.raises(KeyError, match="partially committed"):
        service._locate(["D"])


def test_a_real_put_replaces_the_pending_slot():
    service = RealControllerAdapter().service
    pub = service.notify_put_batch([_tensor("W")], "v0")

    service.notify_put_batch([_tensor("W")], "v0", pending=False)
    service.notify_delete_batch(pub=pub)

    held = service._locate(["W"])["W"]["v0"]
    assert type(held).__name__ == "StorageInfo"
    assert held.object_type is ObjectType.TENSOR


def test_retiring_a_publication_restores_the_live_entry_it_shadowed():
    service = RealControllerAdapter().service
    service.notify_put_batch([_shard("D", 0)], "v0", pending=False)
    pub = service.notify_put_batch([_shard("D", 1)], "v0")

    service.notify_delete_batch(pub=pub)

    held = service._locate(
        ["D"], require_fully_committed=False
    )["D"]["v0"]
    assert held.tensor_slices == {_shard("D", 0).tensor_slice}


def test_retiring_one_publication_keeps_another_publication_on_the_volume():
    service = RealControllerAdapter().service
    first = service.notify_put_batch([_tensor("W0")], "v0")
    second = service.notify_put_batch([_tensor("W1")], "v0")

    service.notify_delete_batch(pub=first)

    volumes, pubs = service.serving_union([_tensor("W1")])
    assert volumes == {"W1": set()}
    assert pubs == {"W1": {second}}
