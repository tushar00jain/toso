"""Pending publication behavior in the real controller."""

from __future__ import annotations

import pytest

from realsim.adapters.real_controller import RealControllerAdapter
from torchstore.controller import ObjectType, _live_view
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
    assert set(raw) == {"v0", "v1"}
    assert raw["v0"][pub].tensor_slices == {
        _shard("D", 0).tensor_slice
    }
    assert not service.controller._is_dtensor_fully_committed(
        "D", _live_view(raw)
    )
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


def test_retiring_a_publication_keeps_the_live_entry():
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

    serving = service.serving_union([_tensor("W1")])
    assert serving == frozenset({(second, "v0")})


def test_one_volume_can_serve_live_and_pending_slices():
    """A volume can appear as both live and pending in the flat union."""
    service = RealControllerAdapter().service
    live = _shard("D", 0)
    pending = _shard("D", 1)
    service.notify_put_batch([live], "v0", pending=False)
    pub = service.notify_put_batch([pending], "v0")
    requests = [live, pending]

    serving = service.serving_union(requests)

    assert serving == frozenset({(0, "v0"), (pub, "v0")})
    assert set(service.controller.keys_to_storage_volumes["D"]["v0"]) == {
        0,
        pub,
    }


def test_landing_keeps_publications_until_their_retirement():
    service = RealControllerAdapter().service
    first = service.notify_put_batch([_shard("D", 0)], "v0")
    second = service.notify_put_batch([_shard("D", 1)], "v0")

    service.notify_put_batch([_shard("D", 0)], "v0", pending=False)

    serving = service.serving_union([_shard("D", 0), _shard("D", 1)])
    assert set(service.controller.keys_to_storage_volumes["D"]["v0"]) == {
        0,
        first,
        second,
    }
    assert serving == frozenset({(0, "v0"), (first, "v0"), (second, "v0")})


def test_delete_drops_live_and_keeps_pending_on_the_slot():
    service = RealControllerAdapter().service
    service.notify_put_batch([_shard("D", 0)], "v0", pending=False)
    pub = service.notify_put_batch([_shard("D", 1)], "v0")

    service.notify_delete_batch({"v0": ["D"]})

    serving = service.serving_union([_shard("D", 0), _shard("D", 1)])
    assert serving == frozenset({(pub, "v0")})
