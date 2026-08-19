"""A promised directory entry is invisible to everything except whoever asked for it.

The directory holds two kinds of entry in one map (:mod:`realsim.seams.projection`),
and every guarantee here is about the boundary between them: an ordinary read, the
DTensor commit check and :meth:`keys` answer as though the promises were not there,
and a real put replaces the promise it lands on instead of unioning with it. A volume
holding part of a key and promising the rest has one entry covering both, so the
boundary runs through an entry rather than between them.
"""

from __future__ import annotations

import asyncio

import pytest

from realsim.adapters.real_controller import RealControllerAdapter
from realsim.seams.projection import Promised
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request, TensorSlice


def _tensor(key: str) -> Request:
    return Request.from_any(key, None).meta_only()


def _shard(key: str, coordinate: int) -> Request:
    return Request.from_any(
        key,
        None,
        TensorSlice(
            offsets=(coordinate * 2,),
            coordinates=(coordinate,),
            global_shape=(4,),
            local_shape=(2,),
            mesh_shape=(2,),
        ),
    ).meta_only()


def _service():
    return RealControllerAdapter().service


def test_a_promise_is_not_a_holder():
    service = _service()
    asyncio.run(service.notify_put_batch([_tensor("W")], "v0"))
    service.project("v1", "W", StorageInfo(ObjectType.TENSOR, {None}))

    assert set(service.locate_raw(["W"])["W"]) == {"v0"}
    assert set(asyncio.run(service.locate_volumes(["W"]))["W"]) == {"v0"}
    assert set(service.locate_raw(["W"], projected=True)["W"]) == {"v0", "v1"}


def test_a_key_nobody_holds_yet_is_missing_from_an_ordinary_read():
    service = _service()
    service.project("v1", "W", StorageInfo(ObjectType.TENSOR, {None}))

    assert service.locate_raw(["W"], missing_ok=True) == {}
    assert asyncio.run(service.keys()) == []
    with pytest.raises(KeyError, match="Unable to locate W"):
        service.locate_raw(["W"])
    assert set(service.locate_raw(["W"], projected=True)["W"]) == {"v1"}


def test_a_promised_shard_does_not_complete_a_dtensor():
    """The filter has to run *above* the commit check, not after it."""
    service = _service()
    asyncio.run(service.notify_put_batch([_shard("D", 0)], "v0"))
    service.project(
        "v1", "D", StorageInfo(ObjectType.TENSOR_SLICE, {_shard("D", 1).tensor_slice})
    )

    with pytest.raises(KeyError, match="only partially committed"):
        service.locate_raw(["D"])
    assert service.locate_raw(["D"], require_fully_committed=False)["D"].keys() == {
        "v0"
    }

    asyncio.run(service.notify_put_batch([_shard("D", 1)], "v1"))
    assert set(service.locate_raw(["D"])["D"]) == {"v0", "v1"}


def test_a_put_replaces_the_promise_it_lands_on():
    service = _service()
    service.project("v0", "W", StorageInfo(ObjectType.TENSOR, {None}))
    asyncio.run(service.notify_put_batch([_tensor("W")], "v0"))

    entry = service.entries["W"]["v0"]
    assert not isinstance(entry, Promised), "a landed write is still tagged a promise"
    assert service.projected_owners() == {}
    assert set(service.locate_raw(["W"])["W"]) == {"v0"}


def test_a_volume_can_hold_part_of_a_key_and_promise_the_rest():
    service = _service()
    asyncio.run(service.notify_put_batch([_shard("D", 0)], "v0"))
    service.project(
        "v0", "D", StorageInfo(ObjectType.TENSOR_SLICE, {_shard("D", 1).tensor_slice})
    )

    held = service.locate_raw(["D"], require_fully_committed=False)["D"]["v0"]
    assert held.tensor_slices == {_shard("D", 0).tensor_slice}
    promised = service.locate_raw(
        ["D"], require_fully_committed=False, projected=True
    )["D"]["v0"]
    assert promised.tensor_slices == {
        _shard("D", 0).tensor_slice,
        _shard("D", 1).tensor_slice,
    }


def test_clearing_a_promise_leaves_the_live_half_it_covered():
    """Restored, not dropped: the slot carried real data underneath the promise."""
    service = _service()
    asyncio.run(service.notify_put_batch([_shard("D", 0)], "v0"))
    service.project(
        "v0", "D", StorageInfo(ObjectType.TENSOR_SLICE, {_shard("D", 1).tensor_slice})
    )
    service.clear_projections("v0")

    assert service.projected_owners() == {}
    held = service.locate_raw(["D"], require_fully_committed=False)["D"]["v0"]
    assert not isinstance(held, Promised)
    assert held.tensor_slices == {_shard("D", 0).tensor_slice}


def test_a_promise_cannot_change_a_keys_object_type():
    service = _service()
    asyncio.run(service.notify_put_batch([_tensor("W")], "v0"))
    service.project("v0", "W", StorageInfo(ObjectType.OBJECT, {None}))

    assert service.projected_owners() == {}
    assert service.locate_raw(["W"])["W"]["v0"].object_type is ObjectType.TENSOR


def test_clearing_a_producer_leaves_the_keys_it_published():
    service = _service()
    for key in ("W0", "W1"):
        service.project("v0", key, StorageInfo(ObjectType.TENSOR, {None}))
    asyncio.run(service.notify_put_batch([_tensor("W0")], "v0"))
    service.clear_projections("v0")

    assert service.projected_owners() == {}
    assert asyncio.run(service.keys()) == ["W0"]
    assert service.locate_raw(["W0", "W1"], missing_ok=True, projected=True).keys() == {
        "W0"
    }


def test_the_live_view_follows_a_delete_under_a_promise():
    service = _service()
    asyncio.run(service.notify_put_batch([_tensor("W")], "v0"))
    service.project("v1", "W", StorageInfo(ObjectType.TENSOR, {None}))
    assert set(service.locate_raw(["W"])["W"]) == {"v0"}

    asyncio.run(service.notify_delete("W", "v0"))

    assert service.locate_raw(["W"], missing_ok=True) == {}
    assert set(service.locate_raw(["W"], projected=True)["W"]) == {"v1"}
