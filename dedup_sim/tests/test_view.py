"""Dedup selectors resolve their domain sensors through the shared map."""

from __future__ import annotations

import pytest

from dedup_sim.control._selector import Candidates
from dedup_sim.control._sensor import (
    Asked, FanoutSensor, Published, Routed, _request_covers, _storage_covers,
)
from proposed import DirectorySensor, Dispatcher, Endpoint, Environment
from proposed.selector import Balance, FirstMatch, Ordered
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request, TensorSlice


class _Holds:
    def __init__(self, volume: str) -> None:
        self.volume = volume

    def locate_raw(self, keys, missing_ok: bool = False):
        return {key: {self.volume: None} for key in keys}


_TOPOLOGY = {
    v: Endpoint(id=v, host=v, node=v) for v in ("origin", "r0", "r1")
}


class _Profile:
    def read_time(self, src: Endpoint, dst: Endpoint, nbytes: int) -> float:
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id == "origin" else 1.0


_PROFILE = _Profile()


def _request(key: str, tensor_slice: TensorSlice | None = None) -> Request:
    return Request.from_any(key, None, tensor_slice).meta_only()


def test_a_fanout_nobody_supplied_raises():
    directory = DirectorySensor(_Holds("origin"))
    with pytest.raises(RuntimeError, match="FanoutSensor"):
        Candidates().attach(
            Environment(_TOPOLOGY, _PROFILE), {DirectorySensor: directory}
        )


def test_the_ranking_keeps_fanout_under_reranking():
    fanout = FanoutSensor(fanout_cap=1)
    dispatcher = Dispatcher()
    dispatcher.compose(fanout)
    dispatcher.dispatch_sync(Routed("r0", "origin"))
    dispatcher.dispatch_sync(Asked("r0", (_request("K"),)))
    dispatcher.dispatch_sync(Asked("r1", (_request("K"),)))
    ranking = Candidates()
    chain = Ordered(FirstMatch([Balance(ranking)]))
    chain.attach(
        Environment(_TOPOLOGY, _PROFILE),
        {
            DirectorySensor: DirectorySensor(_Holds("origin")),
            FanoutSensor: fanout,
        },
    )

    assert ranking.sensor(FanoutSensor) is fanout
    assert chain.select(["K"], "r1").sources == ("r0", "origin")


def test_a_promised_batch_covers_only_its_keys():
    fanout = FanoutSensor()
    dispatcher = Dispatcher()
    dispatcher.compose(fanout)
    requests = {_key: _request(_key) for _key in ("K0", "K1")}
    dispatcher.dispatch_sync(Asked("r0", tuple(requests.values())))

    assert fanout.covers("r0", {"K0": requests["K0"]})
    assert fanout.covers("r0", requests)
    assert not fanout.covers("r0", {"K2": _request("K2")})


def test_a_producer_has_one_publication_in_flight():
    fanout = FanoutSensor()
    dispatcher = Dispatcher()
    dispatcher.compose(fanout)
    dispatcher.dispatch_sync(Asked("r0", (_request("K0"),)))

    with pytest.raises(ValueError, match="r0 already has an in-flight publication"):
        dispatcher.dispatch_sync(Asked("r0", (_request("K1"),)))

    dispatcher.dispatch_sync(Published("r0"))
    request = _request("K1")
    dispatcher.dispatch_sync(Asked("r0", (request,)))
    assert fanout.covers("r0", {"K1": request})


def test_region_coverage_uses_torchstore_metadata():
    half = TensorSlice((0,), (0,), (8,), (4,), (2,))
    quarter = TensorSlice((1,), (0,), (8,), (2,), (4,))
    crossing = TensorSlice((3,), (0,), (8,), (2,), (4,))
    available = _request("K", half)

    assert _request_covers(available, _request("K", quarter))
    assert not _request_covers(available, _request("K", crossing))

    stored = StorageInfo(ObjectType.TENSOR_SLICE, {half})
    assert _storage_covers(stored, _request("K", quarter))
    assert not _storage_covers(stored, _request("K", crossing))
