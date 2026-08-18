"""Dedup selectors resolve their domain sensors through the shared map."""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

from dedup_sim.control._selector import Candidates, Holders
from dedup_sim.control._sensor import (
    Asked,
    DedupDirectorySensor,
    FanoutSensor,
    Published,
    Routed,
)
from dedup_sim.control.routing import Dedup
from proposed import DirectorySensor, Dispatcher, Endpoint, Environment
from proposed.selector import Balance, FirstMatch, Ordered
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport import Request, TensorSlice
from sim_common.async_engine import run_sim


class _Holds:
    def __init__(self, volume: str) -> None:
        self.volume = volume

    def locate_raw(self, keys, missing_ok: bool = False):
        info = StorageInfo(ObjectType.TENSOR, {None})
        return {key: {self.volume: info} for key in keys}


_TOPOLOGY = {
    v: Endpoint(id=v, host=v, node=v) for v in ("origin", "replica", "r0", "r1")
}


class _Profile:
    def read_time(self, src: Endpoint, dst: Endpoint, nbytes: int) -> float:
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id == "origin" else 1.0


_PROFILE = _Profile()


def _request(key: str, tensor_slice: TensorSlice | None = None) -> Request:
    return Request.from_any(key, None, tensor_slice).meta_only()


def _dispatcher(directory, fanout):
    dispatcher = Dispatcher()
    dispatcher.compose(directory)
    dispatcher.compose(fanout)
    return dispatcher


def _routed(directory, requester, order):
    requests = tuple(directory.plan(requester).values())
    fetch = directory.plan_fetch(requests, order, requester=requester)
    return Routed(
        requester=requester,
        sources=fetch.sources,
        pending=tuple(source for source in fetch.sources if source in fetch.pending),
        required=tuple(
            (source, tuple(fetch.required[source].elements()))
            for source in fetch.sources
        ),
    )


def test_dedup_selectors_take_keys():
    assert Candidates.subject_type == Sequence[str]
    assert Holders.subject_type == Sequence[str]


def test_dedup_uses_the_supplied_concrete_directory_sensor():
    directory = DedupDirectorySensor(_Holds("origin"))
    plane = Dedup().attach(
        Environment(_TOPOLOGY, _PROFILE),
        {DedupDirectorySensor: directory},
    )
    assert plane.sensor(DedupDirectorySensor) is directory


def test_a_fanout_nobody_supplied_raises():
    directory = DedupDirectorySensor(_Holds("origin"))
    with pytest.raises(RuntimeError, match="FanoutSensor"):
        Candidates().attach(
            Environment(_TOPOLOGY, _PROFILE), {DedupDirectorySensor: directory}
        )


def test_holders_preserve_directory_order_and_name_each_source_once():
    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            info = StorageInfo(ObjectType.TENSOR, {None})
            return {
                "K0": {"replica": info, "origin": info},
                "K1": {"origin": info, "r0": info},
            }

    directory = DedupDirectorySensor(_Directory())
    selector = Holders().attach(
        Environment(_TOPOLOGY, _PROFILE),
        {DedupDirectorySensor: directory},
    )

    assert selector.select(["K0", "K1"], "r1").sources == (
        "replica",
        "origin",
        "r0",
    )


def test_the_ranking_keeps_fanout_under_reranking():
    fanout = FanoutSensor(fanout_cap=1)
    directory = DedupDirectorySensor(_Holds("origin"))
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("r0", (_request("K"),)))
    dispatcher.dispatch_sync(_routed(directory, "r0", ("origin",)))
    dispatcher.dispatch_sync(Asked("r1", (_request("K"),)))
    ranking = Candidates()
    chain = Ordered(FirstMatch([Balance(ranking)]))
    chain.attach(
        Environment(_TOPOLOGY, _PROFILE),
        {
            DedupDirectorySensor: directory,
            FanoutSensor: fanout,
        },
    )

    assert ranking.sensor(FanoutSensor) is fanout
    assert chain.select(["K"], "r1").sources == (
        "r0",
        "origin",
    )


def test_a_route_requires_regions_for_every_source():
    directory = DedupDirectorySensor(_Holds("origin"))
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)

    with pytest.raises(ValueError, match="non-empty regions for every source"):
        dispatcher.dispatch_sync(Routed("r0", ("origin",)))

    assert fanout.routes() == {}


def test_pending_entries_are_indexed_only_under_their_keys():
    directory = DedupDirectorySensor(_Holds("origin"))
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    requests = {_key: _request(_key) for _key in ("K0", "K1")}
    dispatcher.dispatch_sync(Asked("r0", tuple(requests.values())))

    assert directory.plan("r0") == requests
    pending = directory.pending(["K0"])["K0"]
    assert set(pending) == {"r0"}
    assert pending["r0"].object_type is ObjectType.TENSOR
    assert pending["r0"].tensor_slices == {None}
    assert directory.pending(["K2"]) == {"K2": {}}


def test_a_producer_has_one_publication_in_flight():
    directory = DedupDirectorySensor(_Holds("origin"))
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("r0", (_request("K0"),)))

    with pytest.raises(ValueError, match="r0 already has an in-flight publication"):
        dispatcher.dispatch_sync(Asked("r0", (_request("K1"),)))

    dispatcher.dispatch_sync(Published("r0"))
    request = _request("K1")
    dispatcher.dispatch_sync(Asked("r0", (request,)))
    assert directory.plan("r0") == {"K1": request}


def test_publication_folds_directory_and_fanout_before_one_waiter_wakes():
    async def publish():
        directory = DedupDirectorySensor(_Holds("origin"))
        fanout = FanoutSensor()
        dispatcher = _dispatcher(directory, fanout)
        assert Asked in directory.folds and Asked not in fanout.folds
        assert Published in directory.folds and Published in fanout.folds
        request = _request("K")
        dispatcher.dispatch_sync(Asked("r0", (request,)))
        dispatcher.dispatch_sync(_routed(directory, "r0", ("origin",)))
        ready = dispatcher.gate(lambda: False, (Published("r0"),))
        assert ready is not None
        assert len(dispatcher._waiters[Published("r0")]) == 1

        task = asyncio.create_task(ready())
        await asyncio.sleep(0)
        dispatcher.dispatch_sync(Published("r0"))
        await task
        return directory.in_flight(), fanout.route_required("r0")

    observed, _trace = run_sim(publish())
    assert observed == (set(), {})


def test_pending_sources_do_not_satisfy_live_coverage():
    info = StorageInfo(ObjectType.TENSOR, {None})

    class _Directory:
        def __init__(self):
            self.entries = {}

        def locate_raw(self, keys, missing_ok=False):
            return {key: dict(self.entries.get(key, {})) for key in keys}

    state = _Directory()
    directory = DedupDirectorySensor(state)
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    request = _request("K")
    dispatcher.dispatch_sync(Asked("p", (request,)))
    planned = directory.plan_fetch([request], ("p",))

    assert directory.serving_sources([request]) == ({"p"}, {"p"})
    assert not directory.covers([request], planned.required, live=True)

    state.entries["K"] = {"p": info}
    assert directory.covers([request], planned.required, live=True)
    assert directory.serving_sources([request]) == ({"p"}, set())

    dispatcher.dispatch_sync(Published("p"))
    assert directory.pending(["K"]) == {"K": {}}
    assert directory.serving_sources([request]) == ({"p"}, set())


def test_region_planning_is_torchstores_expansion():
    directory = DirectorySensor(_Holds("origin"))
    half = TensorSlice((0,), (0,), (8,), (4,), (2,))
    quarter = TensorSlice((1,), (0,), (8,), (2,), (4,))
    crossing = TensorSlice((3,), (0,), (8,), (2,), (4,))
    stored = StorageInfo(ObjectType.TENSOR_SLICE, {half})
    quarter_plan = directory.plan_requests(
        [_request("K", quarter)], {"K": {"origin": stored}}
    )
    crossing_plan = directory.plan_requests(
        [_request("K", crossing)], {"K": {"origin": stored}}
    )

    assert quarter_plan["origin"][0].tensor_slice.offsets == quarter.offsets
    assert quarter_plan["origin"][0].tensor_slice.local_shape == quarter.local_shape
    assert crossing_plan["origin"][0].tensor_slice.local_shape == (1,)
    assert crossing_plan["origin"][0].tensor_slice.offsets == (3,)


def test_every_whole_value_source_is_rankable_before_narrowing():
    request = _request("K")
    live = {
        "K": {
            "origin": StorageInfo(ObjectType.TENSOR, {None}),
            "replica": StorageInfo(ObjectType.TENSOR, {None}),
        }
    }

    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            return {key: dict(live.get(key, {})) for key in keys}

    directory = DedupDirectorySensor(_Directory())
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("r0", (request,)))

    candidates, pending = directory.serving_sources([request])
    planned = directory.plan_fetch([request], ("r0", "replica", "origin"))
    without_peer = directory.plan_fetch([request], ("replica", "origin"))

    assert candidates == {"origin", "replica", "r0"}
    assert pending == {"r0"}
    assert planned.by_key == {"K": ("r0",)}
    assert without_peer.by_key == {"K": ("replica",)}


def test_two_pending_slices_form_one_torchstore_fetch_plan():
    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            return {}

    directory = DedupDirectorySensor(_Directory())
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    left = TensorSlice((0,), (0,), (8,), (4,), (2,))
    right = TensorSlice((4,), (1,), (8,), (4,), (2,))
    request = _request("K")
    dispatcher.dispatch_sync(Asked("p0", (_request("K", left),)))
    dispatcher.dispatch_sync(Asked("p1", (_request("K", right),)))

    planned = directory.plan_fetch([request], ("p0", "p1"))
    live = {
        "K": {
            "p0": StorageInfo(ObjectType.TENSOR_SLICE, {left}),
            "p1": StorageInfo(ObjectType.TENSOR_SLICE, {right}),
        }
    }

    assert planned.by_key == {"K": ("p0", "p1")}
    assert planned.pending == {"p0", "p1"}
    assert directory.covers([request], planned.required, live)


def test_sparse_candidate_discovery_visits_only_present_entries(monkeypatch):
    requests = [_request(f"K{i}") for i in range(20)]
    live = {
        request.key: {f"v{i}": StorageInfo(ObjectType.TENSOR, {None})}
        for i, request in enumerate(requests)
    }

    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            return {key: dict(live.get(key, {})) for key in keys}

    directory = DedupDirectorySensor(_Directory())
    visited = 0
    original = directory.plan_requests

    def counted(requests, located=None):
        nonlocal visited
        visited += sum(len(entries) for entries in located.values())
        return original(requests, located)

    monkeypatch.setattr(directory, "plan_requests", counted)

    candidates, pending = directory.serving_sources(requests)

    assert candidates == {f"v{i}" for i in range(20)}
    assert pending == set()
    assert visited == 40


def test_a_multi_slice_plan_waits_for_exactly_its_two_producers():
    left = TensorSlice((0,), (0,), (8,), (4,), (2,))
    right = TensorSlice((4,), (1,), (8,), (4,), (2,))

    class _Directory:
        def __init__(self):
            self.entries = {
                "K": {
                    "t0": StorageInfo(ObjectType.TENSOR_SLICE, {left}),
                    "t1": StorageInfo(ObjectType.TENSOR_SLICE, {right}),
                }
            }

        def locate_raw(self, keys, missing_ok=False):
            return {key: dict(self.entries.get(key, {})) for key in keys}

        def publish(self, source, tensor_slice):
            self.entries["K"][source] = StorageInfo(
                ObjectType.TENSOR_SLICE, {tensor_slice}
            )

    async def decide():
        directory = _Directory()
        ids = ("t0", "t1", "p0", "p1", "r")
        topology = {i: Endpoint(id=i, host=i, node=i) for i in ids}

        class _SliceProfile:
            def read_time(self, src, dst, nbytes):
                if src.id == dst.id:
                    return 0.0
                return 10.0 if src.id.startswith("t") else 1.0

        plane = Dedup().attach(
            Environment(topology, _SliceProfile()),
            {DedupDirectorySensor: DedupDirectorySensor(directory)},
        )
        plane.dispatcher.dispatch_sync(Asked("p0", (_request("K", left),)))
        sensor = plane.sensor(DedupDirectorySensor)
        plane.dispatcher.dispatch_sync(_routed(sensor, "p0", ("t0",)))
        plane.dispatcher.dispatch_sync(Asked("p1", (_request("K", right),)))
        plane.dispatcher.dispatch_sync(_routed(sensor, "p1", ("t1",)))

        planned = await plane._decide([_request("K")], "r")
        assert planned.by_key == {"K": ("p0", "p1")}
        assert set(plane.dispatcher._waiters) == {Published("p0"), Published("p1")}
        assert all(len(waiters) == 1 for waiters in plane.dispatcher._waiters.values())
        task = asyncio.create_task(planned.settled())
        await asyncio.sleep(0)
        directory.publish("p0", left)
        plane.dispatcher.dispatch_sync(Published("p0"))
        await asyncio.sleep(0)
        assert not task.done()
        plane.dispatcher.dispatch_sync(Published("other"))
        await asyncio.sleep(0)
        assert not task.done()
        directory.publish("p1", right)
        plane.dispatcher.dispatch_sync(Published("p1"))
        return await task

    settled, _trace = run_sim(decide())
    assert settled.ready is None


def test_a_partial_reader_can_follow_a_peer_fetching_more_regions():
    left = TensorSlice((0,), (0,), (8,), (4,), (2,))
    right = TensorSlice((4,), (1,), (8,), (4,), (2,))

    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            return {
                "K": {
                    "t0": StorageInfo(ObjectType.TENSOR_SLICE, {left}),
                    "t1": StorageInfo(ObjectType.TENSOR_SLICE, {right}),
                }
            }

    class _SliceProfile:
        def read_time(self, src, dst, nbytes):
            if src.id == dst.id:
                return 0.0
            return 10.0 if src.id.startswith("t") else 1.0

    ids = ("t0", "t1", "p0", "r")
    topology = {i: Endpoint(id=i, host=i, node=i) for i in ids}
    directory = DedupDirectorySensor(_Directory())
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("p0", (_request("K"),)))
    dispatcher.dispatch_sync(_routed(directory, "p0", ("t0", "t1")))
    dispatcher.dispatch_sync(Asked("r", (_request("K", left),)))
    ranking = Ordered(Candidates()).attach(
        Environment(topology, _SliceProfile()),
        {DedupDirectorySensor: directory, FanoutSensor: fanout},
    )

    assert ranking.select(["K"], "r").head == "p0"


def test_a_subset_reader_accounts_for_the_peers_other_keys():
    info = StorageInfo(ObjectType.TENSOR, {None})

    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            all_entries = {"K0": {"t0": info}, "K1": {"t1": info}}
            return {key: all_entries[key] for key in keys}

    class _Profile:
        def read_time(self, src, dst, nbytes):
            if src.id == dst.id:
                return 0.0
            return 10.0 if src.id.startswith("t") else 1.0

    ids = ("t0", "t1", "p0", "r")
    topology = {i: Endpoint(id=i, host=i, node=i) for i in ids}
    directory = DedupDirectorySensor(_Directory())
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("p0", (_request("K0"), _request("K1"))))
    dispatcher.dispatch_sync(_routed(directory, "p0", ("t0", "t1")))
    dispatcher.dispatch_sync(Asked("r", (_request("K0"),)))
    ranking = Ordered(Candidates()).attach(
        Environment(topology, _Profile()),
        {DedupDirectorySensor: directory, FanoutSensor: fanout},
    )

    assert ranking.select(["K0"], "r").head == "p0"


def test_a_route_accepts_a_dependency_that_has_since_published():
    info = StorageInfo(ObjectType.TENSOR, {None})

    class _Directory:
        def __init__(self):
            self.entries = {"K0": {"t0": info}, "K1": {"t1": info}}

        def locate_raw(self, keys, missing_ok=False):
            return {key: dict(self.entries[key]) for key in keys}

    class _Profile:
        def read_time(self, src, dst, nbytes):
            if src.id == dst.id:
                return 0.0
            return 10.0 if src.id.startswith("t") else 1.0

    ids = ("t0", "t1", "q", "p", "r")
    topology = {i: Endpoint(id=i, host=i, node=i) for i in ids}
    state = _Directory()
    directory = DedupDirectorySensor(state)
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("q", (_request("K1"),)))
    dispatcher.dispatch_sync(_routed(directory, "q", ("t1",)))
    dispatcher.dispatch_sync(Asked("p", (_request("K0"), _request("K1"))))
    dispatcher.dispatch_sync(_routed(directory, "p", ("t0", "q")))
    state.entries["K1"]["q"] = info
    dispatcher.dispatch_sync(Published("q"))
    dispatcher.dispatch_sync(Asked("r", (_request("K0"),)))
    ranking = Ordered(Candidates()).attach(
        Environment(topology, _Profile()),
        {DedupDirectorySensor: directory, FanoutSensor: fanout},
    )

    assert ranking.select(["K0"], "r").head == "p"


def test_a_route_rejects_a_registered_source_with_the_wrong_slice():
    left = TensorSlice((0,), (0,), (8,), (4,), (2,))
    right = TensorSlice((4,), (1,), (8,), (4,), (2,))
    request = _request("K", left)
    original = {"K": {"t": StorageInfo(ObjectType.TENSOR_SLICE, {left})}}

    class _Original:
        def locate_raw(self, keys, missing_ok=False):
            return {key: dict(original.get(key, {})) for key in keys}

    planning_directory = DedupDirectorySensor(_Original())
    route = planning_directory.plan_fetch([request], ("t",))

    class _Directory:
        def locate_raw(self, keys, missing_ok=False):
            return {"K": {"t": StorageInfo(ObjectType.TENSOR_SLICE, {right})}}

    ids = ("t", "p", "r")
    topology = {i: Endpoint(id=i, host=i, node=i) for i in ids}
    directory = DedupDirectorySensor(_Directory())
    fanout = FanoutSensor()
    dispatcher = _dispatcher(directory, fanout)
    dispatcher.dispatch_sync(Asked("p", (request,)))
    dispatcher.dispatch_sync(
        Routed(
            requester="p",
            sources=("t",),
            required=(("t", tuple(route.required["t"].elements())),),
        )
    )
    dispatcher.dispatch_sync(Asked("r", (request,)))
    ranking = Ordered(Candidates()).attach(
        Environment(topology, _PROFILE),
        {DedupDirectorySensor: directory, FanoutSensor: fanout},
    )

    assert ranking.select(["K"], "r").sources == ()
