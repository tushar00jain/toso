"""Publication routing state over the real controller."""

from __future__ import annotations

import asyncio

import pytest

from dedup_sim.control._selector import Candidates
from dedup_sim.control._sensor import (
    Asked,
    DedupDirectorySensor,
    FanoutSensor,
    Published,
    Routed,
)
from dedup_sim.control.routing import Dedup, _gate_publications
from proposed import Dispatcher, Endpoint, Environment
from proposed.selector import Ordered
from realsim.adapters.real_controller import RealControllerAdapter
from sim_common.async_engine import run_sim
from torchstore.transport import Request, TensorSlice


def _request(key: str) -> Request:
    return Request.from_any(key, None).meta_only()


def _slice_request(key: str, coordinate: int = 0) -> Request:
    tensor_slice = TensorSlice(
        (coordinate * 4,), (coordinate,), (8,), (4,), (2,)
    )
    return Request.from_tensor_slice(key, tensor_slice).meta_only()


def _directory() -> DedupDirectorySensor:
    return DedupDirectorySensor(RealControllerAdapter().service)


def _dispatcher(directory, fanout):
    dispatcher = Dispatcher()
    dispatcher.compose(directory)
    dispatcher.compose(fanout)
    return dispatcher


class _Profile:
    def read_time(self, src: Endpoint, dst: Endpoint, nbytes: int) -> float:
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id == "origin" else 1.0


def test_dedup_uses_the_supplied_concrete_directory_sensor():
    directory = _directory()
    plane = Dedup().attach(
        Environment(
            {v: Endpoint(v, v, v) for v in ("origin", "r0")}, _Profile()
        ),
        {DedupDirectorySensor: directory},
    )
    assert plane.sensor(DedupDirectorySensor) is directory


def test_a_fanout_nobody_supplied_raises():
    directory = _directory()
    with pytest.raises(RuntimeError, match="FanoutSensor"):
        Candidates().attach(
            Environment(
                {v: Endpoint(v, v, v) for v in ("origin", "r0")}, _Profile()
            ),
            {DedupDirectorySensor: directory},
        )


def test_two_publications_from_one_host_and_redeclared_keys_coexist():
    directory = _directory()
    dispatcher = _dispatcher(directory, FanoutSensor())
    first = directory.declare("r0", (_request("K0"),))
    second = directory.declare("r0", (_request("K1"),))
    duplicate = directory.declare("r0", (_request("K0"),))
    for pub in (first, second, duplicate):
        dispatcher.dispatch_sync(Asked(pub))

    assert directory.in_flight() == {first, second, duplicate}
    assert directory.serving_union((_request("K0"),)) == frozenset({
        first,
        duplicate,
    })
    assert directory.serving_union((_request("K1"),)) == frozenset({
        second
    })


def test_serving_union_returns_one_flat_candidate_set():
    directory = _directory()
    dispatcher = _dispatcher(directory, FanoutSensor())
    first = directory.declare("r0", (_request("K0"),))
    second = directory.declare("r1", (_request("K1"),))
    dispatcher.dispatch_sync(Asked(first))
    dispatcher.dispatch_sync(Asked(second))

    serving = directory.serving_union((_request("K0"), _request("K1")))

    assert serving == frozenset({first, second})


def test_serving_union_filters_slice_candidates():
    directory = _directory()
    dispatcher = _dispatcher(directory, FanoutSensor())
    first = directory.declare("r0", (_slice_request("K", 0),))
    second = directory.declare("r1", (_slice_request("K", 1),))
    dispatcher.dispatch_sync(Asked(first))
    dispatcher.dispatch_sync(Asked(second))

    serving = directory.serving_union((_slice_request("K", 0),))

    assert serving == frozenset({first})


def test_greedy_walk_stops_after_first_synchronized_batch_source():
    directory = _directory()
    requests = (_request("K0"), _request("K1"))
    first = directory.declare("r0", requests)
    second = directory.declare("r1", requests)

    chosen = directory.greedy_cover(requests, (first, second))

    assert chosen == [first]
    assert _gate_publications(chosen) == {first}


def test_greedy_walk_chooses_one_source_per_disjoint_region():
    directory = _directory()
    first = directory.declare("r0", (_request("K0"),))
    second = directory.declare("r1", (_request("K1"),))
    requests = (_request("K0"), _request("K1"))

    chosen = directory.greedy_cover(requests, (first, second))

    assert chosen == [first, second]
    assert _gate_publications(chosen) == {first, second}


def test_batch_one_publication_does_not_open_batch_twos_gate():
    async def publish():
        directory = _directory()
        dispatcher = _dispatcher(directory, FanoutSensor())
        first = directory.declare("r0", (_request("K0"),))
        second = directory.declare("r0", (_request("K1"),))
        dispatcher.dispatch_sync(Asked(first))
        dispatcher.dispatch_sync(Asked(second))
        ready = dispatcher.gate(lambda: False, (Published(first),))
        task = asyncio.create_task(ready())
        await asyncio.sleep(0)
        dispatcher.dispatch_sync(Published(second))
        await asyncio.sleep(0)
        assert not task.done()
        dispatcher.dispatch_sync(Published(first))
        await task

    run_sim(publish())


def test_batch_two_rows_survive_batch_one_publication():
    directory = _directory()
    dispatcher = _dispatcher(directory, FanoutSensor())
    first = directory.declare("r0", (_request("K0"),))
    second = directory.declare("r0", (_request("K1"),))
    dispatcher.dispatch_sync(Asked(first))
    dispatcher.dispatch_sync(Asked(second))

    dispatcher.dispatch_sync(Published(first))

    assert directory.serving_union((_request("K0"),)) == frozenset()
    assert directory.serving_union((_request("K1"),)) == frozenset({
        second
    })


def test_publication_folds_state_before_its_gate_wakes():
    async def publish():
        directory = _directory()
        fanout = FanoutSensor()
        dispatcher = _dispatcher(directory, fanout)
        pub = directory.declare("r0", (_request("K"),))
        dispatcher.dispatch_sync(Asked(pub))
        dispatcher.dispatch_sync(Routed(pub, ("origin",), frozenset(), 10.0))
        ready = dispatcher.gate(lambda: False, (Published(pub),))
        task = asyncio.create_task(ready())
        await asyncio.sleep(0)
        dispatcher.dispatch_sync(Published(pub))
        await task
        return directory.in_flight(), fanout.arrival(pub)

    observed, _trace = run_sim(publish())
    assert observed == (set(), None)


def test_pending_fanout_cap_excludes_a_full_peer():
    ids = ("origin", "p0", "r0", "r1")
    environment = Environment({v: Endpoint(v, v, v) for v in ids}, _Profile())
    directory = _directory()
    directory.directory.notify_put_batch((_request("K"),), "origin", pending=False)
    fanout = FanoutSensor(fanout_cap=1)
    dispatcher = _dispatcher(directory, fanout)
    peer = directory.declare("p0", (_request("K"),))
    dispatcher.dispatch_sync(Asked(peer))
    dispatcher.dispatch_sync(Routed(peer, ("origin",), frozenset(), 10.0))
    reader = directory.declare("r0", (_request("K"),))
    dispatcher.dispatch_sync(Asked(reader))
    dispatcher.dispatch_sync(Routed(reader, ("p0",), frozenset({peer}), 11.0))
    serving = directory.serving_union((_request("K"),))
    ranking = Ordered(Candidates()).attach(
        environment,
        {DedupDirectorySensor: directory, FanoutSensor: fanout},
    )

    sources = ranking.select(serving, "r1").sources
    assert sources[0] == reader
    assert peer not in sources


def test_candidate_wait_ranks_both_pubs_and_greedy_picks_the_faster_one():
    ids = ("origin", "p0", "r0")
    environment = Environment({v: Endpoint(v, v, v) for v in ids}, _Profile())
    directory = _directory()
    fanout = FanoutSensor(fanout_cap=2)
    dispatcher = _dispatcher(directory, fanout)
    fast = directory.declare("p0", (_request("K"),))
    slow = directory.declare("p0", (_request("K"),))
    for pub, arrival in ((fast, 2.0), (slow, 9.0)):
        dispatcher.dispatch_sync(Asked(pub))
        dispatcher.dispatch_sync(Routed(pub, ("origin",), frozenset(), arrival))
    requests = (_request("K"),)
    serving = directory.serving_union(requests)
    candidates = Candidates(fabric=0.0).attach(
        environment,
        {DedupDirectorySensor: directory, FanoutSensor: fanout},
    )

    selection = candidates.select(serving, "r0")

    fast_arrival = fanout.arrival(fast)
    slow_arrival = fanout.arrival(slow)
    assert fast_arrival is not None
    assert slow_arrival is not None
    assert selection.key is not None
    assert selection.sources == (fast, slow)
    assert selection.key[fast] == (fast_arrival + 1.0,)
    assert selection.key[slow] == (slow_arrival + 1.0,)
    assert directory.greedy_cover(requests, selection.sources) == [fast]


def test_publication_arrival_uses_every_chosen_source():
    ids = ("origin", "p0", "p1", "r0")
    environment = Environment({v: Endpoint(v, v, v) for v in ids}, _Profile())
    directory = _directory()
    plane = Dedup().attach(
        environment,
        {DedupDirectorySensor: directory},
    )
    fanout = plane.sensor(FanoutSensor)
    first = directory.declare("p0", (_request("K0"),))
    second = directory.declare("p1", (_request("K1"),))
    for pub, arrival in ((first, 4.0), (second, 9.0)):
        plane.dispatcher.dispatch_sync(Asked(pub))
        plane.dispatcher.dispatch_sync(
            Routed(pub, ("origin",), frozenset(), arrival)
        )

    plan, _trace = run_sim(
        plane._decide((_request("K0"), _request("K1")), "r0")
    )

    first_arrival = fanout.arrival(first)
    second_arrival = fanout.arrival(second)
    assert first_arrival is not None
    assert second_arrival is not None
    assert fanout.arrival(plan.publication) == max(
        first_arrival + 1.0,
        second_arrival + 1.0,
    )
