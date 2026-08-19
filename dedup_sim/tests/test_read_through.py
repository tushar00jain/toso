"""Read-through publication and retry behavior."""

from __future__ import annotations

from typing import Any

from dedup_sim.control._sensor import Published
from dedup_sim.control.routing import ReadPlan
from dedup_sim.data.read_through import ReadThroughPlane
from sim_common.async_engine import run_sim


class _Call:
    def __init__(self, member) -> None:
        self.member = member

    async def call_one(self, *args) -> Any:
        return await self.member(*args)


class _Control:
    def __init__(self, plans) -> None:
        self.asked = []
        self.plans = iter(plans)
        self.sources = _Call(self._sources)

    async def _sources(self, requests, requester):
        self.asked.append((requests, requester))
        return next(self.plans)


class _Dispatcher:
    def __init__(self) -> None:
        self.actions = []
        self.dispatch = _Call(self._dispatch)

    async def _dispatch(self, action) -> None:
        self.actions.append(action)


class _Client:
    def __init__(self, outcomes) -> None:
        self.outcomes = iter(outcomes)
        self.puts = []
        self.gets = []

    async def get_batch(self, entries):
        self.gets.append(entries)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def put_batch(self, entries):
        self.puts.append(entries)


class _Deployment:
    def __init__(self, plans, outcomes) -> None:
        self.control_plane_handle = _Control(plans)
        self.dispatcher_handle = _Dispatcher()
        self.client = _Client(outcomes)
        self.vends = []

    def client_for(self, requester, prefer=None):
        self.vends.append((requester, prefer))
        return self.client


def test_a_batch_put_dispatches_one_publication():
    plan = ReadPlan(("source",), (7, "r0"))
    deployment = _Deployment(
        [plan], [{"K0": "read-K0", "K1": "read-K1"}]
    )
    plane = ReadThroughPlane()
    plane.attach(deployment)

    result, _trace = run_sim(
        plane.read_through("r0", {"K0": None, "K1": None})
    )

    assert result == {"K0": "read-K0", "K1": "read-K1"}
    requests, requester = deployment.control_plane_handle.asked[0]
    assert requester == "r0"
    assert [(request.key, request.tensor_slice) for request in requests] == [
        ("K0", None),
        ("K1", None),
    ]
    assert deployment.client.puts == [result]
    assert deployment.dispatcher_handle.actions == [Published((7, "r0"))]
    assert deployment.vends == [("r0", ("source",)), ("r0", None)]


def test_an_evicted_preference_reasks_and_retires_both_publications():
    first = ReadPlan(("stale", "origin"), (7, "r0"))
    second = ReadPlan(("origin",), (8, "r0"))
    deployment = _Deployment(
        [first, second],
        [KeyError("stale source"), {"K": "read-K"}],
    )
    plane = ReadThroughPlane()
    plane.attach(deployment)

    result, _trace = run_sim(plane.read_through("r0", {"K": None}))

    assert result == {"K": "read-K"}
    assert len(deployment.control_plane_handle.asked) == 2
    assert deployment.dispatcher_handle.actions == [
        Published((7, "r0")),
        Published((8, "r0")),
    ]
    assert deployment.vends == [
        ("r0", ("stale", "origin")),
        ("r0", ("origin",)),
        ("r0", None),
    ]
