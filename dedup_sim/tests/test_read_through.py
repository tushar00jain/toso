"""The read-through publishes one completed batch."""

from __future__ import annotations

from typing import Any

from dedup_sim.control._sensor import Published
from dedup_sim.data.read_through import ReadThroughPlane
from proposed import Selection
from sim_common.async_engine import run_sim


class _Call:
    def __init__(self, member) -> None:
        self.member = member

    async def call_one(self, *args) -> Any:
        return await self.member(*args)


class _Control:
    def __init__(self) -> None:
        self.asked = []
        self.sources = _Call(self._sources)

    async def _sources(self, keys, requester):
        self.asked.append((keys, requester))
        return Selection(sources=("source",))


class _Dispatcher:
    def __init__(self) -> None:
        self.actions = []
        self.dispatch = _Call(self._dispatch)

    async def _dispatch(self, action) -> None:
        self.actions.append(action)


class _Client:
    def __init__(self) -> None:
        self.gets = []
        self.puts = []

    async def get_batch(self, keys):
        self.gets.append(keys)
        return {key: f"read-{key}" for key in keys}

    async def put_batch(self, entries):
        self.puts.append(entries)


class _Deployment:
    def __init__(self) -> None:
        self.control_plane_handle = _Control()
        self.dispatcher_handle = _Dispatcher()
        self.client = _Client()
        self.vends = []

    def client_for(self, requester, prefer=None):
        self.vends.append((requester, prefer))
        return self.client


def test_a_batch_put_dispatches_one_publication():
    deployment = _Deployment()
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
    assert deployment.client.gets == [{"K0": None, "K1": None}]
    assert deployment.client.puts == [{"K0": "read-K0", "K1": "read-K1"}]
    assert deployment.dispatcher_handle.actions == [Published("r0")]
    assert deployment.vends == [("r0", ("source",)), ("r0", None)]
