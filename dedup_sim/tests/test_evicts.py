"""Dedup over bounded volumes that evict between read rounds."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, List

from dedup_sim.control._sensor import DedupDirectorySensor, Published
from dedup_sim.control.routing import Dedup, ReadPlan
from dedup_sim.data.read_through import ReadThroughPlane
from putget_sim.workload.put_get import DEFAULT_N, KEY, PutGetBurst
from realsim.run import Result, Run
from realsim.runner import ItemDispatch, WorkItem
from realsim.simulation import Simulation
from sim_common.async_engine import run_sim
from sim_common.cost_model import DEFAULT_PROFILE

NEXT_VERSION = "W2"
PAYLOAD_BYTES = DEFAULT_N * 4
ROUND = 1.0


class VersionedRounds(PutGetBurst):
    """Read one version, displace it locally, then read it again."""

    def items(self, sim: Simulation) -> List[WorkItem]:
        mesh, value = sim.mesh, self.put_value
        sim.origins(self.origin_id)

        def _get(reader: str, key: str) -> Callable[[], Any]:
            async def call() -> Any:
                mesh.bind_source(reader)
                return await mesh.client(reader).get(key)

            return call

        def _put(reader: str, key: str) -> Callable[[], Any]:
            async def call() -> Any:
                mesh.bind_source(reader)
                return await mesh.client(reader).put(key, value)

            return call

        items: List[WorkItem] = []
        for reader in self.reader_ids:
            items += [
                WorkItem(f"{reader}/1-read", 0.0, _get(reader, KEY), (reader, KEY)),
                WorkItem(f"{reader}/2-next", ROUND, _put(reader, NEXT_VERSION)),
                WorkItem(
                    f"{reader}/3-read", 2 * ROUND, _get(reader, KEY), (reader, KEY)
                ),
            ]
        return items


def _per_item(sim, workload) -> Any:
    plane = ReadThroughPlane()
    plane.attach(sim)

    async def drive(item: Any) -> Any:
        if item.payload is None:
            return await item.run()
        requester, key = item.payload
        return (await plane.read_through(requester, {key: None}))[key]

    return drive


def _run(num_readers: int = 3, *, fanout_cap: int = 1) -> tuple[Result, Dedup]:
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES)
    workload = VersionedRounds(num_readers, profile=profile)
    plane = Dedup(fanout_cap=fanout_cap)
    result = Run(
        f"cap={fanout_cap}",
        workload,
        control=plane,
        data=lambda sim: ItemDispatch(_per_item(sim, workload)),
        profile=profile,
    ).execute()
    return result, plane


def _directory(result: Result) -> dict:
    return result.sim.mesh.directory.controller.keys_to_storage_volumes


def test_the_rounds_do_not_overlap():
    result, _plane = _run()
    assert result.ledger.items_done == result.ledger.items_total == 9
    for row in result.ledger.rows:
        assert row.done < row.released + ROUND, row.id


def test_the_next_version_evicts_the_one_the_reader_cached():
    result, _plane = _run()
    for reader in result.workload.reader_ids:
        volume = result.sim.mesh.volumes[reader].service
        assert sorted(volume.store.kv) == [KEY], reader
        assert volume.resident_bytes == PAYLOAD_BYTES, reader
    assert NEXT_VERSION not in _directory(result)


def test_the_chain_re_forms_after_the_peers_evict():
    result, _plane = _run()
    origin = result.workload.origin_id

    assert result.ledger.origin_bytes == 2 * PAYLOAD_BYTES  # 1x per read of W
    origin_edges = [e for e in result.ledger.edges if e[0] == origin]
    assert len(origin_edges) == 2  # one hop out of the origin, per round
    assert result.ledger.transfer_bytes == 6 * PAYLOAD_BYTES


def test_every_reader_still_receives_the_payload_in_both_rounds():
    result, _plane = _run()
    expected = result.workload.expected
    for item_id, payload in result.results.items():
        if item_id.endswith("-read"):
            assert payload.shape == expected.shape, item_id
            assert payload.dtype == expected.dtype, item_id


def test_the_fabric_is_1x_per_read_for_any_fanout_cap():
    for cap in (1, 2, 3):
        result, _plane = _run(4, fanout_cap=cap)
        assert result.ledger.origin_bytes == 2 * PAYLOAD_BYTES, cap


class _Call:
    def __init__(self, member) -> None:
        self.member = member

    async def call_one(self, *args):
        return await self.member(*args)


class _RetryDeployment:
    def __init__(self) -> None:
        self.asks = 0
        self.actions = []
        self.gets = 0
        self.puts = []
        self.control_plane_handle = type("Control", (), {})()
        self.control_plane_handle.sources = _Call(self._sources)
        self.dispatcher_handle = type("Dispatch", (), {})()
        self.dispatcher_handle.dispatch = _Call(self._dispatch)

    async def _sources(self, requests, requester):
        self.asks += 1
        return ReadPlan(("peer",), (self.asks, requester))

    async def _dispatch(self, action):
        self.actions.append(action)

    def client_for(self, requester, prefer=None):
        return self

    async def get_batch(self, entries):
        self.gets += 1
        if self.gets == 1:
            raise KeyError("peer evicted after ranking")
        return {key: "value" for key in entries}

    async def put_batch(self, entries):
        self.puts.append(entries)


def test_an_eviction_after_ranking_makes_the_requester_reask():
    deployment = _RetryDeployment()
    plane = ReadThroughPlane()
    plane.attach(deployment)

    result, _trace = run_sim(plane.read_through("r0", {KEY: None}))

    assert result == {KEY: "value"}
    assert deployment.asks == 2
    assert deployment.actions == [Published((1, "r0")), Published((2, "r0"))]


def test_the_sensor_retains_no_publications_after_the_run():
    _result, plane = _run()
    directory = plane.sensor(DedupDirectorySensor)
    assert directory.in_flight() == set()
