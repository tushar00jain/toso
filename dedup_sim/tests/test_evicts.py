"""Dedup over volumes that evict: the directory says who holds what, not a memory.

The burst in ``test_dedup.py`` reads one key once, on volumes with room for
everything, so every registration it makes is still true at the end of the run.
That is the easy half. A store whose volumes are bounded *unregisters* too -- a
new model version lands in a reader's cache and displaces the old one -- and a
routing selector that remembers "volume V holds key K" from the moment V registered
it will keep routing readers to V long after V dropped it.

So this run reads the same key twice with an eviction in between:

1. every reader gets ``W`` and read-throughs it into its own volume, which is the
   ordinary dedup chain -- one hop from the origin, the rest peer to peer;
2. every reader takes on the *next* version locally. The volumes are one payload
   deep, so that put evicts ``W`` and the volume tells the directory it is gone
   (:meth:`realsim.seams.volume_service.VolumeService._ask_for_room`);
3. every reader gets ``W`` again.

Round 3 is the whole test. The sensor's routes are unchanged from round 1 -- each
reader is still pointed at the peer ahead of it -- but none of those peers holds
``W`` any more, so each answer has to be withheld again until the peer's
read-through brings it back. Reading the directory is what makes that happen; a
remembered registration would say "ready now" and route every reader to a volume
that has nothing to serve, collapsing the chain back onto the origin.

The fabric assertion is therefore the same one dedup always makes, applied twice:
``W`` crosses from the origin exactly once *per read of it*, eviction or no
eviction.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests/test_evicts.py -q
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Callable, List, Optional

from putget_sim.workload.put_get import DEFAULT_N, KEY, PutGetBurst
from realsim.run import Result, Run
from realsim.runner import ItemDispatch, WorkItem
from realsim.simulation import Simulation
from sim_common.async_engine import run_sim
from sim_common.cost_model import DEFAULT_PROFILE

from dedup_sim.control._view import FanoutView
from dedup_sim.control.routing import Dedup
from proposed import Endpoint, Stored
from dedup_sim.data.read_through import ReadThroughPlane

#: The version that displaces ``W`` in a one-deep reader volume.
NEXT_VERSION = "W2"
#: Bytes of one payload -- and, below, of a whole reader volume.
PAYLOAD_BYTES = DEFAULT_N * 4
#: Virtual seconds between rounds. Orders of magnitude longer than the whole
#: burst takes, and asserted below, so a round observes the one before it whole.
ROUND = 1.0


class VersionedRounds(PutGetBurst):
    """The same burst, three rounds: read ``W``, cache the next version, read ``W``.

    Only :meth:`items` differs from the fixture -- the topology, the payload and
    the origin seed are the ones every other dedup test runs against, so the extra
    variable really is the eviction.

    Each item carries ``(reader, key)`` when the reader should read-through what it
    just fetched, and ``None`` when it should not: round 2's put is the reader
    storing its own new version, not a fetch to publish.
    """

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
                WorkItem(f"{reader}/3-read", 2 * ROUND, _get(reader, KEY), (reader, KEY)),
            ]
        return items


def _per_item(sim, workload) -> Any:
    """Drive each item: a read goes through the plane, anything else is its own call.

    The plane reads the key the item names -- rounds 1 and 3 -- while round 2 is the
    reader storing its *own* new version, which is the workload seeding a state and
    not a read to publish. No subclass: the plane already takes the key.
    """
    plane = ReadThroughPlane(KEY, workload.put_value)
    plane.attach(sim)

    async def drive(item: Any) -> Any:
        if item.payload is None:
            return await item.run()
        return await plane.read_through(*item.payload)

    return drive


def _run(num_readers: int = 3, *, fanout_cap: int = 1) -> tuple[Result, Dedup]:
    """One three-round run on volumes with room for exactly one payload."""
    # One payload per volume: the origin holds W and nothing else, and a reader
    # can cache exactly one version at a time -- which is what makes round 2 an
    # eviction rather than an accumulation.
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
    """``key -> {volume_id -> StorageInfo}`` as the run left it."""
    return result.sim.mesh.directory.controller.keys_to_storage_volumes


# --------------------------------------------------------------------------
# The rounds really are sequential, and round 2 really does evict.
# --------------------------------------------------------------------------


def test_the_rounds_do_not_overlap():
    """Each round finishes inside its own window, so round 3 sees round 2 whole."""
    result, _plane = _run()
    assert result.ledger.items_done == result.ledger.items_total == 9
    for row in result.ledger.rows:
        assert row.done < row.released + ROUND, row.id


def test_the_next_version_evicts_the_one_the_reader_cached():
    """A one-deep volume that takes on a new version drops the old one."""
    result, _plane = _run()
    for reader in result.workload.reader_ids:
        volume = result.sim.mesh.volumes[reader].service
        # Round 3 re-read W, which evicted the version round 2 put there: a
        # one-deep cache holds exactly one payload, whichever it is.
        assert sorted(volume.store.kv) == [KEY], reader
        assert volume.resident_bytes == PAYLOAD_BYTES, reader
    # ...and the directory was told each time, so it lists nobody for a version
    # no volume kept.
    assert NEXT_VERSION not in _directory(result)


# --------------------------------------------------------------------------
# The payoff: the fabric is 1x per read of W, across the eviction.
# --------------------------------------------------------------------------


def test_the_chain_re_forms_after_the_peers_evict():
    """Two reads of W, two crossings of the fabric -- not one per reader.

    Round 3's peers have all dropped W, so every answer but the first has to be
    withheld until the peer's read-through re-registers it. A selector answering
    from a remembered registration would release them at once and send them all
    to the origin (or to a volume holding nothing at all).
    """
    result, _plane = _run()
    origin = result.workload.origin_id

    assert result.ledger.origin_bytes == 2 * PAYLOAD_BYTES  # 1x per read of W
    origin_edges = [e for e in result.ledger.edges if e[0] == origin]
    assert len(origin_edges) == 2  # one hop out of the origin, per round
    # Six gets of W in all; the other four are peer to peer.
    assert result.ledger.transfer_bytes == 6 * PAYLOAD_BYTES


def test_every_reader_still_receives_the_payload_in_both_rounds():
    """Routing to a peer that has to re-fetch first is still a correct answer."""
    result, _plane = _run()
    expected = result.workload.expected
    for item_id, payload in result.results.items():
        if item_id.endswith("-read"):
            assert payload.shape == expected.shape, item_id
            assert payload.dtype == expected.dtype, item_id


def test_the_fabric_is_1x_per_read_for_any_fanout_cap():
    """The cap trades depth against wallclock; it does not spend fabric."""
    for cap in (1, 2, 3):
        result, _plane = _run(4, fanout_cap=cap)
        assert result.ledger.origin_bytes == 2 * PAYLOAD_BYTES, cap


# --------------------------------------------------------------------------
# And the sensor is no bigger at the end of the run than at the start.
# --------------------------------------------------------------------------


class _OwnNode(dict):
    """``volume id -> Endpoint``, inventing one per id: its own host and node."""

    def __missing__(self, volume_id: str) -> Endpoint:
        return Endpoint(id=volume_id, host=volume_id, node=volume_id)


class Directory(FanoutView):
    """The reads a chain makes, over a ``key -> volumes`` map a test controls.

    Waiting is the part of routing that fails by *not* happening, and the two
    ways it can go wrong -- one registration owed to several requesters, and a
    source with no registration owed at all -- both need the directory to change
    between two ``select`` calls at an exact moment. Staging that state directly
    says what the chain is being asked; arranging for a burst to produce it says
    considerably less.

    A view itself, so what the plane composes its sensor onto is this object
    (:meth:`derived`) and the links sense the real one. What writes it is
    :meth:`publish`, called where a run's put would have registered: before the action
    announcing it is dispatched.
    """

    def __init__(self, **holders: str) -> None:
        # No ports: every read below is staged here rather than sensed off a run.
        super().__init__(None, {}, None)
        self.by_key = {key: set(vols.split()) for key, vols in holders.items()}

    # -- the View surface Dedup uses -------------------------------------- #
    def derived(self, cls: type, **sensors: Any) -> "Directory":
        """Compose ``sensors`` onto this view, where the real one builds a fresh view.

        A fresh instance of ``cls`` would sense a run's ports, and the reads here are
        staged rather than sensed -- so the sensor goes onto this object, which is
        already a view.
        """
        assert isinstance(self, cls), f"{type(self).__name__} is not a {cls.__name__}"
        self._sensors.update(sensors)
        return self

    def locate_live(self, keys):
        """The port read staged, so :meth:`~proposed.view.View.locate` is the real one."""
        return {k: {v: None for v in sorted(self.by_key.get(k, ()))} for k in keys}

    @property
    def topology(self):
        """Every volume on a node of its own, so distance never breaks an id tie.

        ``nearest`` then answers in id order, which is what these tests stage
        against: which peer is *closest* is not what any of them is about.
        """
        return _OwnNode()

    def now(self) -> float:
        return 0.0

    # -- what the volumes do to it ------------------------------------------ #
    def publish(self, volume: str, key: str) -> None:
        """``volume``'s read-through lands: the directory gains it.

        What a real ``client.put`` does before it returns, which is why every caller
        below does this and *then* dispatches: a waiter woken by the commit re-reads
        this map and has to find the volume in it.
        """
        self.by_key.setdefault(key, set()).add(volume)

    def evict(self, volume: str, key: str) -> None:
        """``volume`` drops ``key`` (a newer version took the room)."""
        self.by_key[key].discard(volume)


def _sensing(directory: Directory, *, fanout_cap: int) -> Dedup:
    """A plane whose chain senses ``directory``, as a run's wiring leaves it.

    ``attach`` is the whole of it, as it is in a run: it composes the plane's own state
    onto the dispatcher it builds, and that state is the only thing a landed put folds.

    Nothing here is priced, so the transfer-cost half of the port is ``None``.
    """
    plane = Dedup(fanout_cap=fanout_cap)
    plane.attach(directory)
    return plane


def test_one_registration_releases_every_requester_waiting_on_it():
    """A fan-out cap over 1 means several requesters wait on the same peer.

    There is exactly one registration coming for all of them -- the peer publishes
    once -- so they share one gate, and that single registration has to wake all
    of them. (Dropping the released gate does not strand the second: each
    waiter holds the event itself, not a lookup of it.)

    The waiting is inside ``sources``: this plane answers over a service boundary, so
    a requester whose source is not usable yet is a requester whose *answer has not
    come back*. Which is why each ask is a task here -- there is no gate to hand out
    and park on separately.
    """

    async def _burst():
        directory = Directory(W="p")
        plane = _sensing(directory, fanout_cap=2)
        await plane.sources([KEY], "r0")  # r0 <- p, the origin: usable now
        answered: List[str] = []

        async def ask(name: str):
            selection = await plane.sources([KEY], name)
            answered.append(name)
            return selection

        loop = asyncio.get_running_loop()
        tasks = [loop.create_task(ask(name)) for name in ("r1", "r2")]
        await asyncio.sleep(0)
        assert answered == [], "answered before the peer published"
        # r0's put: the directory gains it, and then the one action settles the debt
        # r0 owed -- whose commit is what wakes r1 and r2 to re-read the directory.
        directory.publish("r0", KEY)
        plane.dispatcher.dispatch_sync(Stored("r0", KEY))
        waiters = await asyncio.gather(*tasks)
        assert all(w.sources == ("r0",) for w in waiters)
        # Nothing that crossed the boundary carries a closure with it.
        assert all(w.ready is None for w in waiters)
        return answered

    answered, _trace = run_sim(_burst())
    assert answered == ["r1", "r2"]  # both, in the order they parked


def test_a_peer_that_holds_nothing_and_owes_nothing_is_not_waited_for():
    """The liveness rule: a gate is only opened on a registration that is coming.

    This peer published and was then evicted, and -- unlike a peer mid-fetch -- it
    has no reason to fetch again, so nothing will ever record the fact. Waiting on
    it would hang the requester for the rest of the run, and naming it without
    waiting would route the requester to a volume holding nothing. So it stops
    being a source: this requester gets the directory's own answer, and the next
    one is assigned somewhere else.
    """

    async def _after_the_eviction():
        directory = Directory(W="p")
        plane = _sensing(directory, fanout_cap=3)
        await plane.sources([KEY], "r0")  # r0 <- p
        directory.publish("r0", KEY)                       # r0's read-through lands
        plane.dispatcher.dispatch_sync(Stored("r0", KEY))
        directory.evict("r0", KEY)  # ...and a newer version displaces it

        # It answers at all, which is the hang that would not: nothing to wait for,
        # and nobody to be sent to, so the directory's own answer.
        stale = await plane.sources([KEY], "r1")
        assert stale.sources is None, "parked on a registration nobody owes"

        # r1 is a live source though -- it is fetching now -- so r2 attaches to it
        # rather than to the peer that was retired. Its decision therefore gates,
        # which is why this asks the chain rather than awaiting the plane's answer.
        return await plane._chain.select([KEY], "r2")

    selection, _trace = run_sim(_after_the_eviction())
    assert selection.sources == ("r1",)
    assert selection.ready is not None  # r1's read-through is what it waits on


def test_a_requester_reassigned_after_a_retire_is_not_offered_a_second_time():
    """Being assigned twice does not make a requester a source twice over.

    A requester becomes a source for ``cap`` peers when it is first assigned one.
    Retiring its own source drops its route, so its next ask is assigned afresh --
    and an offer made there would hand it a whole second batch of slots. The
    queue cannot catch that on its own, because a queue only remembers the slots
    still *left*: a requester whose ``cap`` peers have all attached is absent from
    it and looks exactly like one that was never offered. So it would go on to
    feed ``2 x cap``, and the cap the run was configured with would not be the
    one it got.

    Asks the chain rather than the plane, because most of these requesters are
    routed to a peer that never publishes here: what is under test is who was
    assigned to whom, and ``sources`` would rightly still be waiting.
    """
    cap = 2

    async def _fanout_across_a_retire() -> int:
        directory = Directory(W="p")
        plane = _sensing(directory, fanout_cap=cap)
        # r0 pulls from the origin and publishes: a source for cap peers now.
        await plane.sources([KEY], "r0")
        directory.publish("r0", KEY)
        plane.dispatcher.dispatch_sync(Stored("r0", KEY))
        served = 0
        for i in range(cap):  # ...and every one of those slots is taken
            if (await plane._chain.select([KEY], f"r{i + 1}")).sources == ("r0",):
                served += 1
        # The origin drops the key, so r0's source holds nothing and owes nothing:
        # it is retired, and r0 is assigned afresh on the ask after that.
        directory.evict("p", KEY)
        await plane._chain.select([KEY], "r0")
        await plane._chain.select([KEY], "r0")
        for i in range(2 * cap):
            if (await plane._chain.select([KEY], f"x{i}")).sources == ("r0",):
                served += 1
        return served

    served, _trace = run_sim(_fanout_across_a_retire())
    assert served == cap


def test_the_sensor_remembers_no_registrations():
    """What it keeps is per requester, not per (volume, key) ever registered.

    Two rounds of ``W`` and a round of ``W2`` went through this sensor, and what is
    left names neither: the debts are settled, so nothing is outstanding, and the
    routes are one per reader whatever the run read. Nothing accumulates per
    ``(volume, key)`` -- not a registration, and not a waiter either, since who is
    waiting is not recorded anywhere at all
    (:meth:`proposed.dispatch.Dispatcher.gate`).
    """
    result, plane = _run()
    fanout = plane.view.fanout
    assert fanout._promised == set(), "a put owed by a run that finished"
    assert set(fanout._route) == set(result.workload.reader_ids)
    assert not hasattr(fanout, "_ready"), "the waiting is the commit, and is nobody's"
