"""The four shared types every capability plugs into.

:class:`~proposed.view.View` (sense), :class:`~proposed.policy.Policy` (decide),
:class:`~proposed.plane.DataPlane` (what follows a transfer) and
:class:`~realsim.runner.Runner` / :class:`~realsim.runner.ItemDispatch` (drive a run) are the generic half of both capabilities. These tests
pin the contract each one owes its callers:

1. the view reads the *real* directory and the run's virtual clock, and reading
   it never re-enters the controller's routing hook;
2. the naive policy is the directory's own answer -- installing it changes
   nothing, byte for byte, which is what lets a capability policy be swapped in
   as the only difference between two runs;
3. a selection narrows a directory answer to its ranked sources, and withholds
   the answer until its readiness gate opens;
4. the data plane's two methods default to real behaviour (run the call, do
   nothing after), so a capability overrides one method rather than filling in
   a stub;
5. the runner releases items in ``(release_time, id)`` order on the virtual
   clock, installs the mesh once, and records one ledger row per item.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest realsim/tests/test_planes.py -q
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from realsim.mesh import Mesh
from realsim.simulation import Simulation
from proposed import DataPlane
from proposed import Policy, Selection
from proposed.policy import NaivePolicy  # not exported: the base Policy is naive
from realsim.runner import ItemDispatch, Runner, WorkItem
from realsim.seams.transport import Endpoint
from proposed import View
from sim_common.async_engine import run_sim
from sim_common.report import Ledger
from sim_common.topology import Tier
from sim_common.trace import Trace


def _drive(sim, coro):
    """Run one coroutine on an assembled stack's clock (no workload involved)."""
    try:
        return sim.loop.run_until_complete(coro)
    finally:
        sim.loop.close()


def _topology() -> dict[str, Endpoint]:
    """Three nodes: a and b share a node (NVLink), c is remote (cross-node)."""
    return {
        "a": Endpoint(id="vola", host="hostA", node="node0"),
        "b": Endpoint(id="volb", host="hostB", node="node0"),
        "c": Endpoint(id="volc", host="hostC", node="node1"),
    }


def _payload():
    return torch.empty(16, dtype=torch.float32, device="meta")


# --------------------------------------------------------------------------
# 1. View: a real directory read + the virtual clock, no mutation.
# --------------------------------------------------------------------------


def test_view_reads_the_real_directory_topology_and_clock():
    sim = Simulation(_topology())
    view = sim.view

    async def scenario():
        with sim.mesh.installed():
            sim.mesh.bind_source("a")
            await sim.mesh.client("a").put("W", _payload())
            await asyncio.sleep(2.0)
            located = await view.locate(["W", "absent"])
            return located, view.now()

    (located, now) = _drive(sim, scenario())
    assert View.holders(located, "W") == ["a"]
    # Absent keys are simply missing -- a sensor reports, it does not raise.
    assert View.holders(located, "absent") == []
    assert now >= 2.0
    # Topology reads: b is on a's node, c is not.
    assert view.locality("a", "b") is Tier.NVLINK
    assert view.locality("a", "c") is Tier.RDMA
    assert view.nearest(["c", "b"], "a") == "b"
    assert view.endpoint("a").id == "vola"


def test_view_locate_does_not_re_enter_the_routing_hook():
    """A policy senses through the view, so the view must read the raw body."""

    seen: list[str] = []

    class _Counting(Policy):
        async def select(self, view, keys, requester):
            seen.append(requester)
            # Reading the directory from inside select must not recurse.
            await view.locate(keys)
            return Selection()

    sim = Simulation(_topology(), control=_Counting())

    async def scenario():
        with sim.mesh.installed():
            sim.mesh.bind_source("a")
            await sim.mesh.client("a").put("W", _payload())
            sim.mesh.bind_source("c")
            return await sim.mesh.client("c").get("W")

    got = _drive(sim, scenario())
    assert got is not None
    # Exactly one consultation: c's get. The put never locates, and the view read
    # inside select bypasses the hook.
    assert seen == ["c"]


# --------------------------------------------------------------------------
# 2. The naive policy is the directory's own answer.
# --------------------------------------------------------------------------


def _burst_trace(policy) -> str:
    """Run the same two-reader burst with/without a policy; return its trace."""
    trace = Trace()
    sim = Simulation(_topology(), trace=trace, control=policy)

    async def scenario():
        with sim.mesh.installed():
            sim.mesh.bind_source("a")
            await sim.mesh.client("a").put("W", _payload())
            sim.mesh.bind_source("b")
            await sim.mesh.client("b").get("W")
            sim.mesh.bind_source("c")
            await sim.mesh.client("c").get("W")
        return True

    _drive(sim, scenario())
    return trace.render()


def test_installing_the_naive_policy_changes_nothing():
    assert _burst_trace(NaivePolicy()) == _burst_trace(None)


def test_naive_selection_leaves_a_directory_answer_untouched():
    located = {"K": {"v1": "info1", "v0": "info0"}}
    assert Selection().narrow(located) is located


# --------------------------------------------------------------------------
# 3. Selection: ranking + readiness.
# --------------------------------------------------------------------------


def test_selection_narrows_to_its_ranked_sources():
    located = {"K": {"v0": "i0", "v1": "i1", "v2": "i2"}}
    narrowed = Selection.of(["v2", "v0"]).narrow(located)
    assert list(narrowed["K"]) == ["v2", "v0"]  # rank order, v1 dropped


def test_selection_keeps_a_key_no_selected_source_holds():
    """A preference must not make data disappear."""
    located = {"K": {"v0": "i0"}}
    assert Selection.of(["v9"]).narrow(located) == located


def test_selection_withholds_until_its_gate_opens():
    opened = asyncio.Event()
    order: list[str] = []

    async def gate() -> None:
        await opened.wait()

    async def waiter():
        await Selection.of(["v0"], ready=gate).wait()
        order.append("released")

    async def opener():
        await asyncio.sleep(1.0)
        order.append("opened")
        opened.set()

    async def scenario():
        await asyncio.gather(waiter(), opener())
        return True

    ok, _ = run_sim(scenario())
    assert ok
    assert order == ["opened", "released"]


# --------------------------------------------------------------------------
# 4. DataPlane defaults.
# --------------------------------------------------------------------------


def test_data_plane_defaults_run_the_call_and_do_nothing_after():
    """The two defaults are real behaviour: run the item, do nothing after.

    They now live on either side of the boundary -- running an item is the
    runner's ``ItemDispatch``, what follows it is the capability's ``DataPlane``
    -- so the plain path is the two of them composed, which is what a run with no
    capability installed gets.
    """
    calls: list[str] = []

    async def call():
        calls.append("ran")
        return 42

    item = WorkItem(id="i0", run=call)

    async def scenario():
        result = await ItemDispatch().execute(item)
        assert await ItemDispatch().after(item, result) is None
        assert await DataPlane().after(item.id, result) is None
        return result

    assert asyncio.run(scenario()) == 42
    assert calls == ["ran"]


# --------------------------------------------------------------------------
# 5. Runner: release order, one install, one row per item.
# --------------------------------------------------------------------------


def test_runner_releases_in_time_then_id_order_and_records_rows():
    started: list[str] = []

    def item(item_id: str, at: float) -> WorkItem:
        async def call():
            started.append(item_id)
            return item_id.upper()

        return WorkItem(id=item_id, release_time=at, run=call)

    mesh = Mesh(_topology())
    ledger = Ledger()
    runner = Runner(mesh, ledger=ledger)
    # Deliberately out of order, with a tie at t=0 to exercise the id tie-break.
    items = [item("b", 0.0), item("late", 5.0), item("a", 0.0)]

    results, _ = run_sim(runner.run(items))

    assert started == ["a", "b", "late"]
    assert results == {"a": "A", "b": "B", "late": "LATE"}
    assert ledger.items_total == 3
    assert ledger.items_done == 3
    assert {r.id: r.done for r in ledger.rows} == {"a": 0.0, "b": 0.0, "late": 5.0}
    assert ledger.wallclock == 5.0


def test_the_runner_waits_for_an_item_and_for_nothing_else():
    """The gather is the whole wait: there is no drain pass behind it.

    ``ItemDispatch`` used to take an ``on_drain`` awaited after every item, and
    the one capability that passed it was covering for a request whose decode
    outlived its own coroutine. That was fixed where it belonged (the decode leg
    now answers at the last token), and the hook was deleted rather than left for
    the next capability to reach for -- so this asserts both halves: an item that
    takes time is waited for, and the dispatch has nowhere to hang work that is
    not an item's.
    """
    mesh = Mesh(_topology())

    async def call():
        await asyncio.sleep(3.0)
        return asyncio.get_running_loop().time()

    results, _ = run_sim(Runner(mesh).run([WorkItem(id="i0", run=call)]))
    assert results == {"i0": 3.0}      # the clock the item's own coroutine saw
    with pytest.raises(TypeError):
        ItemDispatch(on_drain=call)


def test_runner_installs_the_mesh_exactly_once_and_releases_it():
    from realsim.seams import factory

    owners: list[object] = []

    async def call():
        owners.append(factory.current_owner())

    mesh = Mesh(_topology())
    run_sim(Runner(mesh).run([WorkItem(id="i0", run=call), WorkItem(id="i1", run=call)]))
    assert owners == [mesh, mesh]
    assert factory.current_owner() is None


# --------------------------------------------------------------------------
# Service hops: what reaching a service costs (realsim.seams.link).
# --------------------------------------------------------------------------


def _end_of_run(result) -> float:
    """The virtual clock at the last traced event."""
    return max(t for t, _kind, _msg in result.trace.events)


def test_a_service_hop_is_free_and_inline_by_default():
    """The default seam changes nothing: awaiting a non-suspending call is inline."""
    from sim_common import config

    from realsim.tests._burst import run_burst

    baseline = run_burst(3)
    with config.overrides(controller_rtt=0.0):
        explicit = run_burst(3)
    assert explicit.trace.render() == baseline.trace.render()


def test_the_controller_hop_is_charged_to_every_capability():
    """Even the baseline pays it -- it reaches the directory like everyone else.

    This boundary used to be the one seam charging nothing: the transport charged
    bytes and the coordinator handle charged its round trip, while every
    ``locate_volumes`` / ``notify_put_batch`` in the repo was free. A burst that
    routes nothing and installs no policy still crosses it.
    """
    from sim_common import config

    from realsim.tests._burst import run_burst

    rtt = 0.25
    free = run_burst(3)
    with config.overrides(controller_rtt=rtt):
        distant = run_burst(3)
    # Every directory call now costs a round trip, so the run ends later -- by at
    # least one, since the reads that remain on the critical path are serialized.
    assert _end_of_run(distant) >= _end_of_run(free) + 2 * rtt
