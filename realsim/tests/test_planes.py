"""The four shared types every capability plugs into.

:class:`~proposed.view.View` (sense), :class:`~proposed.selector.KeySelector` (decide),
:class:`~proposed.plane.DataPlane` (what follows a transfer) and
:class:`~realsim.runner.Runner` / :class:`~realsim.runner.ItemDispatch` (drive a run) are the generic half of both capabilities. These tests
pin the contract each one owes its callers:

1. the view reads the *real* directory and the run's virtual clock, and reading
   it never re-enters the controller's routing hook;
2. the naive selector is the directory's own answer -- installing it changes
   nothing, byte for byte, which is what lets a capability selector be swapped in
   as the only difference between two runs;
3. a selection narrows a directory answer to its ranked sources, and withholds
   the answer until its readiness gate opens -- and the two combinators built on
   it tell an *abstention* from the *naive answer*, carry a wrapped selector's
   gate through, and wake every selector they hold, whichever subtype (``KeySelector``
   over keys, ``AnySelector`` over an application payload) that selector is.
   ``FirstMatch`` picks between alternatives; ``Refine`` funnels one ranking
   through the tests behind it, and is barred from the controller for the same
   reason;
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
from typing import Any, Sequence

import pytest
import torch

from realsim.mesh import Mesh
from realsim.simulation import Simulation
from proposed import DataPlane
from proposed import AnySelector, Key, KeySelector, Selection
# Not re-exported by the package: what a deployment implements is one of the two
# subtypes, and these are implementations of them (or the base they share).
from proposed.selector import (
    AbstainOnSelf, FirstMatch, KeySelectorChain, NaiveKeySelector, Refine,
    Refinement, Selector, TakeHead,
)
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
            located = view.locate(["W", "absent"])
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
    """A selector senses through the view, so the view must read the raw body."""

    seen: list[str] = []

    class _Counting(KeySelector):
        async def select(self, keys, requester):
            seen.append(requester)
            # Reading the directory from inside select must not recurse.
            self.view.locate(keys)
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
# 2. The naive selector is the directory's own answer.
# --------------------------------------------------------------------------


def _burst_trace(selector) -> str:
    """Run the same two-reader burst with/without a selector; return its trace."""
    trace = Trace()
    sim = Simulation(_topology(), trace=trace, control=selector)

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


def test_installing_the_naive_selector_changes_nothing():
    assert _burst_trace(NaiveKeySelector()) == _burst_trace(None)


def test_naive_selection_leaves_a_directory_answer_untouched():
    located = {"K": {"v1": "info1", "v0": "info0"}}
    assert Selection().narrow(located) is located


# --------------------------------------------------------------------------
# 2b. One base, two subjects: only a KeySelector reaches the controller.
# --------------------------------------------------------------------------


class _Answers:
    """A ``select`` body: return the selection it was handed, remember who asked.

    Shared by both subtypes below so that the *only* difference between them is
    the base -- which is the claim under test here, and lets section 3b hand a
    combinator either kind.
    """

    def __init__(self, selection: Selection | None = None) -> None:
        self.selection = selection if selection is not None else Selection()
        self.asked: list[str] = []

    async def select(self, subject, requester):
        self.asked.append(requester)
        return self.selection


class _Fixed(_Answers, KeySelector):
    """A selector that decides on command -- the store-question kind."""


class _FixedPlacement(_Answers, AnySelector):
    """The same body over an application payload -- the other kind."""


def test_the_subtypes_add_a_subject_and_nothing_else():
    """Each kind is a subject plus a place it is reached from -- no second interface.

    ``KeySelector`` names the one subject the store can hand down; ``AnySelector``
    leaves it open because this package cannot name a type an application invented.
    Neither adds behaviour, so the two cannot drift apart.
    """
    for kind in (KeySelector, AnySelector):
        assert kind.__bases__ == (Selector,)
        assert not vars(kind).keys() & {"select", "attach", "name"}
    assert KeySelector.subject_type == Sequence[Key]
    assert AnySelector.subject_type is Any


def test_a_subject_is_written_once_as_the_parameter_and_read_back_as_a_value():
    """``Selector[X]`` is the only place a subject is written; the value follows.

    Two annotations for one fact would drift. The parameter is what a reader sees
    and mypy checks; ``subject_type`` is the same type as something the install
    gate can compare, since :pep:`484` erases the parameter at runtime.
    """

    class _Parameterised(Selector[Sequence[Key]]):
        async def select(self, keys, requester):
            return Selection()

    class _Inherits(_Parameterised):        # narrows behaviour, not the subject
        pass

    class _DeclaresItsOwn(Selector):
        @property
        def subject_type(self):             # what Refine does
            return "computed"

        async def select(self, subject, requester):
            return Selection()

    assert _Parameterised.subject_type == Sequence[Key]
    assert _Inherits.subject_type == Sequence[Key]
    assert _DeclaresItsOwn().subject_type == "computed"
    # An unbound parameter is not a subject: AnySelector is Selector[_S].
    assert AnySelector.subject_type is Any


def test_taking_keys_is_not_the_same_claim_as_being_installable():
    """Why the kinds are types and not just a ``subject_type`` comparison.

    A funnel over a key selector takes keys and must still stay out of the
    directory: its judgement is the application's, made while routing, and the
    store-side answer is a different plane. A gate reading ``subject_type`` alone
    would install it.
    """
    funnel = Refine(_Fixed(Selection.of(["v0"])), TakeHead())
    assert funnel.subject_type == KeySelector.subject_type   # same subject ...
    assert not isinstance(funnel, (KeySelector, AnySelector))  # ... neither kind


def test_a_chain_of_key_selectors_is_one_and_a_mixed_chain_is_not():
    """The subject a chain hands down is the subject its links must take."""
    chain = KeySelectorChain([_Fixed(Selection.of([])), _Fixed(Selection.of(["v1"]))])
    assert isinstance(chain, KeySelector)
    assert chain.subject_type == Sequence[Key]
    with pytest.raises(TypeError, match="every link must be a KeySelector"):
        KeySelectorChain([_Fixed(), _FixedPlacement()])


def test_a_selector_hears_registrations_by_subscribing_for_itself():
    """The directory calls back plain callables and knows nothing of selectors.

    Subscribing in ``attach`` is what makes that possible: the view a selector is
    handed exposes the directory behind it, so the wakeup needs no member on
    ``Selector`` and no help from whoever assembles the run.
    """
    heard: list[tuple[str, tuple[str, ...]]] = []

    class _Subscribes(KeySelector):
        def attach(self, view, transfer_cost):
            super().attach(view, transfer_cost)
            view.directory.subscribe(
                lambda volume_id, keys: heard.append((volume_id, tuple(keys)))
            )

        async def select(self, keys, requester):
            return Selection()

    sim = Simulation(_topology(), control=_Subscribes())

    async def scenario():
        with sim.mesh.installed():
            sim.mesh.bind_source("a")
            await sim.mesh.client("a").put("W", _payload())

    _drive(sim, scenario())
    assert heard == [("a", ("W",))]


def test_a_registration_reaches_a_subscriber_before_the_put_returns():
    """Synchronous and inside ``notify_put_batch``: no window, no interleaving.

    A subscriber that could suspend would let a second registration land while
    the first was still being delivered, and a waiter released by the first would
    resume against a directory the second had already changed.
    """
    service = Simulation(_topology()).mesh.directory.service
    seen: list[str] = []
    service.subscribe(lambda volume_id, keys: seen.append(volume_id))
    assert not asyncio.iscoroutinefunction(type(service).subscribe)
    assert seen == []


def test_only_a_key_selector_is_consulted_by_the_controller():
    """A selector over a subject the store cannot read must not answer for it."""

    def _asked(control) -> list[str]:
        sim = Simulation(_topology(), control=control)

        async def scenario():
            with sim.mesh.installed():
                sim.mesh.bind_source("a")
                await sim.mesh.client("a").put("W", _payload())
                sim.mesh.bind_source("c")
                await sim.mesh.client("c").get("W")

        _drive(sim, scenario())
        return control.asked

    assert _asked(_Fixed()) == ["c"]
    assert _asked(_FixedPlacement()) == []


def test_an_installed_selector_is_attached_to_the_run_s_view():
    """The sensor a selector is consulted with is the one it was brought up with.

    A ``KeySelector`` is a ``ControlPlane``, so installing one in the directory and
    attaching it are the same act of assembly -- which is what lets ``select`` take
    a subject and a requester and no view. A selector installed but never attached
    would sense through ``None``.
    """
    selector = _Fixed()
    sim = Simulation(_topology(), control=selector)
    assert selector.view is sim.view
    assert isinstance(selector.view, View)


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
# 3b. Selector combinators: chaining and bounding, without a second verb.
# --------------------------------------------------------------------------


# The stand-ins are ``_Fixed`` / ``_FixedPlacement`` from section 2b: what a
# wrapped selector *decides* is its own business, so a combinator only needs one
# that decides on command and remembers that it was asked.


def _select(selector: Selector, keys=("K",), requester="r") -> Selection:
    """Run one ``select`` off any clock -- no store is involved."""
    return asyncio.run(selector.select(list(keys), requester))


def test_first_match_takes_the_first_answer_and_stops():
    first, second = _Fixed(Selection.of(["v0"])), _Fixed(Selection.of(["v1"]))
    assert _select(FirstMatch([first, second])).sources == ("v0",)
    assert second.asked == []  # never consulted: the chain had its answer


def test_first_match_falls_through_an_empty_ranking():
    """``Selection.of([])`` names nobody, so it is the abstention."""
    abstains, answers = _Fixed(Selection.of([])), _Fixed(Selection.of(["v1"]))
    assert _select(FirstMatch([abstains, answers])).sources == ("v1",)
    assert answers.asked == ["r"]


def test_first_match_stops_at_the_naive_answer():
    """``Selection()`` is a decision -- the directory's -- not silence.

    The opposite of the test above, and the distinction the combinator is built
    on: an empty *ranking* names nobody, while ``sources=None`` names everybody.
    """
    naive, behind = _Fixed(Selection()), _Fixed(Selection.of(["v1"]))
    assert _select(FirstMatch([naive, behind])).sources is None
    assert behind.asked == []


def test_an_exhausted_chain_abstains_so_chains_nest():
    """Running out is not a decision, which is what keeps chaining associative."""
    assert _select(FirstMatch([])).sources == ()
    assert _select(FirstMatch([_Fixed(Selection.of([]))])).sources == ()

    last = _Fixed(Selection.of(["v2"]))
    inner = FirstMatch([_Fixed(Selection.of([])), _Fixed(Selection.of([]))])
    assert _select(FirstMatch([inner, last])).sources == ("v2",)


def test_first_match_keeps_the_winner_s_readiness_gate():
    async def gate() -> None:
        return None

    chained = FirstMatch([_Fixed(Selection.of([])), _Fixed(Selection.of(["v1"], ready=gate))])
    assert _select(chained).ready is gate


def test_first_match_attaches_every_selector_even_unconsulted_ones():
    """A link behind an earlier answer still has to be brought up.

    That is what lets it subscribe: a selector parks requesters on gates nothing
    but its own registration opens, and it never gets the chance if the chain
    only attaches what it consults.
    """
    front, behind = _Fixed(Selection.of(["v0"])), _Fixed(Selection.of(["v1"]))
    chained = FirstMatch([front, behind])
    chained.attach("a-view", "a-cost")
    _select(chained)                      # behind is never asked ...
    assert front.view is behind.view == "a-view"   # ... but is brought up


def test_a_combinator_holds_either_kind_and_is_neither():
    """Mixing kinds is allowed because the mixture cannot reach the controller.

    A combinator is a plain ``Selector`` whatever it wraps, so a chain of two
    key selectors is no more installable than a chain mixing in an application's
    -- which
    is what makes the mixture harmless rather than something to reject.
    """
    mixed = FirstMatch([_Fixed(Selection.of([])), _FixedPlacement(Selection.of(["v1"]))])
    assert _select(mixed).sources == ("v1",)
    for combinator in (mixed, FirstMatch([_Fixed(Selection())])):
        assert isinstance(combinator, Selector)
        assert not isinstance(combinator, (KeySelector, AnySelector))


# --------------------------------------------------------------------------
# 3c. Refine: the other composition -- one ranking, narrowed step by step.
# --------------------------------------------------------------------------


class _Step(Refinement):
    """A refinement that narrows on command, and remembers it was asked."""

    def __init__(self, selection: Selection | None = None) -> None:
        self.selection = selection
        self.seen: list[tuple[tuple[str, ...], str]] = []

    async def refine(self, selection, subject, requester):
        self.seen.append((selection.sources, requester))
        return self.selection if self.selection is not None else selection


def test_refine_puts_the_ranking_through_every_step_in_order():
    first = _Step(Selection.of(["v1", "v2"]))
    second = _Step(Selection.of(["v2"]))
    funnel = Refine(_Fixed(Selection.of(["v0", "v1", "v2"])), first, second)

    assert _select(funnel).sources == ("v2",)
    assert first.seen == [(("v0", "v1", "v2"), "r")]
    assert second.seen == [(("v1", "v2"), "r")]  # handed what the first left


def test_an_abstaining_step_ends_the_funnel():
    """``Selection.of([])`` names nobody, so there is nothing left to narrow."""
    behind = _Step()
    funnel = Refine(_Fixed(Selection.of(["v0"])), _Step(Selection.of([])), behind)

    assert _select(funnel).sources == ()
    assert behind.seen == []


def test_refine_refuses_a_ranking_that_names_everybody():
    """``Selection()`` is the directory's whole answer, which no step can narrow.

    The opposite reading from ``FirstMatch``, where it is the decision that wins
    the chain. Handing one to a step would quietly return an unnarrowed ranking,
    so it is a wiring error and says so.
    """
    with pytest.raises(ValueError, match="cannot narrow"):
        _select(Refine(_Fixed(Selection()), _Step()))

    # ... and with nothing behind it there is no narrowing to contradict.
    assert _select(Refine(_Fixed(Selection()))).sources is None


def test_a_narrowed_selection_keeps_its_gate_and_the_prices_it_kept():
    """What a step drops is sources, not what the source said about them."""

    async def gate() -> None:
        return None

    ranked = Selection.of(["v0", "v1"], ready=gate, payload={"v0": 7, "v1": 9})
    narrowed = _select(Refine(_Fixed(ranked), TakeHead()))

    assert narrowed.sources == ("v0",)
    assert narrowed.ready is gate
    assert narrowed.winner == 7
    assert "v1" not in narrowed.payload  # dropped with the source it priced


def test_abstain_on_self_drops_the_whole_ranking_not_just_the_head():
    """A requester ranked first is preferred to every peer behind it."""
    ranked = _Fixed(Selection.of(["r", "v1"]))
    assert _select(Refine(ranked, AbstainOnSelf(), TakeHead())).sources == ()
    assert _select(Refine(ranked, AbstainOnSelf(), TakeHead()), requester="v0") \
        .sources == ("r",)


def test_refine_brings_up_the_source_and_every_step():
    """One view for the whole funnel: a step senses through what the source did."""
    source, step = _Fixed(), _Step()
    funnel = Refine(source, step)
    funnel.attach("a-view", "a-cost")

    assert funnel.view is source.view is step.view == "a-view"


def test_refine_is_a_plain_selector_whatever_it_narrows():
    """Same bar as ``FirstMatch``: narrowing a selector's ranking with an
    application's test asks something the store cannot read, so the result is
    barred from the controller by being neither subtype."""
    funnel = Refine(_Fixed(Selection.of(["v0"])), TakeHead())
    assert isinstance(funnel, Selector)
    assert not isinstance(funnel, (KeySelector, AnySelector))


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
    routes nothing and installs no selector still crosses it.
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
