"""The four shared types every capability plugs into.

:class:`~proposed.view.View` (sense), :class:`~proposed.selector.KeySelector` (decide),
:class:`~proposed.plane.DataPlane` (what follows a transfer) and
:class:`~realsim.runner.Runner` / :class:`~realsim.runner.ItemDispatch` (drive a run) are the generic half of both capabilities. These tests
pin the contract each one owes its callers:

1. the view reads the *real* directory and the run's virtual clock, and what it
   reports is the directory itself -- never an answer already put in some caller's
   preferred order;
2. the naive selector is the directory's own answer -- preferring what it names
   changes nothing, byte for byte, which is what lets a capability selector be
   swapped in as the only difference between two runs;
3. a preference reorders a directory answer to its ranked sources, a selection
   withholds itself until its readiness gate opens and crosses a service boundary
   without it -- and the two combinators built on it tell an *abstention* from the
   *naive answer*, carry a wrapped selector's gate through, and wake every selector
   they hold, whichever subtype (``KeySelector`` over keys, ``AnySelector`` over an
   application payload) that selector is. ``FirstMatch`` picks between alternatives;
   ``Refine`` funnels one ranking through the tests behind it;
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
from proposed import AnySelector, ControlPlane, Key, KeySelector, Selection
# Not re-exported by the package: what a deployment implements is one of the two
# subtypes, and these are implementations of them (or the base they share).
from proposed.selector import (
    AbstainOnSelf, FirstMatch, KeySelectorChain, NaiveKeySelector, prefer, Refine,
    Refinement, Selector, TakeHead,
)
from realsim.runner import ItemDispatch, Runner, WorkItem
from realsim.seams.link import LocalEndpoint
from realsim.seams.transport import Endpoint
from proposed import locality, nearest, View
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
    assert list(located["W"]) == ["a"]
    # Absent keys are simply missing -- a sensor reports, it does not raise.
    assert "absent" not in located
    assert now >= 2.0
    # Distance is arithmetic on the topology the view carries, not a read of it:
    # b is on a's node, c is not.
    assert locality(view.topology["a"], view.topology["b"]) is Tier.NVLINK
    assert locality(view.topology["a"], view.topology["c"]) is Tier.RDMA
    assert nearest(view.topology, ["c", "b"], "a") == "b"
    assert view.topology["a"].id == "vola"


def test_view_locate_reports_the_directory_and_not_a_preferred_answer():
    """A sensor reads what is there, whatever the caller of a read prefers.

    A control plane ranks the directory; handed an answer already put in somebody's
    order, it would be ranking its own output back.
    """
    sim = Simulation(_topology())

    async def scenario():
        with sim.mesh.installed():
            await sim.mesh.client_for("a").put("W", _payload())
            await sim.mesh.client_for("b").put("W", _payload())
            # A reader that was told to prefer b, and the same directory as sensed.
            sim.mesh.client_for("c", prefer=["b"])
            preferred = await sim.controller_handle.locate_volumes.call_one(["W"])
            return preferred, sim.view.locate(["W"])

    preferred, sensed = _drive(sim, scenario())
    assert list(preferred["W"]) == ["b"]          # what the caller asked for
    assert list(sensed["W"]) == ["a", "b"]        # what the directory holds


# --------------------------------------------------------------------------
# 2. The naive selector is the directory's own answer.
# --------------------------------------------------------------------------


class _Ranks(ControlPlane):
    """A plane whose one member answers with what a selector ranked.

    The shape every capability has: the plane is what a run holds and a caller asks,
    the selector is a utility it consults, and the gate is spent before the answer
    travels.
    """

    def __init__(self, selector: Selector) -> None:
        self.selector = selector

    def attach(self, view, transfer_cost) -> None:
        self.selector.attach(view, transfer_cost)

    async def sources(self, keys, requester) -> Selection:
        return await (await self.selector.select(list(keys), requester)).settled()


def _burst_trace(selector) -> str:
    """Run the same two-reader burst, asking ``selector`` or nobody; its trace."""
    trace = Trace()
    sim = Simulation(
        _topology(), trace=trace,
        control=_Ranks(selector) if selector is not None else None,
    )

    async def scenario():
        with sim.mesh.installed():
            await sim.mesh.client_for("a").put("W", _payload())
            for reader in ("b", "c"):
                chosen = (
                    await sim.control_plane_handle.sources.call_one(["W"], reader)
                    if sim.control_plane_handle is not None
                    else Selection()
                )
                await sim.client_for(reader, prefer=chosen.sources).get("W")
        return True

    _drive(sim, scenario())
    return trace.render()


def test_preferring_the_naive_answer_changes_nothing():
    assert _burst_trace(NaiveKeySelector()) == _burst_trace(None)


def test_no_preference_leaves_a_directory_answer_untouched():
    located = {"K": {"v1": "info1", "v0": "info0"}}
    assert prefer(located, Selection().sources) is located


# --------------------------------------------------------------------------
# 2b. One base, two subjects -- and a selector is a utility, not a plane.
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
    """Each kind is a subject and nothing more -- no second interface.

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
    """``Selector[X, ...]`` is the only place a subject is written; the value follows.

    Two annotations for one fact would drift. The parameter is what a reader sees
    and mypy checks; ``subject_type`` is the same type as something a gate can
    compare, since :pep:`484` erases the parameter at runtime.
    """

    class _Parameterised(Selector[Sequence[Key], None]):
        async def select(self, keys, requester):
            return Selection()

    class _Inherits(_Parameterised):        # narrows behaviour, not the subject
        pass

    class _PricesOnly(KeySelector[int]):    # binds the price, not the subject
        async def select(self, keys, requester):
            return Selection()

    class _DeclaresItsOwn(Selector):
        @property
        def subject_type(self):             # what Refine does
            return "computed"

        async def select(self, subject, requester):
            return Selection()

    assert _Parameterised.subject_type == Sequence[Key]
    assert _Inherits.subject_type == Sequence[Key]
    assert _PricesOnly.subject_type == Sequence[Key]
    assert _DeclaresItsOwn().subject_type == "computed"
    # An unbound parameter is not a subject: AnySelector is Selector[_S, _P].
    assert AnySelector.subject_type is Any


def test_taking_keys_is_not_the_same_claim_as_answering_for_the_store():
    """Why the kinds are types and not just a ``subject_type`` comparison.

    A funnel over a key selector takes keys and is still not one: its judgement is
    the application's, made while routing, and which volume serves a read is a
    different question. Reading ``subject_type`` alone could not tell them apart.
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


def test_a_plane_declares_whatever_it_wants_told():
    """A plane that gates on a write hears about it from whoever made the write.

    The directory has no notion that anything is listening: the plane names the
    member, the seam reads it off the plane like any other, and the data plane calls
    it after its put. Which is the same mechanism as a question -- there is no second
    kind of member here.
    """
    heard: list[tuple[str, tuple[str, ...]]] = []

    class _Hears(ControlPlane):
        async def sources(self, keys, requester):
            return Selection()

        async def published(self, requester, keys):
            heard.append((requester, tuple(keys)))

    sim = Simulation(_topology(), control=_Hears())
    assert sim.control_plane_handle.asked == ("published", "sources")
    _drive(sim, sim.control_plane_handle.published.call_one("a", ["W"]))
    assert heard == [("a", ("W",))]


def test_the_directory_holds_nothing_that_decides():
    """The control plane is not *in* the store: no hook, no callback, no selector.

    What the directory does with a preference is apply it, so there is nothing on the
    service for a capability to install itself into and nothing it calls back. Pinned
    mechanically, because the whole of this arrangement is what the service does
    *not* have.
    """
    service = Simulation(_topology()).mesh.directory.service
    assert {name for name in dir(service) if not name.startswith("_")} == {
        "controller", "keys", "locate_raw", "locate_volumes",
        "notify_delete", "notify_delete_batch", "notify_put_batch",
    }


def test_a_selector_is_a_utility_a_plane_holds_and_not_a_plane():
    """Which is why a run refuses one: there is nothing to hand it and nothing to ask.

    A selector has no ``cluster`` for a run to front and no questions of its own to
    declare -- ``select`` is what the plane holding it consults. So the thing a run
    takes is the plane, and the plane passes the ports down.
    """
    for kind in (KeySelector, AnySelector, Selector):
        assert not issubclass(kind, ControlPlane)
    with pytest.raises(TypeError, match="must be a ControlPlane"):
        Simulation(_topology(), control=_Fixed())

    inner = _Fixed(Selection.of(["a"]))
    sim = Simulation(_topology(), control=_Ranks(inner))
    answer = _drive(sim, sim.control_plane_handle.sources.call_one(["W"], "c"))
    assert answer.sources == ("a",)
    assert inner.asked == ["c"]      # the requester reached the utility unchanged


def test_a_plane_passes_the_run_s_view_down_to_what_it_ranks_with():
    """Fronting a plane and attaching it are one act of assembly.

    Which is what lets ``select`` take a subject and a requester and no view: the
    selector was handed the sensor when the plane was, by the plane. One never
    attached would sense through ``None``.
    """
    selector = _Fixed()
    sim = Simulation(_topology(), control=_Ranks(selector))
    assert selector.view is sim.view
    assert isinstance(selector.view, View)


# --------------------------------------------------------------------------
# 2c. One plane per run, fronted as a service, asked by the members it declares.
# --------------------------------------------------------------------------


class _Decides(ControlPlane):
    """A plane that answers what it was built with, and remembers who asked.

    What it answers with is between it and the hosts that ask -- ``kvcache_sim``'s
    scheduler answers with the winners of two selections, which is not a ``Selection``
    at all.
    """

    def __init__(self, answer: Any = "decided") -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def decide(self, subject, requester):
        self.asked.append(requester)
        return self.answer


def test_the_run_s_plane_is_fronted_as_a_service():
    """Reached over the hop, by the member it declares.

    Nothing looks up its type or names its members: the handle offers ``decide``
    because this plane declares ``decide``.
    """
    plane = _Decides()
    sim = Simulation(_topology(), control=plane)
    assert sim.control_plane_handle is not None
    assert sim.control_plane_handle.control is plane
    answer = _drive(sim, sim.control_plane_handle.decide.call_one("subject", "a"))
    assert answer == "decided"
    assert plane.asked == ["a"]


def test_a_handle_offers_the_members_the_plane_declares_and_no_others():
    """The seam is written once for every capability, however many questions it asks.

    ``ControlPlane`` declares a lifecycle and no questions, so the service and the
    handle read the plane's own public coroutines instead of naming any: a capability
    adding a second question adds nothing to ``realsim/seams``. Underscored members
    are the plane's working, and ``attach`` is the run's, so neither is reachable.
    """

    class _Two(ControlPlane):
        async def decide(self, subject, requester):
            return "decided"

        async def price(self, subject, requester):
            return "priced"

        async def _internal(self):           # its own working, not a question
            return "no"

        def ready(self):                     # not awaited, so not a question
            return True

    sim = Simulation(_topology(), control=_Two())
    handle = sim.control_plane_handle
    assert handle.asked == ("decide", "price")
    assert _drive(sim, handle.price.call_one("subject", "a")) == "priced"
    for hidden in ("_internal", "ready", "attach", "cluster"):
        assert not isinstance(getattr(handle, hidden, None), LocalEndpoint)


def test_a_run_with_no_plane_fronts_nothing():
    """No plane, no service, and a directory that answers for itself.

    The baseline path: nobody asks anything, so nobody hands the store a preference
    and every read is the ordinary one (``putget_sim``).
    """
    sim = Simulation(_topology())
    assert sim.control_plane_handle is None
    assert sim.cluster_handle is None


# --------------------------------------------------------------------------
# 3. Selection: ranking + readiness.
# --------------------------------------------------------------------------


def test_a_preference_reorders_a_directory_answer_to_its_ranked_sources():
    located = {"K": {"v0": "i0", "v1": "i1", "v2": "i2"}}
    preferred = prefer(located, Selection.of(["v2", "v0"]).sources)
    assert list(preferred["K"]) == ["v2", "v0"]  # rank order, v1 dropped


def test_a_preference_keeps_a_key_none_of_its_sources_holds():
    """A preference must not make data disappear."""
    located = {"K": {"v0": "i0"}}
    assert prefer(located, ("v9",)) == located


def test_a_settled_selection_has_waited_and_carries_no_gate():
    """What a plane reached as a service answers with: the ranking, no closure.

    Two properties in one call, because they are the same requirement seen twice --
    a caller in another process could not receive a gate, and would have nothing to
    wait on if it could.
    """
    opened = asyncio.Event()

    async def gate() -> None:
        await opened.wait()

    async def scenario():
        answer = asyncio.get_running_loop().create_task(
            Selection.of(["v0"], ready=gate, payload={"v0": 7}).settled()
        )
        await asyncio.sleep(0)
        assert not answer.done(), "answered before the source was usable"
        opened.set()
        return await answer

    settled, _ = run_sim(scenario())
    assert settled.sources == ("v0",)
    assert settled.ready is None
    assert settled.winner == 7        # the price rides along; the gate does not


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

    That is what lets it gate: a selector parks requesters on facts only its own
    bookkeeping will settle, and it never gets the chance to sense them if the chain
    only attaches what it consults.
    """
    front, behind = _Fixed(Selection.of(["v0"])), _Fixed(Selection.of(["v1"]))
    chained = FirstMatch([front, behind])
    chained.attach("a-view", "a-cost")
    _select(chained)                      # behind is never asked ...
    assert front.view is behind.view == "a-view"   # ... but is brought up


def test_a_combinator_holds_either_kind_and_is_neither():
    """Mixing kinds is allowed because a combinator claims neither.

    A combinator is a plain ``Selector`` whatever it wraps, so a chain of two key
    selectors says no more about its subject than one mixing in an application's --
    which is what makes the mixture harmless rather than something to reject.
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
    application's test asks something the store cannot read, so the funnel claims
    neither subject."""
    funnel = Refine(_Fixed(Selection.of(["v0"])), TakeHead())
    assert isinstance(funnel, Selector)
    assert not isinstance(funnel, (KeySelector, AnySelector))


# --------------------------------------------------------------------------
# 4. The plain path, and what a DataPlane declares (a lifecycle, no verbs).
# --------------------------------------------------------------------------


def test_the_plain_path_is_the_item_s_own_call():
    """No capability installed: the runner awaits the item and nothing surrounds it.

    Which is why ``ItemDispatch``'s one hook defaults to ``item.run`` -- a
    capability that adds something passes a member of its own plane instead, and
    that member owns the whole sequence.
    """
    calls: list[str] = []

    async def call():
        calls.append("ran")
        return 42

    item = WorkItem(id="i0", run=call)
    assert asyncio.run(ItemDispatch().execute(item)) == 42
    assert calls == ["ran"]


def test_a_data_plane_declares_a_lifecycle_and_no_verbs():
    """What a capability *does* is not this package's to name.

    ``attach`` is the whole surface, so the only thing a run can do with a plane is
    hand it the deployment; which member then carries the work is named by whoever
    wires the run. A declared verb would have to be one every capability implements,
    and the two here do not execute alike -- dedup reads through, kvcache walks
    three legs.
    """
    verbs = {
        name for name in vars(DataPlane)
        if not name.startswith("_")
    }
    assert verbs == {"attach"}
    assert DataPlane().attach(object()) is None   # a default that does nothing


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
    routes nothing and asks no control plane still crosses it.
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
