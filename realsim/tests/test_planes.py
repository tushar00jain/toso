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
   without it, and an ordering link -- ``Sort`` / ``Max`` -- is the only thing that puts
   it in an order; the combinators built on it tell an *abstention* from the *naive
   answer*, carry a wrapped selector's gate, key and prices through, and wake every
   selector they hold -- off the view that selector declared -- whether or not they
   consult it. ``FirstMatch`` picks between alternatives, ``Balance`` appends the load
   on the sources one answer named as a further dimension of its key (what to make of
   that is the fold ``Folded`` stamps), and a ranking narrows itself in place
   (``require``, ``take``);
4. the data plane's two methods default to real behaviour (run the call, do
   nothing after), so a capability overrides one method rather than filling in
   a stub;
5. the runner releases items in ``(release_time, id)`` order on the virtual
   clock, installs the mesh once, and records one ledger row per item;
6. a plane declares where in its own answer the address of another host is, and is
   otherwise untouched by saying so; a :class:`~proposed.routed.RoutedPlane` calls
   the member again there for any caller -- through the reference, so every hop is
   charged -- and neither a cycle nor a host holding a peer survives.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest realsim/tests/test_planes.py -q
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

import pytest
import torch

from realsim.mesh import Mesh
from realsim.seams.data_plane_service import DataPlaneService
from realsim.simulation import Simulation
from proposed import DataPlane, routed, RoutedPlane
from proposed import ControlPlane, Key, KeySelector, Selection
# Not re-exported by the package: what a deployment implements is one of the two
# subtypes, and these are implementations of them (or the base they share).
from proposed.selector import (
    Annotate, Balance, Const, declares, FirstMatch, Folded, Max, NaiveKeySelector,
    prefer, Selector, Sort,
)
from realsim.runner import ItemDispatch, Runner, WorkItem
from realsim.seams.link import LocalEndpoint
from realsim.seams.transport import Endpoint
from proposed import locality, LoadView, nearest, Sensed, SensorView, View
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
    # Absent keys are simply missing -- a view reports, it does not raise.
    assert "absent" not in located
    assert now >= 2.0
    # Distance is arithmetic on the topology the view carries, not a read of it:
    # b is on a's node, c is not.
    assert locality(view.topology["a"], view.topology["b"]) is Tier.NVLINK
    assert locality(view.topology["a"], view.topology["c"]) is Tier.RDMA
    assert nearest(view.topology, ["c", "b"], "a") == "b"
    assert view.topology["a"].id == "vola"
    # One view per mesh, and it is the one the run hands the control plane: a fresh
    # view per access would carry a pin scope of its own, so a reader holding it would
    # walk the directory inside a decision that had pinned it (View.pinned).
    assert sim.mesh.view is sim.mesh.view is view


def test_view_locate_reports_the_directory_and_not_a_preferred_answer():
    """A view reads what is there, whatever the caller of a read prefers.

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

    def attach(self, view) -> None:
        self.selector.attach(view)

    async def sources(self, keys, requester) -> Selection:
        return await self.selector.select(list(keys), requester).settled()


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

    def select(self, subject, requester):
        self.asked.append(requester)
        return self.selection


class _Fixed(_Answers, KeySelector):
    """A selector that decides on command -- the store-question kind."""


class _FixedPlacement(_Answers, Selector[Any]):
    """The same body over an application payload -- the other subject."""


def test_the_named_subject_is_a_subject_and_nothing_else():
    """``KeySelector`` is a subject and nothing more -- no second interface.

    It names the one subject the store can hand down, so it cannot drift from the base.
    An application's own subject needs no name here, being ``Selector[ThatSubject]``.
    """
    assert KeySelector.__bases__ == (Selector,)
    assert not vars(KeySelector).keys() & {"select", "attach", "name"}
    assert KeySelector.subject_type == Sequence[Key]


def test_a_subject_is_written_once_as_the_parameter_and_read_back_as_a_value():
    """``Selector[X]`` is the only place a subject is written; the value follows.

    Two annotations for one fact would drift. The parameter is what a reader sees
    and mypy checks; ``subject_type`` is the same type as something a gate can
    compare, since :pep:`484` erases the parameter at runtime.
    """

    class _Parameterised(Selector[Sequence[Key]]):
        def select(self, keys, requester):
            return Selection()

    class _Inherits(_Parameterised):        # narrows behaviour, not the subject
        pass

    class _Bare(KeySelector):               # binds nothing, so it inherits the subject
        def select(self, keys, requester):
            return Selection()

    class _DeclaresItsOwn(Selector):
        @property
        def subject_type(self):             # computed, not read off a base
            return "computed"

        def select(self, subject, requester):
            return Selection()

    assert _Parameterised.subject_type == Sequence[Key]
    assert _Inherits.subject_type == Sequence[Key]
    assert _Bare.subject_type == Sequence[Key]
    assert _DeclaresItsOwn().subject_type == "computed"
    # An unbound parameter is not a subject: the base is Generic[_S].
    assert Selector.subject_type is Any


def test_taking_keys_is_not_the_same_claim_as_answering_for_the_store():
    """The type says which, and a chain does not ask: it checks the subject.

    A selector can take keys and still not answer for the store -- its judgement may be
    the application's, made while routing, and which volume serves a read is a different
    question (``kvcache_sim``'s ``LocalOnly``). So the class is what says which claim is
    being made, while what a chain needs of a link is only that it takes the subject the
    chain hands down.
    """

    class _KeysButNotTheStore(Selector[Sequence[Key]]):
        def select(self, subject, requester):
            return Selection.of([])

    assert _KeysButNotTheStore.subject_type == KeySelector.subject_type  # same subject
    assert not isinstance(_KeysButNotTheStore(), KeySelector)            # other claim
    FirstMatch([_Fixed(), _KeysButNotTheStore()])                        # a link either way


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

    A selector has no sensor for a run to front and no questions of its own to
    declare -- ``select`` is what the plane holding it consults. So the thing a run
    takes is the plane, and the plane passes the ports down.
    """
    for kind in (KeySelector, Selector):
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
    selector was handed the view when the plane was, by the plane. One never
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
    for hidden in ("_internal", "ready", "attach", "sensor"):
        assert not isinstance(getattr(handle, hidden, None), LocalEndpoint)


def test_a_run_with_no_plane_fronts_nothing():
    """No plane, no service, and a directory that answers for itself.

    The baseline path: nobody asks anything, so nobody hands the store a preference
    and every read is the ordinary one (``putget_sim``).
    """
    sim = Simulation(_topology())
    assert sim.control_plane_handle is None
    assert sim.dispatcher_handle is None


# --------------------------------------------------------------------------
# 3. Selection: what a stage annotates, what a fold makes of it, and readiness.
# --------------------------------------------------------------------------
# The ordering links are asked over a ``Const``, which is what a base contributes to a
# chain: a selection somebody else built, handed on whatever the subject.


def test_only_an_ordering_link_puts_a_selection_in_an_order():
    """A stage keys; ``Sort`` orders and ``Max`` picks, off the dimensions it left.

    With no fold the dimensions are compared as they stand, which is what a chain of
    stages costs when nothing has to be blended -- and a selection no stage keyed keeps
    the order it was built with, since there is nothing to beat it.
    """
    keyed = Selection.keyed([("v0", (9,)), ("v1", (2,))])
    assert keyed.sources == ("v0", "v1")            # built, not ranked
    assert _select(Sort(Const(keyed))).sources == ("v1", "v0")
    assert _select(Max(Const(keyed))).sources == ("v1",)
    priced = Const(Selection.priced([("v0", 9), ("v1", 2)]))
    assert _select(Sort(priced)).sources == ("v1", "v0")
    unkeyed = Selection.of(["v1", "v0"])
    assert _select(Sort(Const(unkeyed))) is unkeyed
    assert _select(Max(Const(unkeyed))).sources == ("v1",)


def test_a_fold_blends_the_dimensions_the_stages_left():
    """The dimensions are positional, and a fold is any arithmetic over them.

    Which is what a re-sort could not express: two numbers traded against each other
    rather than one applied after the other. A stage that never ran is a dimension that
    is not there, so a fold reaching for it raises instead of comparing whatever landed
    in that position.
    """
    two = Const(Selection.keyed([("v0", (4, 3)), ("v1", (5, 0))]))
    assert _select(Sort(two)).sources == ("v0", "v1")                     # 4 < 5
    blended = Folded(two, lambda d: d[0] * (1 + d[1]))
    assert _select(Sort(blended)).sources == ("v1", "v0")                 # 16 > 5
    short = Folded(Const(Selection.priced([("v0", 4)])), lambda d: d[0] + d[1])
    with pytest.raises(IndexError):
        _select(Sort(short))


def test_a_fold_rides_on_the_answer_so_nothing_that_orders_it_names_one():
    """``Folded`` writes the fold; ``Sort`` and ``Max`` read it and take no argument.

    Which is what stops two callers of one ranking folding it two different ways -- and
    stamping one orders nothing by itself.
    """
    ranking = Folded(
        Const(Selection.keyed([("v0", (4, 3)), ("v1", (5, 0))])),
        lambda d: d[0] * (1 + d[1]),
    )
    stamped = _select(ranking)
    assert stamped.fold is not None
    assert stamped.sources == ("v0", "v1")                  # built, not ranked
    assert _select(Sort(ranking)).sources == ("v1", "v0")
    assert _select(Max(ranking)).sources == ("v1",)


def test_the_id_breaks_every_tie_in_one_place():
    """Two sources the stages cannot separate go id-ascending, folded or not.

    In one place, so a run reproduces however many stages annotated the selection and
    whatever the fold made of them.
    """
    tied = Const(Selection.keyed([("v1", (5, 1)), ("v0", (5, 1))]))
    assert _select(Sort(tied)).sources == ("v0", "v1")
    assert _select(Sort(Folded(tied, lambda d: 0))).sources == ("v0", "v1")


def test_both_empties_survive_an_ordering():
    """Neither names a source, so there is nothing to order and nothing to pick."""
    for empty in (Selection.of([]), Selection()):
        assert _select(Sort(Const(empty))) is empty
        assert _select(Max(Const(empty))) is empty


def test_max_keeps_the_gate_and_the_winner_s_key():
    """A plane orders and *then* spends the gate, so the pick has to carry it."""

    async def gate() -> None:
        return None

    best = _select(
        Max(Const(Selection.keyed([("v0", (9,)), ("v1", (2,))], ready=gate)))
    )
    assert best.head == "v1"
    assert best.ready is gate
    assert best.key == {"v1": (2,)}      # what the stages measured about the pick
    assert "v0" not in best.key          # dropped with the source it spoke for


class _Counted:
    """A fold reading that tallies the comparisons an ordering link spends on it."""

    compared = 0

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return self.value == other.value                        # type: ignore[attr-defined]

    def __lt__(self, other: "_Counted") -> bool:
        type(self).compared += 1
        return self.value < other.value


def test_max_takes_the_winner_in_one_pass_and_sort_would_leave_it_in_front():
    """A chain wanting a head pays for a head, and gets the one an order would give it.

    The two links have to agree, or the same pool ranked and picked would name different
    sources: they share the comparable, which admits no ties, so the least of it is
    exactly what a full order puts first. What ``Max`` does *not* spend is the ordering of
    the sources behind that one -- ``n - 1`` comparisons, not ``n log n``.
    """
    pool = Selection.keyed([(f"v{i}", (v,)) for i, v in enumerate([3, 7, 1, 8, 2, 6, 5, 4])])
    folded = Folded(Const(pool), lambda d: _Counted(d[0]))

    _Counted.compared = 0
    assert _select(Max(folded)).sources == ("v2",)               # the 1
    assert _Counted.compared == len(pool.sources or ()) - 1

    _Counted.compared = 0
    ordered = _select(Sort(folded)).sources
    assert ordered[0] == "v2"                                   # the same winner
    assert _Counted.compared > len(pool.sources or ()) - 1       # and the losers cost extra

    # Every case an unordered pool can be in, the two links still agreeing on the head.
    for ranking in (
        Const(pool),
        folded,
        Const(Selection.keyed([("v1", (5, 1)), ("v0", (5, 1))])),    # tie -> id
        Const(Selection.of(["v1", "v0"])),                          # keyed by nothing
    ):
        assert _select(Max(ranking)).head == _select(Sort(ranking)).head


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
            Selection.priced([("v0", 7)], ready=gate).settled()
        )
        await asyncio.sleep(0)
        assert not answer.done(), "answered before the source was usable"
        opened.set()
        return await answer

    settled, _ = run_sim(scenario())
    assert settled.sources == ("v0",)
    assert settled.ready is None
    assert settled.key == {"v0": (7,)}   # the key rides along; the gate does not


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
    return selector.select(list(keys), requester)


class _Load:
    """Per-source load, moved by hand: what a ``Balance`` reads off its view."""

    def __init__(self, **counts: int) -> None:
        self.counts = dict(counts)

    def named(self):
        return self.counts

    def sent(self, source) -> None:
        """What a decision naming ``source`` does to it."""
        self.counts[source] = self.counts.get(source, 0) + 1


class _Senses:
    """The whole of what a ``Balance`` senses: one load sensor, and no directory."""

    def __init__(self, load: Optional[_Load] = None) -> None:
        self.load = load if load is not None else _Load()

    def now(self) -> float:
        return 0.0

    def subset(self, *views: type) -> "_Senses":
        """Every declaration reaches this same stand-in: there is one read here."""
        return self


def _bounded(bound: int):
    """A fold in which load may cost a source ``bound`` of price and no more.

    The knob every caller of this combinator ends up wanting, written where it belongs
    -- in a ``Folded`` over the two dimensions the pairing leaves: the price, then the
    load. The raw count rides behind the bounded number, so two sources the bound has
    levelled keep alternating.
    """
    def fold(dims):
        price, load = dims
        return (price + min(load, bound), load)

    return fold


def _heads(selector: Selector, *, count: int, moving: bool = True) -> list:
    """The winner of ``count`` successive asks, the load moving as each is decided.

    The chain production declares: the stages key, a ``Max`` takes one, the decision that
    follows names that source, and the sensor counts it. ``moving=False`` is a load
    nothing moves.
    """
    senses = _Senses()
    best = Max(selector).attach(senses)
    heads = []
    for _ in range(count):
        head = _select(best).head
        heads.append(head)
        if moving:
            senses.load.sent(head)
    return heads


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


def test_the_abstention_is_the_identity_of_choosing_between_answers():
    """A link that abstains cannot change what the rest of a chain answers.

    On either side of it, which is what lets ``FirstMatch`` seed its fold with an
    abstention and lets a gate be wired anywhere in a chain.
    """
    answer = Selection.of(["v0"])
    assert Selection.abstain().otherwise(answer) is answer
    assert answer.otherwise(Selection.abstain()) is answer


def test_choosing_between_answers_is_associative():
    """Where a chain is bracketed cannot change its answer, so chains nest.

    The property ``FirstMatch([FirstMatch([a, b]), c])`` rests on, stated at the
    operation underneath it rather than only at the combinator.
    """
    empty, naive = Selection.abstain(), Selection.universe()
    for a in (empty, naive, Selection.of(["v0"])):
        for b in (empty, Selection.of(["v1"])):
            for c in (empty, Selection.of(["v2"])):
                assert a.otherwise(b).otherwise(c) == a.otherwise(b.otherwise(c))


def test_the_naive_answer_absorbs_whatever_would_be_chosen_after_it():
    """``Selection.universe`` decides, so nothing chosen after it is reachable.

    The asymmetry between the two empties, as a law: only an abstention falls through,
    and ``sources is None`` names every holder rather than nobody.
    """
    naive = Selection.universe()
    assert naive.otherwise(Selection.of(["v1"])) is naive
    assert not naive.abstains
    assert Selection.abstain().abstains


def test_first_match_keeps_the_winner_s_readiness_gate():
    async def gate() -> None:
        return None

    chained = FirstMatch([_Fixed(Selection.of([])), _Fixed(Selection.of(["v1"], ready=gate))])
    assert _select(chained).ready is gate


class _Thing(SensorView):
    """One sensor, for a link that declares it."""

    thing = Sensed()


class _SensesThing(_Fixed):
    """A link whose header says which view it reads."""

    sensors = (_Thing,)


def test_a_combinator_hands_each_link_the_view_that_link_declared():
    """A combinator declares nothing, so it narrows on each link's own header instead.

    Which is what keeps a chain from being the one place a declaration is not a fact --
    both of ``dedup_sim``'s links sit inside one. A link declaring nothing is handed
    what the combinator was given, which is what lets a ranking be wired to a stand-in
    that is no view at all.
    """
    view = View(None, {}).derived(_Thing, thing="sensed")
    senses, plain = _SensesThing(Selection.of([])), _Fixed(Selection.of(["v1"]))
    FirstMatch([senses, plain]).attach(view)
    assert senses.view.thing == "sensed"       # composed out of the one it was given
    assert senses.view is not view
    assert plain.view is view

    under = _SensesThing(Selection.priced([("v0", 1)]))
    Balance(under).attach(view)
    assert under.view is senses.view           # one view per distinct declaration


class _ThingAndLoad(_Thing, LoadView):
    """A view carrying both reads: the ranking's sensor and the combinator's load."""


def test_a_combinator_declares_its_base_s_reads_as_well_as_its_own():
    """Otherwise the narrowing loses the base's sensor, and the base senses nothing.

    A combinator senses load, so it narrows -- and a view is composed of exactly what
    was declared, so a declaration naming only load would hand the ranking under it a
    view without the ranking's own sensor, which raises rather than answering quietly.
    This is the shape it happens in: a balanced ranking as a link of a chain, where what
    each link is attached to is its own header.
    """
    view = View(None, {}).derived(_ThingAndLoad, thing="sensed", load=_Load())
    under = _SensesThing(Selection.priced([("v0", 1)]))
    balanced = Balance(under)
    assert declares((LoadView,), under) == (LoadView, _Thing) == balanced.sensors

    FirstMatch([balanced]).attach(view)
    assert under.view.thing == "sensed"        # the read its header declared
    assert balanced.view.load.named() == {}    # and the one the combinator declared
    assert _select(balanced).sources == ("v0",)


def test_first_match_attaches_every_selector_even_unconsulted_ones():
    """A link behind an earlier answer still has to be brought up.

    That is what lets it gate: a selector parks requesters on facts only its own
    bookkeeping will settle, and it never gets the chance to sense them if the chain
    only attaches what it consults.
    """
    front, behind = _Fixed(Selection.of(["v0"])), _Fixed(Selection.of(["v1"]))
    chained = FirstMatch([front, behind])
    assert chained.attach("a-view") is chained     # wiring is one expression
    _select(chained)                      # behind is never asked ...
    assert front.view is behind.view == "a-view"   # ... but is brought up


def test_attaching_a_selector_hands_it_back_however_it_is_wrapped():
    """So a selector can be built and wired in one expression.

    Every override returns too, or the shape would work for a bare ranking and not
    for the two combinators -- which is where a run does most of its wiring.
    """
    ranking = _Fixed(Selection.priced([("v0", 1)]))
    assert ranking.attach("a-view") is ranking
    balanced = Balance(ranking)
    assert balanced.attach("a-view") is balanced


def test_a_chain_takes_its_links_subject_and_refuses_links_that_disagree():
    """The subject a chain hands down is the subject every link must take.

    ``select`` passes it through untouched, so a link reading it as something else would
    answer a question it was not asked. Checked at construction, which is what lets the
    chain take the subject as its own -- and so be a link of a chain itself.
    """
    chain = FirstMatch([_Fixed(Selection.of([])), _Fixed(Selection.of(["v1"]))])
    assert _select(chain).sources == ("v1",)
    assert chain.subject_type == Sequence[Key]
    assert FirstMatch([chain]).subject_type == Sequence[Key]
    assert FirstMatch([]).subject_type is Any        # nothing to take it from

    with pytest.raises(TypeError, match="must select over the same one"):
        FirstMatch([_Fixed(), _FixedPlacement()])


def test_a_stage_measures_off_the_view_and_the_subject_alone():
    """The two things a stage may read, and the whole of what ``Annotate`` hands it.

    Which is what keeps a chain declarable: a stage needing a second ranking's answer
    would need a join, and a join is the plane's -- it does one and hands the result down
    as part of the subject.
    """
    seen = []

    def readings(view, subject):
        seen.append((view, tuple(subject)))
        return lambda source: len(source) + view.load.named().get(source, 0)

    senses = _Senses(_Load(v0=3))
    staged = Annotate(
        _Fixed(Selection.of(["v0", "v11"])), readings, senses=(LoadView,)
    )
    assert staged.sensors == (LoadView,)         # declared, so the view carries the read
    staged.attach(senses)
    assert _select(staged).key == {"v0": (5,), "v11": (3,)}
    assert seen == [(senses, ("K",))]            # once per answer, not once per source


def test_balance_appends_a_dimension_and_computes_nothing():
    """What it adds is a reading, behind what the ranking under it said.

    So the fold sees both numbers and decides between them -- and the base's own
    dimension arrives as the base wrote it, which is what lets a caller price the winner
    against something of its own without ever reading a weighed figure.
    """
    base = _FixedPlacement(Selection.keyed([("h0", (5,)), ("h1", (9,))]))
    balanced = Balance(base).attach(_Senses(_Load(h0=2)))
    annotated = _select(balanced)
    assert annotated.key == {"h0": (5, 2), "h1": (9, 0)}
    assert annotated.sources == ("h0", "h1")          # ordered nothing


def test_balance_moves_the_tie_between_equally_priced_sources():
    """The point of the combinator: two sources a ranking cannot separate.

    The base answers the same way every time, so the alternation is the load's and
    nothing else's -- and it goes on alternating rather than reverting to id order once
    both sources carry a grant, because the load dimension is still there to compare
    when the price has been levelled.
    """
    balanced = Balance(_Fixed(Selection.priced([("v0", 5), ("v1", 5)])))
    assert balanced.subject_type == KeySelector.subject_type   # a chain can hold it
    assert _heads(balanced, count=4) == ["v0", "v1", "v0", "v1"]


def test_balance_reads_a_load_it_does_not_keep():
    """Asking is a read: the ranking writes nothing, so a load nothing moves is inert.

    Which is what makes this a combinator over an observation rather than a tally of
    its own -- what moves the number is the decision the answer is consulted for, and
    a plane that prices ten candidates and decides once moves it once.
    """
    equal = Selection.priced([("v0", 5), ("v1", 5)])
    assert _heads(Balance(_Fixed(equal)), count=4, moving=False) == ["v0"] * 4

    # ...and the winner follows the load, whoever moved it.
    senses = _Senses(_Load(v0=1))
    best = Max(Balance(_Fixed(equal))).attach(senses)
    assert _select(best).head == "v1"
    senses.load.sent("v1")
    assert _select(best).head == "v0"


def test_the_fold_a_chain_is_stamped_with_decides_what_load_may_outvote():
    """The bound is the application's, in its own units, and so is its absence.

    Compared as they stand, the price is the first dimension and a source ahead on it
    wins however loaded it is; folded against a bound wide enough to cover the gap, load
    does trade the price away -- and only once the bound has been fully spent. Both live
    at the fold, which is why the combinator has no knob of its own to get the units of
    wrong.
    """
    apart = Balance(_Fixed(Selection.priced([("v0", 5), ("v1", 9)])))
    assert _heads(apart, count=4) == ["v0"] * 4
    assert _heads(Folded(apart, _bounded(1)), count=4) == ["v0"] * 4
    assert _heads(Folded(apart, _bounded(4)), count=5) == [
        "v0", "v0", "v0", "v0", "v1",
    ]


def test_balance_passes_an_answer_with_no_source_to_rank_straight_through():
    """The two empties again: neither names a source, so neither is annotated.

    Returned as they were built, and nothing is read off the load either: there is no
    source to measure.
    """
    for empty in (Selection.of([]), Selection()):
        balanced = Balance(_Fixed(empty))
        balanced.attach(_Senses())
        assert _select(balanced) is empty
        assert _select(balanced) is empty


def test_balance_over_a_ranking_that_keys_nothing_is_ordered_by_load_alone():
    """A pool and nothing more, ranked by the one dimension appended behind it.

    Least-loaded routing, and the only shape it can take: a stage appends behind what
    the stages before it left, so behind a ranking that named a pool and keyed none of
    it, load is the whole of the order.
    """
    ranked = Sort(Balance(_Fixed(Selection.of(["v0", "v1"]))))
    ranked.attach(_Senses(_Load(v0=2, v1=1)))
    assert _select(ranked).sources == ("v1", "v0")


def test_balance_takes_the_subject_of_the_ranking_it_wraps():
    """One combinator over both subjects, and a chain still refuses a mismatch.

    A combinator asks whatever its ranking asks -- it hands the subject down untouched
    -- so its own subject cannot be declared once; it is read off the ranking. Which is
    what makes a balanced ranking a chain link exactly where the ranking under it would
    be, with no second class per combinator.
    """
    over_keys = Balance(_Fixed(Selection.priced([("v0", 5)])))
    over_payload = Balance(_FixedPlacement(Selection.priced([("v0", 5)])))

    assert over_keys.subject_type == KeySelector.subject_type
    assert over_payload.subject_type is Any          # read off the ranking under it

    FirstMatch([over_keys, _Fixed()])                # a link like the ranking it wraps
    with pytest.raises(TypeError, match="must select over the same one"):
        FirstMatch([over_keys, over_payload])


def test_balance_spreads_an_application_ranking_too():
    """The tie this breaks is not the store's question in particular.

    Two hosts an application priced the same sort on its last tie-break, exactly as two
    volumes do, so the combinator is the same one -- which is the whole point of it
    taking any subject.
    """
    ranking = _FixedPlacement(Selection.priced([("h0", 5), ("h1", 5)]))
    assert _heads(Balance(ranking), count=4) == ["h0", "h1", "h0", "h1"]


def test_balance_attaches_the_ranking_under_it_and_keeps_what_it_answered():
    """The wrapped selector is sensing, and the answer is annotated and nothing else.

    The gate rides through and every source keeps the *base's* dimension, so a caller
    pricing the winner against something of its own never reads a weighed figure.
    """

    async def gate() -> None:
        return None

    base = _Fixed(Selection.priced([("v0", 5), ("v1", 5)], ready=gate))
    balanced = Balance(base)
    senses = _Senses()
    best = Max(balanced).attach(senses)
    ranked = Sort(balanced).attach(senses)

    assert base.view is senses                          # brought up by its holder
    senses.load.sent(_select(best).head)                # v0 won, and decided on ...
    second = _select(ranked)                            # ... so v1 leads now
    assert second.sources == ("v1", "v0")
    assert second.ready is gate
    assert second.head == "v1"
    assert second.key == {"v0": (5, 1), "v1": (5, 0)}   # the base's 5, then the load


# --------------------------------------------------------------------------
# 3c. Narrowing: a ranking is narrowed by the caller holding it, not by an object.
# --------------------------------------------------------------------------


def test_require_keeps_a_ranking_whose_head_passes_and_abstains_otherwise():
    ranked = Selection.of(["v0", "v1"])
    assert ranked.require(lambda head: head == "v0") is ranked
    assert ranked.require(lambda head: head == "v1").sources == ()


def test_require_drops_the_whole_ranking_not_just_the_head():
    """Filtering would promote ``v1``, which the ranking preferred *less*.

    The ranking need not be in the order the test measures, so reaching past a head it
    rejected would overrule the ranking from outside it.
    """
    assert Selection.of(["r", "v1"]).require(lambda head: head != "r").sources == ()


def test_require_leaves_an_abstention_alone_and_refuses_the_naive_answer():
    """The two empties again, and they behave as oppositely here as in a chain.

    ``Selection.of([])`` names nobody, so there is no head to judge and nothing to do.
    ``Selection()`` is the directory's whole answer: narrowing it would quietly return
    an unnarrowed ranking, so it is a wiring error and says so.
    """
    abstained = Selection.of([])
    assert abstained.require(lambda head: False) is abstained

    with pytest.raises(ValueError, match="no head"):
        Selection().require(lambda head: True)


def test_a_narrowed_selection_keeps_its_gate_and_the_keys_it_kept():
    """What narrowing drops is sources, not what the selector said about them."""

    async def gate() -> None:
        return None

    narrowed = Selection.priced([("v0", 7), ("v1", 9)], ready=gate).take(1)

    assert narrowed.sources == ("v0",)
    assert narrowed.ready is gate
    assert narrowed.key == {"v0": (7,)}
    assert "v1" not in narrowed.key      # dropped with the source it priced


def test_head_is_the_leading_source_and_what_was_measured_is_under_it():
    """Both empties read as ``None``: neither names a source to act on."""
    ranked = Selection.priced([("v0", 7), ("v1", 9)])
    assert (ranked.head, ranked.key[ranked.head]) == ("v0", (7,))
    assert Selection.of([]).head is None
    assert Selection().head is None


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
    and the two here do not execute alike -- dedup reads through, kvcache prefills
    and decodes.

    ``routes`` is the one thing beside it, and it is a declaration rather than a
    verb: empty until a plane says a member of its own answers with an address.
    """
    verbs = {
        name for name in vars(DataPlane)
        if not name.startswith("_") and callable(getattr(DataPlane, name))
    }
    assert verbs == {"attach"}
    assert DataPlane.routes == {}
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


# --------------------------------------------------------------------------
# 6. Reroutes: the answer names the host, and a caller goes there.
# --------------------------------------------------------------------------


class _Rerouted(DataPlane):
    """``ServingHost``'s shape with the work taken out.

    One member and one signature: it works out where the subject belongs, and either
    answers with that address or serves it. Whoever the address names is called with
    exactly what this one was.
    """

    def __init__(self, me: str, *, names: Optional[str] = None) -> None:
        self.me = me
        self.names = names
        self.seen: list[str] = []

    @routed(at=lambda answered: answered.elsewhere)
    async def serve(self, subject: str):
        self.seen.append(subject)
        if self.names is not None:
            return _Answer(elsewhere=self.names)
        return _Answer(served=f"{subject}@{self.me}")


class _Answer:
    """What ``_Rerouted.serve`` answers: an address, or the subject served."""

    def __init__(self, *, elsewhere=None, served=None) -> None:
        self.elsewhere = elsewhere
        self.served = served


class _Circling(DataPlane):
    """A plane that always names the other host: an answer that never is one."""

    def __init__(self, me: str, other: str) -> None:
        self.me = me
        self.other = other

    @routed(at=lambda answered: answered.elsewhere)
    async def serve(self, subject: str):
        return _Answer(elsewhere=self.other)


class _Refuses(DataPlane):
    """A plane that answers nothing at all."""

    @routed(at=lambda answered: answered.elsewhere)
    async def serve(self, subject: str) -> None:
        return None


def _endpointish(method):
    """Stand in for Monarch's ``@endpoint``: a descriptor holding the method.

    Not the real one -- what matters is the shape a wrapper has (an object whose only
    reference to the method is private), which is why ``routed`` may not be one and
    has to record its declaration where a class can still find it.
    """

    class _Property:
        def __init__(self, method) -> None:
            self._method = method

        def __get__(self, instance, owner):
            return self

    return _Property(method)


class _EitherOrder(DataPlane):
    """The declaration composes with a wrapper above it and below it."""

    @_endpointish
    @routed(at=lambda answered: None)
    async def over(self, subject: str) -> None:
        ...

    @routed(at=lambda answered: None)
    @_endpointish
    async def under(self, subject: str) -> None:
        ...


def _fronted(**planes) -> Simulation:
    """An assembled stack with one data plane per node, fronted for callers."""
    sim = Simulation(_topology())
    for node, plane in sorted(planes.items()):
        sim.front_plane(node, plane)
    return sim


def test_a_declaration_leaves_the_member_alone_and_lands_on_the_class():
    """``routed`` records; it does not wrap.

    The member is the same object it was, so a service still finds a coroutine where
    it looks for one and nothing follows an address near the host. Where the
    declaration lands is the class -- and it survives a wrapper on either side of it,
    because the class attribute is the wrapper by the time anything reads it.
    """
    async def member(self, subject: str) -> None:
        ...

    assert routed(at=lambda answered: None)(member) is member
    assert list(_Rerouted.routes) == ["serve"]
    assert set(_EitherOrder.routes) == {"over", "under"}
    # What the declaration is: where the answer says to go.
    assert _Rerouted.routes["serve"](_Answer(elsewhere="b")) == "b"


def test_a_call_is_sent_wherever_its_answer_says():
    """Two hosts, one member, and the caller names only the first.

    The address is in the answer the first host produced, so the reroute is the
    server's decision -- and the call the second host gets is the one the first got,
    unchanged, so nothing about a member's signature says whether it was rerouted into.
    """
    a, b = _Rerouted("a", names="b"), _Rerouted("b")
    sim = _fronted(a=a, b=b)
    answered = _drive(sim, RoutedPlane(sim, _Rerouted).serve("x", at="a"))
    assert answered.served == "x@b"
    assert a.seen == ["x"] and b.seen == ["x"]


def test_an_answer_that_names_nobody_is_the_result():
    """A member that never yields a host is one call, and so is a refusal."""
    lone = _Rerouted("a")                          # names nobody, so it serves
    sim = _fronted(a=lone)
    assert _drive(sim, RoutedPlane(sim, _Rerouted).serve("x", at="a")).served == "x@a"
    assert lone.seen == ["x"]

    sim = _fronted(a=_Refuses())
    assert _drive(sim, RoutedPlane(sim, _Refuses).serve("x", at="a")) is None


def test_a_cycle_hits_the_hop_cap():
    """A plane that keeps naming a host cannot hang the caller.

    Nothing is wrong with being sent on once or twice -- that is the point -- so the
    only thing that can tell a long chain from an endless one is a cap on the hops.
    """
    sim = _fronted(a=_Circling("a", "b"), b=_Circling("b", "a"))
    with pytest.raises(RuntimeError, match="still being sent on"):
        _drive(sim, RoutedPlane(sim, _Circling, max_hops=4).serve("x", at="a"))


def test_a_host_that_holds_its_peers_is_refused():
    """Engines xor a peer table, checked where a host becomes reachable.

    A plane that answers with an address *and* can call it forwards instead of
    answering, which is what answering with one exists instead of -- and the
    structure lint cannot see it, because a peer is not a control port.
    """
    DataPlaneService(_Rerouted("a", names="b"))     # holds nobody: fronted happily

    holds_one = _Rerouted("a", names="b")
    holds_one.peer = _Rerouted("b")
    with pytest.raises(TypeError, match="holds another host"):
        DataPlaneService(holds_one)

    holds_a_table = _Rerouted("a", names="b")
    holds_a_table.peers = {"b": _Rerouted("b")}
    with pytest.raises(TypeError, match="holds another host"):
        DataPlaneService(holds_a_table)


def test_every_hop_pays_the_client_hop():
    """Each call is charged out and back, and free by default.

    A caller is off the box, so being sent on is a boundary crossing like reaching the
    directory. Two hops at ``rtt`` cost four one-way trips; at the default the hop is
    inline and the whole thing advances the clock not at all.
    """
    from sim_common import config

    rtt = 0.25
    with config.overrides(client_rtt=rtt):
        sim = _fronted(a=_Rerouted("a", names="b"), b=_Rerouted("b"))
        walked = _drive(
            sim, _timed(RoutedPlane(sim, _Rerouted).serve("x", at="a"))
        )
    assert walked == 4 * rtt

    sim = _fronted(a=_Rerouted("a", names="b"), b=_Rerouted("b"))
    assert _drive(sim, _timed(RoutedPlane(sim, _Rerouted).serve("x", at="a"))) == 0.0


async def _timed(awaitable) -> float:
    """How much virtual time ``awaitable`` took."""
    started = asyncio.get_running_loop().time()
    await awaitable
    return asyncio.get_running_loop().time() - started
