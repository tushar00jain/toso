"""One question, two subjects: which sources serve this, and when::

    Selector[Subject].select(subject, requester) -> Selection

The answer is always volume ids, what orders them, and the moment they become usable. A
stage **annotates**: it appends a dimension to the sort key and leaves the order alone,
so a chain of them costs one fold rather than one sort per stage, and that fold can be
greedy over every dimension at once where a re-sort could only take them one at a time.
Ordering is a link like any other (:func:`Sort`, :func:`Max`), and the only kind that
touches :attr:`Selection.sources`.

What differs is the **subject**, and every selector names its own in its header::

    class RoutedPull(KeySelector):              # keys, the store's own question
    class Hosts(Selector[Request]):             # an application's own subject

A selector is a **utility**, not a plane. Nothing outside a capability reaches one:
a run knows about the capability's :class:`~proposed.plane.ControlPlane`, that plane
declares the questions its callers may ask, and a selector is one of the things it
may work the answer out with. So a selector needs no lifecycle beyond the view it
ranks against (:meth:`Selector.attach`), and a ranking that never leaves the plane
that built it may hold whatever it likes -- including a readiness gate. What crosses
a service boundary is the plane's business (:meth:`Selection.settled`).

The subject is the one thing a header carries, because what a selector takes is worth
saying where a reader already looks -- and what it answers with does not vary: a
:class:`Selection` is ids and their dimensions, and a dimension is whatever the stage
that appended it measured (:data:`Dims`). One subject has a name, because the store
asks it: :class:`KeySelector` -- ``Sequence[Key]``, which volume serves these bytes.
``dedup_sim`` wants a reader routed to a *peer* about to hold the key, ``kvcache_sim``
the peer holding the longest reusable prefix. An application's own subject is
``Selector[ThatSubject]`` and needs no name here.

What a check compares is :attr:`Selector.subject_type` and not the class, since
:pep:`484` erases the parameter and a combinator's subject is the one it was handed
rather than one it declares.

Admission and SLO gates are neither: an answer that is not a ranked set of sources
does not belong in a :class:`Selection` at all. A gate rides *with* one instead --
a selector that refuses abstains (:meth:`Selection.abstain`), and what it would have
answered is simply not in the ranking.

A decision is **declared**: a chain, built where the selector is wired, whose links each
fill one of four roles -- a **base** makes a :class:`Selection` out of a subject
(:class:`Const` over a pool the caller already knows, or a capability's own ranking); a
**stage** appends a dimension (:class:`Annotate`, :class:`Balance`); :func:`Folded` says
how the key is read; :func:`Sort` and :func:`Max` order or cut. :class:`FirstMatch`
picks between whole alternatives -- ask each in order, take the first answer -- and checks
its links agree on one subject at construction, since a chain hands *one* subject to
every link. Every combinator hands the subject down untouched and takes it off what it
holds, which is why one :class:`Balance` annotates a ranking over keys and one over an
application's own candidates alike.

A stage measures from the view and the subject alone: it appends behind whatever the
stages before it left, reads no key and names no source, so a fold still reads what each
earlier one measured, and behind a ranking that keyed nothing one reading is the whole of
the order. Two rankings combined into one answer is a **plane's** job, not a chain's: it
does the join and hands the result down as part of the subject. A stage takes that earlier
answer as a **value**, so it is measured once per decision; one holding a *selector* would
re-select once per candidate.

Narrowing an answer is not a composition of selectors in this package: a test an
application owns is applied to the ranking it was given, by whoever has both
(:meth:`Selection.require`, :meth:`Selection.take`). A capability that writes a
combinator of its own -- one narrowing an answer so that a chain reaches the link
behind it, say -- needs only :func:`declares` and :func:`declared` from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import (
    Any, Awaitable, Callable, Dict, Generic, List, Mapping, NamedTuple, Optional,
    Protocol, Sequence, Tuple, TypeVar, Union, get_args, get_origin,
)

from proposed.deployment import Key, VolumeId
from proposed.view import LoadView, View

__all__ = [
    "Ready", "Dims", "Fold", "Readings", "Selection", "prefer", "DecisionLog",
    "declared",
    "declares", "Selector", "KeySelector", "NaiveKeySelector",
    "FirstMatch", "Const", "Annotate", "Balance", "Lift", "Folded", "Sort", "Max",
]

# A readiness gate: called with no arguments, awaited until the chosen source is
# usable. ``None`` means "usable now".
Ready = Callable[[], Awaitable[None]]

#: What a selector's ``select`` takes: the **subject**, and the one parameter a header
#: binds, which :meth:`Selector.__init_subclass__` reads back into ``subject_type``.
_S = TypeVar("_S")

#: The selector :meth:`Selector.attach` hands back: whatever it was called on, so a
#: wired chain is still a chain and a wired combinator still a combinator.
_Sel = TypeVar("_Sel", bound="Selector[Any]")

#: What orders one source (:attr:`Selection.key`): one reading per stage that measured
#: it, **positional** -- each stage appends its own behind the ones already there
#: (:meth:`Selection.annotated`) -- and **lower is better** throughout. A stage may
#: append what it *holds* rather than a number (a plan, off which a fold reads the one
#: figure it orders by); such a dimension is safe because nothing can compare it -- a
#: fold that named no figure raises rather than ordering by something meaningless.
Dims = Tuple[Any, ...]

#: How a caller blends one source's dimensions into the single comparable a fold
#: orders by: ``dims -> comparable``, lower still better. Positional, so a fold reads
#: ``dims[1]`` and a chain missing that stage raises instead of quietly comparing the
#: wrong number. ``None`` is the lexicographic default, which needs no arithmetic at all
#: (:func:`Sort`).
Fold = Callable[[Dims], Any]

#: What one stage appends (:meth:`Selection.annotated`): a reading per source, either
#: measured already or measurable on demand.
Readings = Union[Mapping[VolumeId, Any], Callable[[VolumeId], Any]]


@dataclass(frozen=True)
class Selection:
    """Sources for one subject, what orders them, and when they become usable.

    A stage annotates and does not order: it appends to :attr:`key` and leaves
    :attr:`sources` however it built them. Only :func:`Sort` and :func:`Max` order one,
    and there is no flag saying that either has: a selection's order is whatever its
    producer left.

    Args:
        sources: volume ids. ``None`` -- the default -- means *every holder, in
            directory order*, which is what the real directory returns on its own, so
            a ``None`` selection leaves the store's answer untouched (:func:`prefer`);
            :meth:`universe` names it. ``()`` names nobody and decides nothing
            (:meth:`abstain`, :attr:`abstains`). The two empties are opposites where a
            chain chooses between answers (:meth:`otherwise`).
        key: ``source id -> the dimensions that order it`` (:data:`Dims`), one per
            stage that measured it. What a stage *holds* about a source rides here as a
            dimension too -- a plan, a score -- so a ranking cannot come apart from what
            produced it, and a caller reads it back by position
            (``key[head][0]``). ``None`` for a producer with nothing to say about the
            order.
        ready: optional gate, for a selector that routes a requester to a peer which
            has not registered yet. Spent by :meth:`settled` before the answer
            travels, never handed to whoever asked.
        fold: how to read :attr:`key`, stamped by the link that knows both the dimensions
            there are and what they mean together (:func:`Folded`) -- so nothing that
            orders this names a fold, and two callers of one ranking cannot fold it two
            different ways. ``None`` compares the dimensions as they stand. A closure, so
            :meth:`settled` drops it as it drops the gate.
    """

    sources: Optional[Tuple[VolumeId, ...]] = None
    key: Optional[Mapping[VolumeId, Dims]] = None
    ready: Optional[Ready] = None
    fold: Optional[Fold] = None

    @classmethod
    def of(
        cls,
        sources: Sequence[VolumeId],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection":
        """``sources`` as they are given, keyed by nothing.

        The three builders are the three kinds of producer: this one for a selector
        that only names sources, :meth:`priced` for one whose own measure is the whole
        of the order, :meth:`keyed` for one that gives the dimensions itself -- an
        order that negates what it measured, or more dimensions than one.
        """
        return cls(sources=tuple(sources), ready=ready)

    @classmethod
    def keyed(
        cls,
        candidates: Sequence[Tuple[VolumeId, Dims]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection":
        """``(id, dims)`` pairs: what orders each source, from one sequence.

        The sequence's own order is not a ranking -- :func:`Sort` and :func:`Max` read
        the dimensions.
        """
        return cls(
            sources=tuple(i for i, _d in candidates),
            key={i: tuple(dims) for i, dims in candidates},
            ready=ready,
        )

    @classmethod
    def priced(
        cls,
        candidates: Sequence[Tuple[VolumeId, Any]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection":
        """``(id, price)`` pairs, that price standing as the one dimension: cheapest is
        best."""
        return cls.keyed([(i, (p,)) for i, p in candidates], ready=ready)

    @classmethod
    def universe(cls) -> "Selection":
        """Every holder, in directory order: the store's own answer (:attr:`sources`)."""
        return cls()

    @classmethod
    def abstain(cls) -> "Selection":
        """Nobody, deciding nothing: the identity of :meth:`otherwise`."""
        return cls.of([])

    def annotated(self, readings: "Readings") -> "Selection":
        """This selection with one reading per source appended as a further dimension.

        What a stage that measures rather than ranks does (:class:`Annotate`). Appended
        behind what the stages before it left, never in place of it, so every fold
        reads a position that does not move.

        ``readings`` is a mapping covering every source, or a callable measuring one --
        which is the form for a stage whose readings do not exist yet, since building a
        mapping to hand over here would walk the sources an extra time.

        One call per stage: the whole key mapping is rebuilt here, so a call per source
        would cost a walk per source.
        """
        read = readings if callable(readings) else readings.__getitem__
        return replace(self, key={
            source: (*(self.key or {}).get(source, ()), read(source))
            for source in (self.sources or ())
        })

    async def wait(self) -> None:
        """Block until the chosen sources are usable (returns at once if ready)."""
        if self.ready is not None:
            await self.ready()

    async def settled(self) -> "Selection":
        """This selection with its gate spent: awaited, then dropped.

        What a plane reached as a service answers with. :attr:`ready` is a closure,
        so it cannot cross the boundary a handle stands for -- the ranking and its
        key can, being values. Awaiting it here is also what makes the answer
        true when it arrives: the caller is about to read from these sources, and
        a ranking released early names a volume holding nothing yet.
        """
        await self.wait()
        if self.ready is None and self.fold is None:
            return self
        return replace(self, ready=None, fold=None)

    @property
    def abstains(self) -> bool:
        """Whether this names nobody: the one empty a chain passes over."""
        return self.sources == ()

    def otherwise(self, other: "Selection") -> "Selection":
        """This selection if it decided anything, else ``other`` (:class:`FirstMatch`)."""
        return other if self.abstains else self

    @property
    def head(self) -> Optional[VolumeId]:
        """The leading source, or ``None`` if this names none in particular.

        The id a caller acts on, and the *best* source once this has been ordered
        (:func:`Sort`, :func:`Max`); what the winning stage measured is
        ``key[head]``. ``None`` for both empties, which a caller reading the head cannot
        tell apart and does not need to: neither one names a source to act on.
        """
        if not self.sources:
            return None
        return self.sources[0]

    def only(self, sources: Sequence[VolumeId]) -> "Selection":
        """This selection cut down to ``sources``, in the order given.

        The readiness gate rides along and each kept source keeps its key, so one
        selection narrowed by somebody else still answers for whoever built it.
        """
        kept = tuple(sources)
        return Selection(
            sources=kept,
            key=None if self.key is None else {
                s: self.key[s] for s in kept if s in self.key
            },
            ready=self.ready,
            fold=self.fold,
        )

    def take(self, n: int) -> "Selection":
        """The leading ``n`` sources, key and gate intact."""
        return self.only((self.sources or ())[:n])

    def require(self, ok: Callable[[VolumeId], bool]) -> "Selection":
        """This selection if its head satisfies ``ok``, else the abstention.

        The narrowing a caller does *to* a ranking, as a method on the ranking, so a
        test an application owns needs no object to live in and composes by being
        called again.

        All or nothing: filtering the head out would **promote** the source behind it,
        and a ranking need not be in the order ``ok`` measures -- the sources behind the
        head are the ones the ranking preferred *less*, so promoting one on a raw
        measurement would overrule it from outside. Which is why a ranking is ordered
        first (:func:`Sort`, :func:`Max`): unordered, the head this judges is
        the producer's build order and nothing more. An abstention is returned
        unchanged, since there is no head to judge.

        Raises:
            ValueError: on the default selection (every holder in directory order),
                which names no head and so cannot be narrowed.
        """
        if self.sources is None:
            raise ValueError(
                "a selection naming every holder in directory order has no head to "
                "require anything of: narrow one that ranks"
            )
        if not self.sources or ok(self.sources[0]):
            return self
        return Selection.abstain()


def prefer(
    located: Dict[str, Dict[str, Any]],
    sources: Optional[Sequence[VolumeId]],
) -> Dict[str, Dict[str, Any]]:
    """A directory answer reordered to put ``sources`` first, best first.

    What the store does with a preference its caller handed it: ``locate_volumes``
    reads the directory and then applies this, so a client that takes the first
    volume listed per key reads from the source the caller named. It consults
    nothing -- ``sources`` is a value, typically :attr:`Selection.sources` from a
    control plane the caller asked itself.

    ``None`` -- no preference -- is the directory's own answer, returned unchanged.
    A key none of ``sources`` holds is also left untouched: this is a preference,
    not a filter that can make data disappear.
    """
    if sources is None:
        return located
    scoped: Dict[str, Dict[str, Any]] = {}
    for key, volume_map in located.items():
        ranked = {vid: volume_map[vid] for vid in sources if vid in volume_map}
        scoped[key] = ranked if ranked else volume_map
    return scoped


class DecisionLog(Protocol):
    """Somewhere a selector can explain itself.

    Optional and never load-bearing: a selector must behave identically with none
    attached. Declared here so a selector need not name the simulator's trace.
    """

    def record(self, at: float, kind: str, message: str) -> None:
        ...


class Selector(ABC, Generic[_S]):
    """Rank the sources that should serve a subject, and say when they are usable.

    Written with the subject it takes (``Selector[Request]``), so ``select``'s
    signature is in the header a reader already looks at; what it answers with is a
    :class:`Selection` whatever the subject.

    A utility a control plane consults, and deliberately **not** a
    :class:`~proposed.plane.ControlPlane`: a run never holds one, so it needs no
    sensor to harvest and no service in front of it. Subclass this, or
    :class:`KeySelector` where the subject is the store's own.
    """

    name = "selector"

    #: What :meth:`select` takes, as a value: :pep:`484` erases the parameter, so
    #: ``isinstance(x, Selector[Sequence[Key]])`` raises and the one place a
    #: subject is *checked* needs something comparable. Set by
    #: :meth:`__init_subclass__` from the parameter, so the annotation and the value
    #: cannot disagree.
    subject_type: Any = Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Read :attr:`subject_type` off whichever base bound :data:`_S`.

        A subject is the only thing a header binds, so a base that binds one binds it
        as its single argument; a class no base of which binds it inherits its parent's
        subject, which is how a bare subclass (``class RoutedPull(KeySelector)``) and
        one that only narrows behaviour both resolve. One that declares ``subject_type``
        itself is left alone, since a computed subject would be overwritten here.

        Reads ``cls.__dict__`` rather than :func:`getattr`, which would find the
        base's ``__orig_bases__`` and give every unparameterized subclass the bare
        type variable.
        """
        super().__init_subclass__(**kwargs)
        if "subject_type" in vars(cls):
            return
        for base in cls.__dict__.get("__orig_bases__", ()):
            if _S not in getattr(get_origin(base) or base, "__parameters__", ()):
                continue
            subject = get_args(base)[0]
            if not isinstance(subject, TypeVar):
                cls.subject_type = subject
            return

    #: What this selector senses through: ``None`` until :meth:`attach`, and never
    #: read by one that ranks only what it is handed.
    view: Optional[View] = None

    #: The views this selector reads, as :class:`~proposed.view.View` subclasses. What
    #: it is attached to composes exactly these (:meth:`~proposed.view.View.subset`),
    #: so the header says what a ranking senses and an undeclared read raises instead
    #: of quietly working. ``()`` -- the default -- is the whole view, which is also
    #: what a ranking sensing nothing is handed.
    sensors: Tuple[type, ...] = ()

    def attach(self: _Sel, view: Any) -> _Sel:
        """Keep the view this selector senses and prices through, and return it.

        The one the plane holding it was handed
        (:meth:`proposed.plane.ControlPlane.attach`), passed straight down: a
        selector runs beside the directory it senses, so the view is the run's and
        not a per-call argument -- one selector, one view, whoever asks.

        Returned so building one and wiring it is a single expression::

            source = Balance(LongestPrefixKeySelector()).attach(view)
        """
        self.view = view
        return self

    @abstractmethod
    def select(self, subject: _S, requester: str) -> Selection:
        """Rank the sources that should serve ``subject`` for ``requester``.

        ``subject`` is whatever this selector was parameterized with, and
        :attr:`subject_type` is that type as a value a check can compare.

        Synchronous, so a whole chain is one turn: nothing can be decided between the
        readings a ranking prices against and the answer they produce. A ranking that
        must wait says so with a gate on the answer instead (:attr:`Selection.ready`),
        which is spent where the answer crosses a boundary (:meth:`Selection.settled`).
        """


class KeySelector(Selector[Sequence[Key]]):
    """A selector whose subject is **keys**: which volume serves these bytes.

    The store's own question, and what a control plane answering "which volumes
    should serve this read" ranks with. A ranking over an application's own subject
    is ``Selector[ThatSubject]`` and needs no name of its own -- this package cannot
    name a type an application invented.
    """


class NaiveKeySelector(KeySelector):
    """Every holder, in directory order, usable now.

    Precisely the real directory's own answer, so this returns
    :meth:`Selection.universe` rather than re-deriving it: a read preferring what it
    names is byte-identical to a read that names nothing (:func:`prefer`).
    """

    name = "naive"

    def select(self, keys: Sequence[Key], requester: str) -> Selection:
        return Selection.universe()


def declares(
    own: Sequence[type], base: "Selector[Any]"
) -> Tuple[type, ...]:
    """What a combinator senses: ``own``, plus whatever ``base`` does, each named once.

    A view is composed of exactly what a selector declared
    (:attr:`Selector.sensors`), so a combinator declaring only its own read would
    attach its base to a view missing the base's -- and an absent sensor raises rather
    than answering quietly (:class:`~proposed.view.Sensed`). A combinator that senses
    nothing itself declares nothing and hands the whole view down
    (:class:`FirstMatch`); one that senses something has to say both.

    ``()`` from ``base`` is the whole view, and every view carries the directory, so a
    base that declared nothing loses nothing by being handed a narrower one.
    """
    return tuple(dict.fromkeys(tuple(own) + tuple(base.sensors)))


def declared(view: Any, selector: "Selector[Any]") -> Any:
    """The view ``selector`` declared, out of the one a combinator was handed.

    What a combinator narrows to for a selector it holds is that selector's own header
    -- otherwise a chain would be the one place a declaration is not a fact, and both of
    ``dedup_sim``'s links sit inside one. Which is only reachable because the combinator
    declared the link's reads along with its own (:func:`declares`). Something that is
    no view at all is handed on untouched, which only a selector declaring nothing can
    be attached to anyway (:attr:`Selector.sensors`).
    """
    return view.subset(*selector.sensors) if selector.sensors else view


class FirstMatch(Selector[_S]):
    """Ask each selector in order; the first one that answers is the answer.

    A :class:`Selection` can be empty in two ways, and they mean opposite things:

    * :meth:`Selection.universe` -- ``sources is None`` -- is *every holder, in
      directory order*, the decision :class:`NaiveKeySelector` makes. It **wins the
      chain**, and the selectors behind it are never consulted.
    * :meth:`Selection.abstain` names nobody. That is the **abstention**, and it falls
      through (:attr:`Selection.abstains`).

    The chain is a fold of :meth:`Selection.otherwise` over the links' answers, seeded
    with the abstention, and three laws of that operation are what a chain rests on: the
    abstention is its identity, so an exhausted chain abstains in turn; it is
    associative, so ``FirstMatch([FirstMatch([a, b]), c])`` still reaches ``c``, as it
    could not if the inner chain's exhaustion arrived looking like a decision; and the
    universe absorbs whatever would be chosen after it. A chain that should always answer
    ends with a :class:`NaiveKeySelector`. The winner is returned exactly as built, so a
    readiness gate rides along untouched.

    The subject goes down every link untouched, so a link reading it as anything else
    would answer a question it was not asked: the links must agree on one
    :attr:`~Selector.subject_type`, checked at construction rather than trusted, and
    the chain takes it as its own. Compared as a value and not as a class, because a
    combinator's subject is the one it was handed rather than one it declares
    (:class:`Balance`).

    Args:
        selectors: consulted left to right, all over one subject -- a chain answers as
            its links do. An empty chain is legal and abstains.
    """

    name = "first-match"

    def __init__(self, selectors: Sequence[Selector[_S]]) -> None:
        self.selectors: Tuple[Selector[_S], ...] = tuple(selectors)
        subjects = {s.subject_type for s in self.selectors}
        if len(subjects) > 1:
            raise TypeError(
                f"a chain hands one subject to every link, so all of them must select "
                f"over the same one; these are "
                f"{', '.join(sorted(str(s) for s in subjects))}"
            )
        #: The links' own, so a chain is a link of a chain (:func:`declared`).
        self.subject_type = subjects.pop() if subjects else Any

    def attach(self, view: Any) -> "FirstMatch[_S]":
        """Hand every wrapped selector the view it declared, answering or not.

        One that senses through a view of its own must be brought up even if it
        never answers, so a link behind an earlier answer is still sensing when its
        turn comes.
        """
        for selector in self.selectors:
            selector.attach(declared(view, selector))
        return self

    def select(self, subject: _S, requester: str) -> Selection:
        """The first non-abstaining answer, or an abstention if there is none.

        A link is asked only once every link before it has abstained, so the seed is what
        an exhausted chain answers with and no link behind a decision is consulted.
        """
        answer = Selection.abstain()
        for selector in self.selectors:
            answer = selector.select(subject, requester).otherwise(answer)
            if not answer.abstains:
                break
        return answer


class Const(Selector[Any]):
    """One fixed :class:`Selection`, whatever the subject.

    The base of a chain over a pool the caller already knows -- the constant function,
    not the identity: the subject reaches the stages above this and never this.
    """

    name = "const"

    def __init__(self, selection: Selection) -> None:
        self.selection = selection

    def select(self, subject: Any, requester: str) -> Selection:
        return self.selection


class _Link(Selector[_S]):
    """What every combinator over exactly one ranking shares.

    The subject is read off that ranking rather than declared, so a wrapped ranking is a
    chain link exactly where the ranking under it would be (:class:`FirstMatch`), and the
    ranking is wired to the view its own header declared (:func:`declared`) -- reachable
    only because ``senses`` is declared together with the ranking's own reads
    (:func:`declares`).
    """

    def __init__(self, ranking: Selector[_S], senses: Sequence[type] = ()) -> None:
        self.ranking = ranking
        self.subject_type = ranking.subject_type
        self.sensors = declares(senses, ranking)

    def attach(self: _Sel, view: Any) -> _Sel:
        """Sense through ``view``, and hand the ranking the view it declared."""
        super().attach(view)
        self.ranking.attach(declared(view, self.ranking))
        return self


class Annotate(_Link[_S]):
    """One ranking with a further dimension appended: what ``readings`` measured.

    Args:
        ranking: the selector asked. Its own dimensions ride through untouched, so a fold
            reads what it measured beside this one; behind one that keyed nothing, this
            reading is the whole of the order.
        readings: ``(view, subject) -> Readings`` -- the measure, taken once per answer
            (:meth:`Selection.annotated`). A callable because a reading does not exist
            until there is a subject to take it of and a view to take it through.
        senses: the views ``readings`` reads, declared beside the ranking's.
    """

    name = "annotate"

    def __init__(
        self,
        ranking: Selector[_S],
        readings: Callable[[Any, _S], Readings],
        senses: Sequence[type] = (),
    ) -> None:
        super().__init__(ranking, senses)
        self.readings = readings

    def select(self, subject: _S, requester: str) -> Selection:
        """``ranking``'s answer with one reading per source appended to its key.

        Every source is measured, not just whichever one leads, so a caller that folds
        this and rejects the winner has the rest measured too. Nothing is ordered and
        nothing is written.

        An answer with no source to measure goes back untouched. Both empties qualify,
        for the same reason and not by accident: an abstention names nobody, and the
        default selection names every holder in directory order rather than any source in
        particular.
        """
        ranked = self.ranking.select(subject, requester)
        if not ranked.sources:
            return ranked
        return ranked.annotated(self.readings(self.view, subject))


def _load_at(view: Any, subject: Any) -> Readings:
    """What each source has lately been sent (:class:`~proposed.view.LoadView`).

    Read once per answer, with nothing awaited between the read and the dimension it
    becomes, so no decision can land inside one answer. Absent is nothing sent.
    """
    load = view.load.named()
    return lambda source: load.get(source, 0)


class Balance(Annotate[_S]):
    """One ranking, annotated with how loaded each source it named is.

    Load spreading as a layer over *any* ranking that says what orders it, rather than
    a property of one ranking: two sources a ranking keys the same are left to the id
    the fold breaks its ties on, so every read goes to the same volume. This puts
    something that changes ahead of that tie-break.

    What it senses, and nothing else: :class:`~proposed.view.LoadView`, whose ``named()``
    says what has lately been sent at each source. So this combinator holds no tally of
    its own -- what it appends is an observation somebody else keeps, moved by the
    decision that names a source -- and what that number means is stated once, on the
    view.

    **No arithmetic**, so there is nothing here for a caller to supply: whoever folds
    decides what a busy source costs, which is the application's own trade -- blocks of
    prefix run against reads routed at a host, seconds of link time against seconds of
    queue (:func:`Folded`).

    Args:
        ranking: the selector asked. Behind one that keyed nothing, the load is the whole
            of the order -- which is how a bare pool is ranked by load alone.
    """

    name = "balance"
    sensors = (LoadView,)

    def __init__(self, ranking: Selector[_S]) -> None:
        super().__init__(ranking, _load_at, senses=(LoadView,))


def _comparable(selection: Selection) -> Optional[Callable[[VolumeId], Any]]:
    """What orders one of ``selection``'s sources, or ``None`` if nothing says what best is.

    The fold it carries (:attr:`Selection.fold`) blends one source's dimensions into the
    comparable to order by; with none they are compared as they stand, which is
    lexicographic over the stages in the order they annotated. Either way the id is the
    last thing compared, here and nowhere else, so no two sources compare equal: a run
    reproduces, and the least of a pool (:func:`_best`) is the front of a sort of it
    (:func:`_ordered`).
    """
    if not selection.sources or selection.key is None:
        return None
    key, fold = selection.key, selection.fold
    if fold is None:
        return lambda s: (*key[s], s)
    return lambda s: (fold(key[s]), s)


#: What a :class:`Lift` applies to the answer it was handed: one endomorphism of a
#: :class:`Selection`, total over both empties and over an answer no stage keyed.
_Endo = Callable[[Selection], Selection]


def _stamp(fold: Optional[Fold]) -> _Endo:
    """Write ``fold`` onto an answer, ``None`` included (:func:`Folded`)."""
    return lambda answer: replace(answer, fold=fold)


def _ordered(answer: Selection) -> Selection:
    """``answer`` best-first, or untouched if nothing says what best is.

    Both empties, and an answer no stage keyed: the producer's own order stands.
    """
    order = _comparable(answer)
    if order is None:
        return answer
    return answer.only(sorted(answer.sources or (), key=order))


def _best(answer: Selection) -> Selection:
    """``answer`` cut to its single best source, that source's key and the gate intact.

    One pass, not a sort of the pool: still the source :func:`_ordered` would leave in
    front, since :func:`_comparable` admits no ties. Keyed by nothing, the leader stands.
    """
    if not answer.sources:
        return answer
    order = _comparable(answer)
    best = answer.sources[0] if order is None else min(answer.sources, key=order)
    return answer.only((best,))


class Lift(_Link[_S]):
    """One ranking's answer with ``endo`` applied to it.

    The shape :func:`Folded`, :func:`Sort` and :func:`Max` share. The endo is handed the
    whole answer, so a readiness gate and the dimensions ride through whatever it does
    with the sources, and what comes back is a selection however far it cut -- a chain
    that named one source can still be settled (:meth:`Selection.settled`).

    Args:
        ranking: the selector asked.
        endo: what to make of what it answered (:data:`_Endo`).
    """

    name = "lift"

    def __init__(self, ranking: Selector[_S], endo: _Endo) -> None:
        super().__init__(ranking)
        self.endo = endo

    def select(self, subject: _S, requester: str) -> Selection:
        return self.endo(self.ranking.select(subject, requester))


def Folded(ranking: Selector[_S], fold: Optional[Fold]) -> Lift[_S]:
    """One ranking's answer with ``fold`` stamped on it (:attr:`Selection.fold`)."""
    return Lift(ranking, _stamp(fold))


def Sort(ranking: Selector[_S]) -> Lift[_S]:
    """One ranking's answer, ordered best-first (:func:`_ordered`)."""
    return Lift(ranking, _ordered)


def Max(ranking: Selector[_S]) -> Lift[_S]:
    """The single best source of one ranking's answer (:func:`_best`)."""
    return Lift(ranking, _best)
