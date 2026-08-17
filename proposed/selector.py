"""One question, two subjects: which sources serve this, and when::

    Selector[Subject, *Dims].select(subject, requester) -> Selection[*Dims]

The answer is always volume ids, what orders them, and the moment they become usable.
**Annotating** appends a dimension to the sort key and leaves the order alone, so a
ranking costs one fold rather than one sort per dimension, and that fold can be greedy
over every dimension at once where a re-sort could only take them one at a time. Ordering
is a link like any other (:func:`Ordered`, :func:`Best`), and the only kind that touches
:attr:`Selection.sources`.

What differs is the **subject** and how many dimensions order the answer, and every
selector names both in its header::

    class RoutedPull(KeySelector[()]):          # keys, and nothing orders the answer
    class Priced(Selector[PrefillAsk, Plan]):   # own subject, keyed at what it costs

A selector is a **utility**, not a plane. Nothing outside a capability reaches one: a run
knows about the capability's :class:`~proposed.plane.ControlPlane`, that plane declares the
questions its callers may ask, and a selector is one of the things it may work the answer
out with. So a selector needs no lifecycle beyond the view it ranks against
(:meth:`Selector.attach`), and a ranking that never leaves the plane that built it may hold
whatever it likes, gate and all. What crosses a service boundary is the plane's business
(:meth:`Selection.settled`).

A header carries what a selector takes and what orders what it answers with, both worth
saying where a reader already looks: a :class:`Selection` is ids and their dimensions, and a
dimension is whatever appended it measured (:data:`Ks`). One subject has a name, because the
store asks it: :class:`KeySelector` -- ``Sequence[Key]``, which volume serves these bytes.
``dedup_sim`` wants a reader routed to a *peer* about to hold the key, ``kvcache_sim``
the peer holding the longest reusable prefix. An application's own subject is
``Selector[ThatSubject]`` and needs no name here.

What a check compares is :attr:`Selector.subject_type` and not the class, since :pep:`484`
erases the parameter and a combinator's subject is the one it was handed rather than one it
declares.

Admission and SLO gates are neither: an answer that is not a ranked set of sources does not
belong in a :class:`Selection` at all. A gate rides *with* one -- a selector that refuses
abstains (:meth:`Selection.abstain`), and what it would have answered is not in the ranking.

A decision is **declared**: a chain, built where the selector is wired, in two halves.
The **base** makes a :class:`Selection` out of a subject and settles what is measured --
:class:`Const` over a pool the caller already knows, or a capability's own ranking, with
whatever annotates it wrapped around it there (``Balance(ranking)``). A **stage** then
interprets what the base measured without moving it: :class:`WithFold` says how the key is
read, :func:`Ordered` and :func:`Best` order or cut. So a chain is one arity end to end and
a list of stages (:data:`Stage`, :func:`pipe`). :class:`FirstMatch` picks between whole
alternatives -- ask each in order, take the first answer -- and checks its links agree on
one subject at construction, since a chain hands *one* subject to every link. Every
combinator hands the subject down untouched and takes it off what it holds, which is why
the same :data:`Balance` annotates a ranking over keys and one over an application's own
candidates alike.

Annotating measures from the view and the subject alone: it appends behind whatever is
already there, reads no key and names no source, so a fold still reads what each earlier one
measured, and over a ranking that keyed nothing one reading is the whole of the order. Two
rankings combined into one answer is a **plane's** job: it joins them and hands the result
down as part of the subject, as a **value**, measured once per decision where holding a
*selector* would re-select once per candidate.

Narrowing an answer is not a composition of selectors here: a test an application owns is
applied to the ranking it was given, by whoever has both (:meth:`Selection.require`,
:meth:`Selection.take`). A capability writing a combinator of its own needs only
:func:`declares` and :func:`declared` from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import (
    Any, Awaitable, Callable, Dict, Generic, Mapping, Optional, Protocol, Sequence,
    Tuple, TypeVar, TypeVarTuple, Unpack, get_args, get_origin,
)

from proposed.deployment import Key, VolumeId
from proposed.view import LoadView, View

__all__ = [
    "Ready", "Ks", "Fold", "Readings", "Selection", "prefer", "DecisionLog",
    "declared",
    "declares", "Selector", "KeySelector", "NaiveKeySelector", "Stage", "pipe",
    "FirstMatch", "Const", "Annotate", "Balance", "Lift", "WithFold", "Ordered", "Best",
]

# A readiness gate: called with no arguments, awaited until the chosen source is
# usable. ``None`` means "usable now".
Ready = Callable[[], Awaitable[None]]

#: What a selector's ``select`` takes: the **subject**, and the one parameter a header
#: binds, which :meth:`Selector.__init_subclass__` reads back into ``subject_type``.
_S = TypeVar("_S")

#: The selector :meth:`Selector.attach` hands back: whatever it was called on, so a
#: wired chain is still a chain and a wired combinator still a combinator.
_Sel = TypeVar("_Sel", bound="Selector[Any, Unpack[Tuple[Any, ...]]]")

#: What orders one source (:attr:`Selection.key`), as a type parameter: one reading per
#: stage that measured it, in the order they annotated, **lower is better** throughout.
#: ``Selection[Plan, float]`` is what two stages keyed and a :data:`Fold` over it takes
#: ``Tuple[Plan, float]``, so one reading a different number of them is refused where the
#: chain is declared. A stage may append what it *holds* rather than a number (a plan, off
#: which a fold reads the figure it orders by); nothing can compare such a dimension, so a
#: fold that named no figure raises rather than ordering by something meaningless.
#:
#: ``Unpack[Tuple[Any, ...]]`` is an arity nothing static knows (:class:`FirstMatch`), and
#: the one case where reading past the end is still an :exc:`IndexError` per decision.
Ks = TypeVarTuple("Ks")

#: One reading a stage appends (:meth:`Selection.annotated`): the last of its arity.
_R = TypeVar("_R")

#: :meth:`Selection.priced`'s one dimension, bounded because that reading *is* the order.
_C = TypeVar("_C", bound="_Comparable")

class _Comparable(Protocol):
    """What a fold must answer with: something an ordering link can put in an order.

    ``__lt__`` alone, since that is all :func:`sorted` and :func:`min` ask for, and a tuple
    of comparables is one too -- how the id is appended as the last dimension
    (:func:`_comparable`).
    """

    def __lt__(self, other: Any) -> bool:
        ...


#: How a caller blends one source's dimensions into the single comparable a fold orders
#: by: ``dims -> comparable``, lower still better, read by position (:data:`Ks`). ``None``
#: is the lexicographic default, which needs no arithmetic at all (:func:`Ordered`).
Fold = Callable[[Tuple[Unpack[Ks]]], _Comparable]

#: What one stage appends (:meth:`Selection.annotated`): the measure of one source, called
#: once per source, written at the type it measures. A mapping already in hand is passed as
#: its ``__getitem__``.
Readings = Callable[[VolumeId], _R]


def _named_once(sources: Tuple[VolumeId, ...]) -> Tuple[VolumeId, ...]:
    """``sources`` unchanged, or a :exc:`ValueError` naming whichever it repeats.

    A source named twice is keyed once, so which position a caller reads back is undefined.
    Only the producers take a sequence from outside; everything else permutes or cuts one.
    """
    if len(set(sources)) != len(sources):
        raise ValueError(
            f"a selection names each source once; repeated: "
            f"{sorted(s for s in set(sources) if sources.count(s) > 1)}"
        )
    return sources


@dataclass(frozen=True)
class Selection(Generic[Unpack[Ks]]):
    """Sources for one subject, what orders them, and when they become usable.

    Annotating does not order: it appends to :attr:`key` and leaves :attr:`sources` as
    built. Only :func:`Ordered` and :func:`Best` order one, and no flag says either has --
    a selection's order is whatever its producer left.

    Args:
        sources: volume ids. ``None`` -- the default -- means *every holder, in
            directory order*, which is what the real directory returns on its own, so
            a ``None`` selection leaves the store's answer untouched (:func:`prefer`);
            :meth:`universe` names it. ``()`` names nobody and decides nothing, which is
            the opposite answer (:meth:`abstain`, :attr:`abstains`, :meth:`otherwise`).
        key: ``source id -> the dimensions that order it``, one per stage that measured
            it, and the arity this is written at (:data:`Ks`). What a stage *holds* about
            a source rides here as a dimension too -- a plan, a score -- so a ranking
            cannot come apart from what produced it, and a caller reads it back by
            position (``key[head][0]``). ``None`` for a producer with nothing to say
            about the order; otherwise it covers exactly :attr:`sources` (:meth:`only`).
        ready: optional gate, for a selector that routes a requester to a peer which
            has not registered yet. Spent by :meth:`settled` before the answer
            travels, never handed to whoever asked.
        fold: how to read :attr:`key`, stamped by the link that knows both the dimensions
            there are and what they mean together (:class:`WithFold`) -- so nothing that
            orders this names a fold, and two callers of one ranking cannot fold it two
            different ways. ``None`` compares the dimensions as they stand. A closure, so
            :meth:`settled` drops it as it drops the gate.
    """

    sources: Optional[Tuple[VolumeId, ...]] = None
    key: Optional[Mapping[VolumeId, Tuple[Unpack[Ks]]]] = None
    ready: Optional[Ready] = None
    fold: Optional[Fold[Unpack[Ks]]] = None

    @classmethod
    def of(
        cls,
        sources: Sequence[VolumeId],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[Unpack[Tuple[()]]]":
        """``sources`` as they are given, keyed by nothing.

        The three builders are the three kinds of producer: this one for a selector that
        only names sources, :meth:`priced` for one whose own measure is the whole of the
        order, :meth:`keyed` for one giving the dimensions itself.
        """
        return Selection(sources=_named_once(tuple(sources)), ready=ready)

    @classmethod
    def keyed(
        cls,
        candidates: Sequence[Tuple[VolumeId, Tuple[Unpack[Ks]]]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[Unpack[Ks]]":
        """``(id, dims)`` pairs: what orders each source, from one sequence.

        The sequence's own order is not a ranking -- :func:`Ordered` and :func:`Best` read
        the dimensions.
        """
        return Selection(
            sources=_named_once(tuple(i for i, _d in candidates)),
            key=dict(candidates),
            ready=ready,
        )

    @classmethod
    def priced(
        cls,
        candidates: Sequence[Tuple[VolumeId, _C]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_C]":
        """``(id, price)`` pairs, that price standing as the one dimension: cheapest is
        best."""
        return Selection.keyed([(i, (p,)) for i, p in candidates], ready=ready)

    @classmethod
    def universe(cls) -> "Selection[Unpack[Tuple[()]]]":
        """Every holder, in directory order: the store's own answer (:attr:`sources`)."""
        return Selection()

    @classmethod
    def abstain(cls) -> "Selection[Unpack[Ks]]":
        """Nobody, deciding nothing: the identity of :meth:`otherwise`.

        Any arity, naming no source to key, so a selector that keys two when it answers
        still abstains with this.
        """
        return Selection(sources=())

    def annotated(self, readings: "Readings[_R]") -> "Selection[Unpack[Ks], _R]":
        """This selection with one reading per source appended as a further dimension.

        Exactly one position, at the end, never over one already written: the only operation
        that moves the arity. :meth:`only`, :meth:`take` and :meth:`require` carry every kept
        source's dimensions through as they stand, so a fold written against one stage's
        position still reads it after a cut. One call per stage: the key mapping is rebuilt
        here, so a call per source would cost a walk per source. The one arity claim a
        checker cannot tie up is here -- the ``()`` a ranking that keyed nothing spreads back
        is not its ``Selection[()]``.
        """
        return replace(self, key={  # type: ignore[return-value]
            source: (*(self.key or {}).get(source, ()), readings(source))
            for source in (self.sources or ())
        })

    async def wait(self) -> None:
        """Block until the chosen sources are usable (returns at once if ready)."""
        if self.ready is not None:
            await self.ready()

    async def settled(self) -> "Selection[Unpack[Ks]]":
        """This selection with its gate spent: awaited, then dropped.

        What a plane reached as a service answers with. :attr:`ready` is a closure, so it
        cannot cross the boundary a handle stands for, where the ranking and its key can,
        being values. Awaiting it here is what makes the answer true when it arrives: a
        ranking released early names a volume holding nothing yet.
        """
        await self.wait()
        if self.ready is None and self.fold is None:
            return self
        return replace(self, ready=None, fold=None)

    @property
    def abstains(self) -> bool:
        """Whether this names nobody: the one empty a chain passes over."""
        return self.sources == ()

    def otherwise(self, other: "Selection[Unpack[Ks]]") -> "Selection[Unpack[Ks]]":
        """This selection if it decided anything, else ``other`` (:class:`FirstMatch`)."""
        return other if self.abstains else self

    @property
    def head(self) -> Optional[VolumeId]:
        """The leading source, or ``None`` if this names none in particular.

        The id a caller acts on, and the *best* source once this has been ordered
        (:func:`Ordered`, :func:`Best`); what the winning stage measured is
        ``key[head]``. ``None`` for both empties: neither names a source to act on.
        """
        if not self.sources:
            return None
        return self.sources[0]

    def only(self, sources: Sequence[VolumeId]) -> "Selection[Unpack[Ks]]":
        """This selection cut down to ``sources``, in the order given.

        The readiness gate rides along and each kept source keeps its key, so one selection
        narrowed by somebody else still answers for whoever built it. Keeping only what was
        named and keying all of it makes a key's coverage **structural**: no operation
        builds a selection naming a source it cannot order, so nothing checks per build.

        Raises:
            ValueError: on the default selection (every holder in directory order):
                cutting it answers with *nobody*, the opposite decision, and the refusal
                :meth:`require` makes too.
            ValueError: if ``sources`` names anything this selection did not -- nothing
                here knows what would order a source this selection never priced.
        """
        if self.sources is None:
            raise ValueError(
                "a selection naming every holder in directory order cannot be cut: "
                "narrowing the store's own answer to a subset of nothing would answer "
                "with nobody, which decides the opposite; abstain on purpose instead"
            )
        kept = tuple(sources)
        mine = set(self.sources)
        stray = [s for s in kept if s not in mine]
        if stray:
            raise ValueError(
                f"a cut may only keep sources this selection named; "
                f"{sorted(set(stray))} were not among them"
            )
        return Selection(
            sources=kept,
            key=None if self.key is None else {s: self.key[s] for s in kept},
            ready=self.ready,
            fold=self.fold,
        )

    def take(self, n: int) -> "Selection[Unpack[Ks]]":
        """The leading ``n`` sources, key and gate intact; refuses ⊤ as :meth:`only` does."""
        return self.only((self.sources or ())[:n])

    def require(self, ok: Callable[[VolumeId], bool]) -> "Selection[Unpack[Ks]]":
        """This selection if its head satisfies ``ok``, else the abstention.

        All or nothing: filtering the head out would **promote** the source behind it,
        which the ranking preferred *less*, overruling it on a measurement it is not in
        the order of. So judge an ordered ranking (:func:`Ordered`, :func:`Best`); an
        abstention comes back unchanged, having no head to judge.

        Raises:
            ValueError: on the default selection (every holder in directory order),
                which names no head to narrow -- as :meth:`only` refuses to cut it.
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


class Selector(ABC, Generic[_S, Unpack[Ks]]):
    """Rank the sources that should serve a subject, and say when they are usable.

    Written with the subject it takes and the arity it answers at
    (``Selector[Request, Plan, float]``), so ``select``'s signature is in the header a reader
    already looks at and a fold above is checked against those dimensions (:data:`Ks`). The
    subject stays argument 0, where :meth:`__init_subclass__` reads it.

    A utility a control plane consults, and deliberately **not** a
    :class:`~proposed.plane.ControlPlane`: a run never holds one, so it needs no
    sensor to harvest and no service in front of it. Subclass this, or
    :class:`KeySelector` where the subject is the store's own.
    """

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
            bound = getattr(get_origin(base) or base, "__parameters__", ())
            if _S not in bound:  # type: ignore[misc]  # a TypeVar object, not a type here
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
        (:meth:`proposed.plane.ControlPlane.attach`), passed straight down: one selector, one
        view, whoever asks -- never a per-call argument::

            source = Balance(LongestPrefixKeySelector()).attach(view)
        """
        self.view = view
        return self

    @abstractmethod
    def select(self, subject: _S, requester: str) -> Selection[Unpack[Ks]]:
        """Rank the sources that should serve ``subject`` for ``requester``.

        ``subject`` is whatever this selector was parameterized with, and
        :attr:`subject_type` is that type as a value a check can compare.

        Synchronous, so a whole chain is one turn: nothing can be decided between the
        readings a ranking prices against and the answer they produce. A ranking that
        must wait says so with a gate on the answer instead (:attr:`Selection.ready`),
        which is spent where the answer crosses a boundary (:meth:`Selection.settled`).
        """


class KeySelector(Selector[Sequence[Key], Unpack[Ks]]):
    """A selector whose subject is **keys**: which volume serves these bytes.

    The store's own question, what a control plane answering "which volumes serve this
    read" ranks with; the arity stays open, so a key ranking says what it keys
    (``KeySelector[int]``). An application's own subject is ``Selector[ThatSubject, ...]``
    and needs no name here.
    """


class NaiveKeySelector(KeySelector[Unpack[Tuple[()]]]):
    """Every holder, in directory order, usable now.

    Precisely the real directory's own answer, so this returns
    :meth:`Selection.universe` rather than re-deriving it: a read preferring what it
    names is byte-identical to a read that names nothing (:func:`prefer`).
    """

    def select(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[Unpack[Tuple[()]]]:
        return Selection.universe()


def declares(
    own: Sequence[type], base: "Selector[Any, Unpack[Tuple[Any, ...]]]"
) -> Tuple[type, ...]:
    """What a combinator senses: ``own``, plus whatever ``base`` does, each named once.

    A view is composed of exactly what a selector declared (:attr:`Selector.sensors`), so a
    combinator declaring only its own read would attach its base to a view missing the
    base's, and an undeclared read raises (:class:`~proposed.view.Sensed`). One that senses
    nothing declares nothing and hands the whole view down (:class:`FirstMatch`); ``()``
    from ``base`` is the whole view, and every view carries the directory, so a base that
    declared nothing loses nothing by being handed a narrower one.
    """
    return tuple(dict.fromkeys(tuple(own) + tuple(base.sensors)))


def declared(
    view: Any, selector: "Selector[Any, Unpack[Tuple[Any, ...]]]"
) -> Any:
    """The view ``selector`` declared, out of the one a combinator was handed.

    What a combinator narrows to for a selector it holds is that selector's own header,
    otherwise a chain would be the one place a declaration is not a fact -- and both of
    ``dedup_sim``'s links sit inside one. Something that is no view at all is handed on
    untouched, which only a selector declaring nothing can be attached to anyway.
    """
    return view.subset(*selector.sensors) if selector.sensors else view


#: One operation on a ranking that **interprets** what is already measured, leaving the
#: dimensions where they are: :class:`WithFold` stamps, :func:`Ordered` and :func:`Best`
#: order or cut. An operation with a parameter of its own takes that first, so it is
#: already a stage where a chain names it. The arity is preserved in the type, so a chain
#: of these is one arity end to end (:func:`pipe`).
#:
#: What is not one: a **base**, which makes a ranking rather than taking one
#: (:class:`Const`); :class:`FirstMatch`, which takes a list of alternatives; and
#: **annotating** (:class:`Annotate`, :data:`Balance`), which appends a dimension and so
#: goes on the base -- ``Balance(ranking)``, where what is measured is settled.
Stage = Callable[[Selector[_S, Unpack[Ks]]], Selector[_S, Unpack[Ks]]]


class FirstMatch(Selector[_S, Unpack[Tuple[Any, ...]]]):
    """Ask each selector in order; the first one that answers is the answer.

    A :class:`Selection` can be empty in two ways, and they mean opposite things:

    * :meth:`Selection.universe` -- ``sources is None`` -- is *every holder, in
      directory order*, the decision :class:`NaiveKeySelector` makes. It **wins the
      chain**, and the selectors behind it are never consulted.
    * :meth:`Selection.abstain` names nobody. That is the **abstention**, and it falls
      through (:attr:`Selection.abstains`).

    The chain is a fold of :meth:`Selection.otherwise`, seeded with the abstention: an
    associative operation with that as its identity, so an exhausted chain abstains in turn
    and a chain of chains answers as one would. One that should always answer ends with a
    :class:`NaiveKeySelector`. The winner comes back exactly as built, gate and all.

    The subject goes down every link untouched, so the links must agree on one
    :attr:`~Selector.subject_type`, checked at construction rather than trusted, and the
    chain takes it as its own. Compared as a value and not as a class, because a
    combinator's subject is the one it was handed rather than one it declares
    (:data:`Balance`).

    **Arity, unlike the subject, is not statically known here and the links need not
    agree on it.** Which one answers is a runtime fact, and a tail keying nothing behind a
    ranking keying plenty is what a tail is for. So a :class:`WithFold` over a chain is
    unchecked; both real chains end in :func:`Ordered` and name no fold.

    Args:
        selectors: consulted left to right, all over one subject -- a chain answers as
            its links do. An empty chain is legal and abstains.
    """

    def __init__(
        self, selectors: Sequence[Selector[_S, Unpack[Tuple[Any, ...]]]]
    ) -> None:
        self.selectors: Tuple[Selector[_S, Unpack[Tuple[Any, ...]]], ...] = tuple(
            selectors
        )
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

    def select(self, subject: _S, requester: str) -> Selection[Unpack[Tuple[Any, ...]]]:
        """The first non-abstaining answer, or an abstention if there is none.

        A link is asked only once every link before it has abstained, so the seed is what
        an exhausted chain answers with and no link behind a decision is consulted.
        """
        answer: Selection[Unpack[Tuple[Any, ...]]] = Selection.abstain()
        for selector in self.selectors:
            answer = selector.select(subject, requester).otherwise(answer)
            if not answer.abstains:
                break
        return answer


class Const(Selector[Any, Unpack[Ks]]):
    """One fixed :class:`Selection`, whatever the subject.

    The base of a chain over a pool the caller already knows -- the constant function, not
    the identity: the subject reaches the stages above this and never this. The arity is the
    held selection's.
    """

    def __init__(self, selection: Selection[Unpack[Ks]]) -> None:
        self.selection = selection

    def select(self, subject: Any, requester: str) -> Selection[Unpack[Ks]]:
        return self.selection


class _Link(Selector[_S, Unpack[Ks]]):
    """What every combinator over exactly one ranking shares.

    The subject is read off that ranking rather than declared, so a wrapped ranking is a
    chain link exactly where the ranking under it would be (:class:`FirstMatch`), and the
    ranking is wired to the view its own header declared (:func:`declared`), reachable only
    because ``senses`` is declared with the ranking's own reads (:func:`declares`).
    """

    def __init__(
        self, ranking: Selector[_S, Unpack[Ks]], senses: Sequence[type] = ()
    ) -> None:
        self.ranking = ranking
        self.subject_type = ranking.subject_type
        self.sensors = declares(senses, ranking)

    def attach(self, view: Any) -> "_Link[_S, Unpack[Ks]]":
        """Sense through ``view``, and hand the ranking the view it declared."""
        super().attach(view)
        self.ranking.attach(declared(view, self.ranking))
        return self


@dataclass(frozen=True)
class Annotate(Generic[_R]):
    """A further dimension appended to whatever ranking this is applied to.

    It holds the measure -- no ranking, view or subject -- so one may be shared by every
    chain wanting it (:data:`Balance`), and is written at what it measures: ``Annotate[int]``
    over a ``Selector[S, Plan]`` answers ``Selector[S, Plan, int]``.

    Args:
        readings: ``(view, subject) -> Readings[_R]`` -- the measure, taken once per
            answer (:meth:`Selection.annotated`). A callable because a reading does not
            exist until there is a subject and a view to take it through. Annotate it at
            its reading type, or the dimension erases and a fold above compares clean
            against anything.
        senses: the views ``readings`` reads, declared beside the ranking's.
    """

    readings: Callable[[Any, Any], Readings[_R]]
    senses: Tuple[type, ...] = ()

    def __call__(
        self, ranking: Selector[_S, Unpack[Ks]]
    ) -> "_Annotated[_S, Unpack[Ks], _R]":
        return _Annotated(ranking, self.readings, self.senses)


class _Annotated(_Link[_S, Unpack[Ks]]):
    """One ranking with a further dimension appended: what ``readings`` measured.

    What :class:`Annotate` builds. The ranking's own dimensions ride through untouched, so
    a fold reads what it measured beside this one; behind one that keyed nothing, this
    reading is the whole of the order.

    :data:`Ks` here is the arity this *answers* at, one longer than the ranking's -- no
    :class:`typing.TypeVarTuple` states that, so ``ranking`` is taken at unknown arity and
    :meth:`Annotate.__call__` declares the append.
    """

    def __init__(
        self,
        ranking: Selector[_S, Unpack[Tuple[Any, ...]]],
        readings: Callable[[Any, Any], Readings[Any]],
        senses: Sequence[type] = (),
    ) -> None:
        super().__init__(ranking, senses)
        self.readings = readings

    def select(
        self, subject: _S, requester: str
    ) -> Selection[Unpack[Tuple[Any, ...]]]:
        """``ranking``'s answer with one reading per source appended to its key.

        Every source is measured, not just whichever one leads, so a caller that folds
        this and rejects the winner has the rest measured too. Nothing is ordered and
        nothing is written. Both empties name no source to measure, so both go back
        untouched.
        """
        ranked = self.ranking.select(subject, requester)
        if not ranked.sources:
            return ranked
        return ranked.annotated(self.readings(self.view, subject))


def _load_at(view: Any, subject: Any) -> Readings[int]:
    """What each source has lately been sent (:class:`~proposed.view.LoadView`).

    Read once per answer, with nothing awaited between the read and the dimension it
    becomes, so no decision can land inside one answer. Absent is nothing sent.
    """
    load = view.load.named()
    return lambda source: load.get(source, 0)


#: :class:`Annotate` partially applied at the load view: ``Balance(ranking)`` is that
#: ranking annotated with how loaded each source it named is. A preset, not an operation of
#: its own, so there is one of it for the whole process.
#:
#: Load spreading as a layer over *any* ranking that says what orders it, rather than a
#: property of one ranking: two sources a ranking keys the same are left to the id the
#: fold breaks its ties on, so every read goes to the same volume. This puts something
#: that changes ahead of that tie-break. Behind a ranking that keyed nothing the load is
#: the whole of the order, which is how a bare pool is ranked by load alone.
#:
#: What it senses, and nothing else: :class:`~proposed.view.LoadView`, whose ``named()``
#: says what has lately been sent at each source. So this holds no tally of its own --
#: what it appends is an observation somebody else keeps, moved by the decision that
#: names a source -- and what that number means is stated once, on the view.
#:
#: **No arithmetic**, so there is nothing here for a caller to supply: whoever folds
#: decides what a busy source costs, which is the application's own trade -- blocks of
#: prefix run against reads routed at a host, seconds of link time against seconds of
#: queue (:class:`WithFold`).
Balance = Annotate(_load_at, senses=(LoadView,))


def _comparable(
    selection: Selection[Unpack[Ks]],
) -> Optional[Callable[[VolumeId], Any]]:
    """What orders one of ``selection``'s sources, or ``None`` if nothing says what best is.

    The fold it carries (:attr:`Selection.fold`) blends one source's dimensions into the
    comparable to order by; with none they are compared as they stand, lexicographic in
    the order they were annotated. Either way the id is the last thing compared, here and
    nowhere else, so no two sources compare equal: a run reproduces, and the least of a
    pool (:func:`_best`) is the front of a sort of it (:func:`_ordered`).
    """
    if not selection.sources or selection.key is None:
        return None
    key, fold = selection.key, selection.fold
    if fold is None:
        return lambda s: (*key[s], s)
    return lambda s: (fold(key[s]), s)


#: What a :class:`Lift` applies to the answer it was handed: one endomorphism of a
#: :class:`Selection`, total over both empties and over an answer no stage keyed. An
#: endomorphism, so the arity it is handed is the arity it answers at.
_Endo = Callable[[Selection[Unpack[Ks]]], Selection[Unpack[Ks]]]


def _stamp(fold: Optional[Fold[Unpack[Ks]]]) -> _Endo[Unpack[Ks]]:
    """Write ``fold`` onto an answer, ``None`` included (:class:`WithFold`)."""
    return lambda answer: replace(answer, fold=fold)


def _ordered(answer: Selection[Unpack[Ks]]) -> Selection[Unpack[Ks]]:
    """``answer`` best-first, or untouched if nothing says what best is: both empties, and
    an answer no stage keyed, leave the producer's own order standing.
    """
    order = _comparable(answer)
    if order is None:
        return answer
    return answer.only(sorted(answer.sources or (), key=order))


def _best(answer: Selection[Unpack[Ks]]) -> Selection[Unpack[Ks]]:
    """``answer`` cut to its single best source, that source's key and the gate intact.

    One pass, not a sort, and still the source :func:`_ordered` would leave in front since
    :func:`_comparable` admits no ties. Keyed by nothing, the leader stands.
    """
    if not answer.sources:
        return answer
    order = _comparable(answer)
    best = answer.sources[0] if order is None else min(answer.sources, key=order)
    return answer.only((best,))


class Lift(_Link[_S, Unpack[Ks]]):
    """One ranking's answer with ``endo`` applied to it.

    The shape :class:`WithFold`, :func:`Ordered` and :func:`Best` share. The endo is handed
    the whole answer, so the gate and the dimensions ride through whatever it does to the
    sources: a chain cut to one source can still be settled (:meth:`Selection.settled`).

    Args:
        ranking: the selector asked.
        endo: what to make of what it answered (:data:`_Endo`).
    """

    def __init__(
        self, ranking: Selector[_S, Unpack[Ks]], endo: _Endo[Unpack[Ks]]
    ) -> None:
        super().__init__(ranking)
        self.endo = endo

    def select(self, subject: _S, requester: str) -> Selection[Unpack[Ks]]:
        return self.endo(self.ranking.select(subject, requester))


@dataclass(frozen=True)
class WithFold(Generic[Unpack[Ks]]):
    """``fold`` stamped on whatever ranking this is applied to (:attr:`Selection.fold`).

    A :data:`Stage` once built, carrying the arity its fold reads, so one over a ranking of
    any other arity is refused where the chain is declared. Named where both halves of the
    key exist: the dimensions there are, and what they mean together. ``None`` infers no
    arity -- say which key it leaves alone, ``WithFold[float, int](None)``.

    Args:
        fold: how to read the key, or ``None`` to compare the dimensions as they stand.
    """

    fold: Optional[Fold[Unpack[Ks]]]

    def __call__(self, ranking: Selector[_S, Unpack[Ks]]) -> Lift[_S, Unpack[Ks]]:
        return Lift(ranking, _stamp(self.fold))


def Ordered(ranking: Selector[_S, Unpack[Ks]]) -> Lift[_S, Unpack[Ks]]:
    """One ranking's answer, ordered best-first (:func:`_ordered`)."""
    return Lift(ranking, _ordered)


def Best(ranking: Selector[_S, Unpack[Ks]]) -> Lift[_S, Unpack[Ks]]:
    """The single best source of one ranking's answer (:func:`_best`)."""
    return Lift(ranking, _best)


def pipe(
    base: Selector[_S, Unpack[Ks]],
    *stages: Stage[_S, Unpack[Ks]],
) -> Selector[_S, Unpack[Ks]]:
    """``base`` with each stage applied in turn, left to right:
    ``pipe(r, WithFold(f), Best) == Best(WithFold(f)(r))``.

    Every stage preserves the arity, so one name covers a chain of any length. Refused here:
    a fold reading other than what ``base`` keyed, and an annotating stage in the list --
    that goes on the base (:data:`Stage`).
    """
    for stage in stages:
        base = stage(base)
    return base
