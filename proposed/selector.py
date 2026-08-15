"""One question, two subjects: which sources serve this, and when::

    Selector[Subject, Price].select(subject, requester) -> Selection[Price]

The answer is always volume ids, what orders them, and the moment they become usable
(and, for a selector handed candidates it did not price, what each one priced at). A
selector **annotates**: it appends a dimension to the sort key and leaves the order
alone, so a chain of them costs one fold rather than one sort per stage, and that fold
can be greedy over every dimension at once where a re-sort could only take them one at
a time. Whoever answers folds -- :meth:`Selection.sort`, :meth:`Selection.max` -- and
those two are the only things here that establish an order.

What differs is the **subject**, and every selector names its own in its header::

    class RoutedPull(KeySelector[None]):        # keys, ranked without pricing
    class Hosts(AnySelector[Request, Plan]):    # an application's subject, priced

A selector is a **utility**, not a plane. Nothing outside a capability reaches one:
a run knows about the capability's :class:`~proposed.plane.ControlPlane`, that plane
declares the questions its callers may ask, and a selector is one of the things it
may work the answer out with. So a selector needs no lifecycle beyond the view it
ranks against (:meth:`Selector.attach`), and a ranking that never leaves the plane
that built it may hold whatever it likes -- including a readiness gate. What crosses
a service boundary is the plane's business (:meth:`Selection.settled`).

The two named subjects are types as well, because what a selector takes is worth
saying in the header a reader already looks at:

* :class:`KeySelector` -- ``Sequence[Key]``: which volume serves these bytes, the
  store's own question. ``dedup_sim`` wants a reader routed to a *peer* about to
  hold the key, ``kvcache_sim`` the peer holding the longest reusable prefix.
* :class:`AnySelector` -- an application's own subject, whatever it is: which peer
  to source a prefix from, which host prefills, which host decodes. Its
  ``subject_type`` stays ``Any`` and a subclass narrows it, since this package
  cannot name a type an application invented.

Admission and SLO gates are neither: an answer that is not a ranked set of sources
does not belong in a :class:`Selection` at all. A gate rides *with* one instead --
a selector that refuses abstains (``Selection.of([])``), and what it would have
answered is simply not in the ranking.

Selectors compose two ways, both of them one selector holding others:
:class:`FirstMatch` picks between alternatives -- ask each in order, take the first
answer. :class:`Balance` annotates one answer -- ask, then append how loaded each source
it named is as a further dimension. Both hand the subject down untouched, which is why
one ``Balance`` annotates a ranking over keys and one over an application's own
candidates alike, taking the kind of whichever it wraps. ``FirstMatch`` is over keys
only, and checks its links are at construction: a chain hands *one* subject to every
link.

Annotating is why a ranking says what orders it rather than leaving that to a sort of
its own: a stage appended behind one that said nothing would be the whole of the order,
so :class:`Balance` refuses such a ranking. The *price* rides through untouched, so a
caller pricing the winner against something of its own still reads what the base said
about it.

Narrowing an answer is not a composition of selectors in this package: a test an
application owns is applied to the ranking it was given, by whoever has both
(:meth:`Selection.require`, :meth:`Selection.take`). A capability that writes a
combinator of its own -- one narrowing an answer so that a chain reaches the link
behind it, say -- needs only :func:`declares` and :func:`declared` from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    Any, Awaitable, Callable, Dict, Generic, List, Mapping, NamedTuple, Optional,
    Protocol, Sequence, Tuple, TypeVar, get_args, get_origin,
)

from proposed.deployment import Key, VolumeId
from proposed.view import LoadView, View

__all__ = [
    "Ready", "Dims", "Fold", "Selection", "prefer", "DecisionLog", "declared",
    "declares", "Selector", "KeySelector", "AnySelector", "NaiveKeySelector",
    "FirstMatch", "Balance",
]

# A readiness gate: called with no arguments, awaited until the chosen source is
# usable. ``None`` means "usable now".
Ready = Callable[[], Awaitable[None]]

#: What a selector holds about each source it ranks -- a price, a plan, a batch
#: size. The one thing about an answer that varies, so it is the only thing
#: :class:`Selection` is generic in, and a selector's second parameter: the
#: *subject* belongs to the selector alone, the price to both. ``None`` for one that
#: ranks without pricing.
_P = TypeVar("_P")

#: What a selector's ``select`` takes: the **subject** parameter, which
#: :meth:`Selector.__init_subclass__` reads back into ``subject_type`` by finding
#: whichever argument lines up with this variable.
_S = TypeVar("_S")

#: The selector :meth:`Selector.attach` hands back: whatever it was called on, so a
#: wired chain is still a chain and a wired combinator still a combinator.
_Sel = TypeVar("_Sel", bound="Selector[Any, Any]")

#: What orders one source (:attr:`Selection.key`): one comparable per stage that
#: measured it, **positional** -- each stage appends its own behind the ones already
#: there (:meth:`Selection.annotated`) -- and **lower is better** throughout.
Dims = Tuple[Any, ...]

#: How a caller blends one source's dimensions into the single comparable a fold
#: orders by: ``dims -> comparable``, lower still better. Positional, so a fold reads
#: ``dims[1]`` and a chain missing that stage raises instead of quietly comparing the
#: wrong number. ``None`` at a fold is the lexicographic default, which needs no
#: arithmetic at all (:meth:`Selection.sort`).
Fold = Callable[[Dims], Any]


@dataclass(frozen=True)
class Selection(Generic[_P]):
    """Sources for one subject, what orders them, and when they become usable.

    A stage annotates and does not order: it appends to :attr:`key` and leaves
    :attr:`sources` however it built them. Whoever answers folds -- :meth:`sort` or
    :meth:`max` -- and there is no flag saying which of the two has happened: a
    selection's order is whatever its producer left, so a caller that needs one asks
    for it.

    Args:
        sources: volume ids. ``None`` -- the default -- means *every holder, in
            directory order*, which is what the real directory returns on its own, so
            a ``None`` selection leaves the store's answer untouched (:func:`prefer`).
        key: ``source id -> the dimensions that order it`` (:data:`Dims`), one per
            stage that measured it. ``None`` for a producer with nothing to say about
            the order.
        payload: ``source id -> what this selector holds about that source``,
            application-defined because this package cannot read an application's
            values. A ranking alone loses what produced it, and a selector handed
            alternatives somebody else priced has to give the winner's price back with
            it. Not the same thing as the key wherever a stage ranks something it was
            handed -- a plan ordered by the TTFT predicted for it -- and the two are
            built from one sequence (:meth:`keyed`) so they cannot come apart.
        ready: optional gate, for a selector that routes a requester to a peer which
            has not registered yet. Spent by :meth:`settled` before the answer
            travels, never handed to whoever asked.
    """

    sources: Optional[Tuple[VolumeId, ...]] = None
    key: Optional[Mapping[VolumeId, Dims]] = None
    payload: Mapping[VolumeId, _P] = field(default_factory=dict)
    ready: Optional[Ready] = None

    @classmethod
    def of(
        cls,
        sources: Sequence[VolumeId],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_P]":
        """``sources`` as they are given, keyed by nothing and priced at nothing.

        The three builders are the three kinds of producer: this one for a selector
        that only names sources, :meth:`priced` for one whose price is also the one
        dimension that orders them, :meth:`keyed` for one where the two differ.
        """
        return cls(sources=tuple(sources), ready=ready)

    @classmethod
    def keyed(
        cls,
        candidates: Sequence[Tuple[VolumeId, Dims, _P]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_P]":
        """``(id, dims, payload)`` triples: what orders each source and what is held
        about it, from one sequence.

        The sequence's own order is not a ranking -- :meth:`sort` and :meth:`max` read
        the dimensions. Neither mapping is reachable on its own, so a key and the
        payload beside it cannot be built out of step.
        """
        return cls(
            sources=tuple(i for i, _d, _p in candidates),
            key={i: tuple(dims) for i, dims, _p in candidates},
            payload={i: p for i, _d, p in candidates},
            ready=ready,
        )

    @classmethod
    def priced(
        cls,
        candidates: Sequence[Tuple[VolumeId, _P]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_P]":
        """``(id, price)`` pairs, the price standing as the one dimension too:
        cheapest is best."""
        return cls.keyed([(i, (p,), p) for i, p in candidates], ready=ready)

    def annotated(self, readings: Mapping[VolumeId, Any]) -> "Selection[_P]":
        """This selection with one reading per source appended as a further dimension.

        What a stage that measures rather than ranks does (:class:`Balance`). Appended
        behind what the stages before it left, never in place of it, so every fold
        reads a position that does not move.
        """
        return replace(self, key={
            source: (*(self.key or {}).get(source, ()), readings[source])
            for source in (self.sources or ())
        })

    def _ranked(self, fold: Optional[Fold]) -> Optional[Tuple[VolumeId, ...]]:
        """These sources best-first, or ``None`` if nothing here says what best is.

        ``fold`` blends one source's dimensions into the comparable to order by; with
        none they are compared as they stand, which is lexicographic over the stages in
        the order they annotated. Either way the id is the last thing compared, here
        and nowhere else, so the order is total whatever the stages keyed on and a run
        reproduces.
        """
        if not self.sources or self.key is None:
            return None
        if fold is None:
            return tuple(sorted(self.sources, key=lambda s: (*self.key[s], s)))
        return tuple(sorted(self.sources, key=lambda s: (fold(self.key[s]), s)))

    def sort(self, fold: Optional[Fold] = None) -> "Selection[_P]":
        """This selection ordered best-first -- a **new** one, this being frozen.

        Both empties pass through, as does a selection no stage keyed: the order a
        producer left is the answer when there is nothing to beat it.
        """
        ranked = self._ranked(fold)
        return self if ranked is None else self.only(ranked)

    def max(self, fold: Optional[Fold] = None) -> "Selection[_P]":
        """The single best source, with its key, its price and the gate.

        What a plane naming one source folds to, and it is still a selection so
        :meth:`settled` can be spent on it afterwards. Both empties pass through.
        """
        if not self.sources:
            return self
        ranked = self._ranked(fold)
        return self.only((ranked or self.sources)[:1])

    @property
    def winner(self) -> Optional[_P]:
        """What the leading source was chosen *with*, or ``None`` if none was.

        The payload under the head of :attr:`sources`, so a caller wanting the one
        answer does not have to index a ranking to reach it. ``None`` covers all
        three ways there is no such thing: an abstention, the default selection
        (which names no source in particular), and a selector that ranks without
        pricing.
        """
        if not self.sources:
            return None
        return self.payload.get(self.sources[0])

    async def wait(self) -> None:
        """Block until the chosen sources are usable (returns at once if ready)."""
        if self.ready is not None:
            await self.ready()

    async def settled(self) -> "Selection[_P]":
        """This selection with its gate spent: awaited, then dropped.

        What a plane reached as a service answers with. :attr:`ready` is a closure,
        so it cannot cross the boundary a handle stands for -- the ranking and the
        prices can, being values. Awaiting it here is also what makes the answer
        true when it arrives: the caller is about to read from these sources, and
        a ranking released early names a volume holding nothing yet.
        """
        await self.wait()
        if self.ready is None:
            return self
        return replace(self, ready=None)

    @property
    def head(self) -> Optional[VolumeId]:
        """The leading source, or ``None`` if this names none in particular.

        The id, where :attr:`winner` is the price under it, and the *best* source once
        this has been folded (:meth:`sort`, :meth:`max`). ``None`` for both empties,
        which a caller reading the head cannot tell apart and does not need to: neither
        one names a source to act on.
        """
        if not self.sources:
            return None
        return self.sources[0]

    def only(self, sources: Sequence[VolumeId]) -> "Selection[_P]":
        """This selection cut down to ``sources``, in the order given.

        The readiness gate rides along and each kept source keeps its key and its
        price, so one selection narrowed by somebody else still answers for whoever
        built it.
        """
        kept = tuple(sources)
        return Selection(
            sources=kept,
            key=None if self.key is None else {
                s: self.key[s] for s in kept if s in self.key
            },
            payload={s: self.payload[s] for s in kept if s in self.payload},
            ready=self.ready,
        )

    def take(self, n: int) -> "Selection[_P]":
        """The leading ``n`` sources, key, price and gate intact."""
        return self.only((self.sources or ())[:n])

    def require(self, ok: Callable[[VolumeId], bool]) -> "Selection[_P]":
        """This selection if its head satisfies ``ok``, else the abstention.

        The narrowing a caller does *to* a ranking, as a method on the ranking, so a
        test an application owns needs no object to live in and composes by being
        called again.

        All or nothing: filtering the head out would **promote** the source behind it,
        and a ranking need not be in the order ``ok`` measures -- the sources behind the
        head are the ones the ranking preferred *less*, so promoting one on a raw
        measurement would overrule it from outside. Which is why a caller folds first
        (:meth:`sort`, :meth:`max`): on an unfolded selection the head this judges is
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
        return Selection.of([])


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


class Selector(ABC, Generic[_S, _P]):
    """Rank the sources that should serve a subject, and say when they are usable.

    Written with the subject it takes and the price it hands each source back with
    (``AnySelector[Request, Plan]``), so both halves of ``select``'s signature are
    in the header a reader already looks at. ``None`` is the price of a selector
    that only ranks.

    A utility a control plane consults, and deliberately **not** a
    :class:`~proposed.plane.ControlPlane`: a run never holds one, so it needs no
    sensor to harvest and no service in front of it. Everything shared by every
    kind lives here -- ``select``, ``subject_type``, the view -- so
    :class:`KeySelector` and :class:`AnySelector` cannot drift apart. Implement one
    of those, or this base directly.
    """

    name = "selector"

    #: What :meth:`select` takes, as a value: :pep:`484` erases the parameter, so
    #: ``isinstance(x, Selector[Sequence[Key], None])`` raises and the one place a
    #: subject is *checked* needs something comparable. Set by
    #: :meth:`__init_subclass__` from the parameter, so the annotation and the value
    #: cannot disagree.
    subject_type: Any = Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Read :attr:`subject_type` off whichever base bound :data:`_S`.

        Positional, not "the first argument": a base may bind the price and leave
        the subject alone (``KeySelector[None]``), so the argument to read is the
        one lining up with :data:`_S` in that base's own ``__parameters__``. A class
        no base of which binds :data:`_S` inherits its parent's subject, which is how
        both a priced-only header (``KeySelector[int]``) and one that leaves the price
        open too (``KeySelector[_P]``) resolve. One that declares ``subject_type``
        itself is left alone, since a computed subject would be overwritten here.

        Reads ``cls.__dict__`` rather than :func:`getattr`, which would find the
        base's ``__orig_bases__`` and give every unparameterized subclass the bare
        type variable.
        """
        super().__init_subclass__(**kwargs)
        if "subject_type" in vars(cls):
            return
        for base in cls.__dict__.get("__orig_bases__", ()):
            params = getattr(get_origin(base) or base, "__parameters__", ())
            if _S not in params:
                continue
            subject = get_args(base)[params.index(_S)]
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
    async def select(self, subject: _S, requester: str) -> Selection[_P]:
        """Rank the sources that should serve ``subject`` for ``requester``.

        Both types are whatever this selector was parameterized with, and
        :attr:`subject_type` is ``_S`` as a value a check can compare.
        """


class KeySelector(Selector[Sequence[Key], _P]):
    """A selector whose subject is **keys**: which volume serves these bytes.

    The store's own question, and what a control plane answering "which volumes
    should serve this read" ranks with. A type as well as a :attr:`subject_type`, so
    that a chain can check its links are all over keys (:class:`FirstMatch`).
    """



class AnySelector(Selector[_S, _P]):
    """A selector whose subject is an **application payload**.

    An application question that happens to be a selection -- which host prefills,
    which host decodes, which of these priced candidates wins. :attr:`subject_type`
    stays ``Any`` here and a subclass narrows it: this package cannot name a type an
    application invented. Being this type instead of :class:`KeySelector` says the
    subject is not the store's, which is worth saying where the ranking looks
    otherwise identical.
    """


class NaiveKeySelector(KeySelector[_P]):
    """Every holder, in directory order, usable now.

    Precisely the real directory's own answer, so this returns the empty
    :class:`Selection` rather than re-deriving it: a read preferring what it names is
    byte-identical to a read that names nothing (:func:`prefer`).

    Generic in the price because it quotes none: an empty payload is a payload in
    whatever terms the chain it tails prices in.
    """

    name = "naive"

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        return Selection()


def declares(
    own: Sequence[type], base: "Selector[Any, Any]"
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


def declared(view: Any, selector: "Selector[Any, Any]") -> Any:
    """The view ``selector`` declared, out of the one a combinator was handed.

    What a combinator narrows to for a selector it holds is that selector's own header
    -- otherwise a chain would be the one place a declaration is not a fact, and both of
    ``dedup_sim``'s links sit inside one. Which is only reachable because the combinator
    declared the link's reads along with its own (:func:`declares`). Something that is
    no view at all is handed on untouched, which only a selector declaring nothing can
    be attached to anyway (:attr:`Selector.sensors`).
    """
    return view.subset(*selector.sensors) if selector.sensors else view


class FirstMatch(KeySelector[_P]):
    """Ask each selector in order; the first one that answers is the answer.

    A :class:`Selection` can be empty in two ways, and they mean opposite things:

    * ``Selection()`` -- ``sources is None`` -- is *every holder, in directory
      order*, the decision :class:`NaiveKeySelector` makes. It **wins the chain**, and
      the selectors behind it are never consulted.
    * ``Selection.of([])`` names nobody. That is the **abstention**, and it falls
      through.

    An exhausted chain abstains in turn, which keeps chaining associative:
    ``FirstMatch([FirstMatch([a, b]), c])`` still reaches ``c``, as it could not if
    the inner chain's exhaustion arrived looking like a decision. A chain that
    should always answer ends with a :class:`NaiveKeySelector`. The winner is returned
    exactly as built, so a readiness gate rides along untouched.

    Over keys, and so a :class:`KeySelector` itself: the subject goes down every link
    untouched, so a link reading it as anything else would answer a question it was not
    asked -- checked at construction rather than trusted. A chain of alternatives over
    an application's own subject has no caller, and would be this class with ``keys``
    read as that subject.

    Args:
        selectors: consulted left to right, each a :class:`KeySelector` pricing in the
            chain's terms -- a chain answers as its links do, which is what the price
            parameter says. An empty chain is legal and abstains.
    """

    name = "first-match"

    def __init__(self, selectors: Sequence[KeySelector[_P]]) -> None:
        self.selectors: Tuple[KeySelector[_P], ...] = tuple(selectors)
        wrong = [type(s).__name__ for s in self.selectors if not isinstance(s, KeySelector)]
        if wrong:
            raise TypeError(
                f"a FirstMatch chain selects over keys, so every link must be a "
                f"KeySelector; {', '.join(wrong)} "
                f"{'is' if len(wrong) == 1 else 'are'} not"
            )

    def attach(self, view: Any) -> "FirstMatch[_P]":
        """Hand every wrapped selector the view it declared, answering or not.

        One that senses through a view of its own must be brought up even if it
        never answers, so a link behind an earlier answer is still sensing when its
        turn comes.
        """
        for selector in self.selectors:
            selector.attach(declared(view, selector))
        return self

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        """The first non-abstaining answer, or an abstention if there is none."""
        for selector in self.selectors:
            selection = await selector.select(keys, requester)
            if selection.sources is None or selection.sources:
                return selection
        return Selection.of([])


class Balance(Selector[_S, _P]):
    """One ranking, annotated with how loaded each source it named is.

    Load spreading as a layer over *any* ranking that says what orders it, rather than
    a property of one ranking: two sources a ranking keys the same are left to the id
    the fold breaks its ties on, so every read goes to the same volume. This puts
    something that changes ahead of that tie-break.

    What it senses, and nothing else: :class:`~proposed.view.LoadView`, whose
    ``named()`` says what has lately been sent at each source. So this combinator holds
    no tally of its own -- what it appends is an observation somebody else keeps, moved
    by the decision that names a source and read here -- and what that number means is
    stated once, on the view.

    **No arithmetic**, so there is nothing here for a caller to supply: it appends the
    load as one more dimension (:meth:`Selection.annotated`) and whoever folds decides
    what a busy source costs, which is the application's own trade -- blocks of prefix
    run against reads routed at a host, seconds of link time against seconds of queue.
    The base's key and payload both ride through, so a fold reads the ranking's own
    numbers beside this one and a caller pricing the winner against something of its
    own still reads what the *base* said about it.

    Over any subject, because the subject is handed to ``ranking`` untouched and never
    read here -- one wrapper serves a ranking over keys and one over an application's
    own candidates alike. Which *kind* it is follows what it wraps (:meth:`__new__`).

    Determinism: the load is read once per ranking, with nothing awaited between the
    read and the dimension it appends, so no fold can land inside one answer. Nothing
    here reads a wall clock or an unseeded RNG, and nothing here decides an order at
    all (:meth:`Selection.sort`).

    Args:
        ranking: the selector asked, which must key every source it ranks. What it
            senses is declared here too (:func:`declares`).
    """

    name = "balance"
    sensors = (LoadView,)

    def __new__(
        cls, ranking: "Selector[_S, _P]", *args: Any, **kwargs: Any
    ) -> "Balance[_S, _P]":
        """Take the kind of the ranking wrapped: over keys, a :class:`KeySelector`.

        A combinator asks whatever question its ranking asks, so the kind cannot be
        declared once here -- and it has to be a type, because that is how a chain
        checks its links (:class:`FirstMatch`). So a balanced key ranking is a chain
        link and a balanced application ranking is refused by one, which is the same
        answer the ranking itself would get.
        """
        return object.__new__(
            _KeyBalance if cls is Balance and isinstance(ranking, KeySelector)
            else cls
        )

    def __init__(self, ranking: Selector[_S, _P]) -> None:
        self.ranking = ranking
        #: What it annotates, which is what it takes: the subject is not this
        #: combinator's, so it is read off the ranking rather than declared.
        self.subject_type = ranking.subject_type
        #: Load, and whatever the ranking senses (:func:`declares`).
        self.sensors = declares((LoadView,), ranking)

    def attach(self, view: Any) -> "Balance[_S, _P]":
        """Sense through ``view``, and hand the ranking the view it declared."""
        super().attach(view)
        self.ranking.attach(declared(view, self.ranking))
        return self

    async def select(self, subject: _S, requester: str) -> Selection[_P]:
        """``ranking``'s answer with the load at each source appended to its key.

        Every source is annotated, not just whichever one led, so a caller that folds
        this and rejects the winner has the rest measured too. Nothing is ordered here
        and nothing is written: the load is an observation, and what moves it is the
        decision this answer is consulted for.

        An answer with no source to measure goes back untouched. Both empties qualify,
        for the same reason and not by accident: an abstention names nobody, and the
        default selection names every holder in directory order rather than any source
        in particular.

        Raises:
            ValueError: if ``ranking`` left a source it ranked with no key. The load
                would then be the *whole* of the order rather than a dimension behind
                the ranking's own, which is overruling it.
        """
        ranked = await self.ranking.select(subject, requester)
        if not ranked.sources:
            return ranked
        keyed = ranked.key or {}
        unkeyed = [s for s in ranked.sources if s not in keyed]
        if unkeyed:
            raise ValueError(
                f"{type(self.ranking).__name__} ranked {', '.join(unkeyed)} without "
                f"keying them, so a load appended here would be the whole of the "
                f"order: a ranking under one keys every source it ranks"
            )
        load = self.view.load.named()
        return ranked.annotated(
            {source: load.get(source, 0) for source in ranked.sources}
        )


class _KeyBalance(Balance[Sequence[Key], _P], KeySelector[_P]):
    """A :class:`Balance` over keys, and so the store's own question.

    What :class:`Balance` answers with when the ranking it wraps is a
    :class:`KeySelector`, so such a one is a chain link like the ranking under it.
    """
