"""One question, two subjects: which sources serve this, and when::

    Selector[Subject, Price].select(subject, requester) -> Selection[Price]

The answer is always volume ids, best first, plus the moment they become usable
(and, for a selector handed candidates it did not price, what each one priced at).
What differs is the **subject**, and every selector names its own in its header::

    class RoutedPull(KeySelector[None]):        # keys, ranked without pricing
    class Hosts(AnySelector[Request, Plan]):    # an application's subject, priced

A selector is a **utility**, not a plane. Nothing outside a capability reaches one:
a run knows about the capability's :class:`~proposed.plane.ControlPlane`, that plane
declares the questions its callers may ask, and a selector is one of the things it
may work the answer out with. So a selector needs no lifecycle beyond the sensor it
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

Selectors compose one way, and it is not a selector holding another:
:class:`FirstMatch` picks between alternatives -- ask each in order, take the first
answer. Every link takes keys, checked at construction, so the chain is a
:class:`KeySelector` too and says so in its type.

Narrowing an answer is not a composition of selectors at all: a test an application
owns is applied to the ranking it was given, by the caller that has both
(:meth:`Selection.require`, :meth:`Selection.take`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Awaitable, Callable, Dict, Generic, Mapping, Optional, Protocol,
    Sequence, Tuple, TypeVar, get_args, get_origin,
)

from proposed.deployment import Key, VolumeId
from proposed.view import View

__all__ = [
    "Ready", "Selection", "prefer", "DecisionLog", "Selector", "KeySelector",
    "AnySelector", "NaiveKeySelector", "FirstMatch",
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


@dataclass(frozen=True)
class Selection(Generic[_P]):
    """Ranked sources for one subject, plus when they become usable.

    Args:
        sources: volume ids, best first. ``None`` -- the default -- means *every
            holder, in directory order*, which is what the real directory returns
            on its own, so a ``None`` selection leaves the store's answer untouched
            (:func:`prefer`).
        ready: optional gate, for a selector that routes a requester to a peer
            which has not registered yet. Spent by :meth:`settled` before the
            answer travels, never handed to whoever asked.
        payload: ``source id -> what this selector holds about that source``,
            application-defined because this package cannot read an application's
            values. A ranking alone loses what produced it, and a selector handed
            alternatives somebody else priced has to give the winner's price back
            with it. Keyed by id rather than parallel to ``sources``, so a selection
            cannot be built out of step with itself; empty for a selector that only
            ranks.
    """

    sources: Optional[Tuple[VolumeId, ...]] = None
    ready: Optional[Ready] = None
    payload: Mapping[VolumeId, _P] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        sources: Sequence[VolumeId],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_P]":
        """A selection ranking ``sources`` best-first, priced at nothing.

        The two builders are the two kinds of selector: this one for a ranking,
        :meth:`priced` for a ranking whose sources carry what they were priced at.
        Prices are not reachable from here, so a ranking and the prices for it cannot
        be built out of step with each other.
        """
        return cls(sources=tuple(sources), ready=ready)

    @classmethod
    def priced(
        cls,
        candidates: Sequence[Tuple[VolumeId, _P]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_P]":
        """Ranked ids and what each was priced at, from one ordered sequence."""
        return cls(
            sources=tuple(i for i, _ in candidates),
            ready=ready,
            payload={i: price for i, price in candidates},
        )

    @property
    def winner(self) -> Optional[_P]:
        """What the best-ranked source was chosen *with*, or ``None`` if none was.

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
        return Selection(sources=self.sources, payload=self.payload)

    @property
    def head(self) -> Optional[VolumeId]:
        """The best-ranked source, or ``None`` if this names none in particular.

        The id, where :attr:`winner` is the price under it. ``None`` for both empties,
        which a caller reading the head cannot tell apart and does not need to: neither
        one names a source to act on.
        """
        if not self.sources:
            return None
        return self.sources[0]

    def only(self, sources: Sequence[VolumeId]) -> "Selection[_P]":
        """This selection cut down to ``sources``, in the order given.

        The readiness gate rides along and each kept source keeps its price, so
        one selection narrowed by somebody else still answers for whoever built it.
        """
        kept = tuple(sources)
        return Selection(
            sources=kept,
            ready=self.ready,
            payload={s: self.payload[s] for s in kept if s in self.payload},
        )

    def take(self, n: int) -> "Selection[_P]":
        """The best ``n`` sources, price and gate intact."""
        return self.only((self.sources or ())[:n])

    def require(self, ok: Callable[[VolumeId], bool]) -> "Selection[_P]":
        """This selection if its head satisfies ``ok``, else the abstention.

        The narrowing a caller does *to* a ranking, as a method on the ranking, so a
        test an application owns needs no object to live in and composes by being
        called again.

        All or nothing: filtering the head out would **promote** the source behind it,
        and a ranking need not be in the order ``ok`` measures -- the sources behind the
        head are the ones the ranking preferred *less*, so promoting one on a raw
        measurement would overrule it from outside. An abstention is returned unchanged,
        since there is no head to judge.

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
    ``cluster`` to harvest and no service in front of it. Everything shared by every
    kind lives here -- ``select``, ``subject_type``, the sensor -- so
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
        both a narrowing subclass (``SpreadReadsKeySelector``) and a priced-only
        header resolve. One that declares ``subject_type`` itself is left alone, since
        a computed subject would be overwritten here.

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

    def attach(self, view: Any) -> None:
        """Keep the sensor this selector senses and prices through.

        The one the plane holding it was handed
        (:meth:`proposed.plane.ControlPlane.attach`), passed straight down: a
        selector runs beside the directory it senses, so the sensor is the run's and
        not a per-call argument -- one selector, one view, whoever asks.
        """
        self.view = view

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


class NaiveKeySelector(KeySelector[None]):
    """Every holder, in directory order, usable now.

    Precisely the real directory's own answer, so this returns the empty
    :class:`Selection` rather than re-deriving it: a read preferring what it names is
    byte-identical to a read that names nothing (:func:`prefer`).
    """

    name = "naive"

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[None]:
        return Selection()


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

    def attach(self, view: Any) -> None:
        """Hand the stack's ports to every wrapped selector, answering or not.

        One that senses through a view of its own must be brought up even if it
        never answers, so a link behind an earlier answer is still sensing when its
        turn comes.
        """
        for selector in self.selectors:
            selector.attach(view)

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        """The first non-abstaining answer, or an abstention if there is none."""
        for selector in self.selectors:
            selection = await selector.select(keys, requester)
            if selection.sources is None or selection.sources:
                return selection
        return Selection.of([])
