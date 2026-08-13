"""One question, two subjects: which sources serve this, and when::

    Selector[Subject, Price].select(subject, requester) -> Selection[Price]

The answer is always volume ids, best first, plus the moment they become usable
(and, for a selector handed candidates it did not price, what each one priced at).
What differs is the **subject**, and every selector names its own in its header::

    class RoutedPull(KeySelector[None]):        # keys, ranked without pricing
    class Hosts(AnySelector[Request, Plan]):    # an application's subject, priced

Two of those subjects are *types* as well, because a run reaches a selector by
which one it is:

* :class:`KeySelector` -- ``Sequence[Key]``: which volume serves these bytes, the
  store's own question. ``dedup_sim`` wants a reader routed to a *peer* about to
  hold the key, ``kvcache_sim`` the peer holding the longest reusable prefix. The
  only kind installable in the controller.
* :class:`AnySelector` -- an application's own subject, whatever it is: which peer
  to source a prefix from, which host prefills, which host decodes. Never
  installed; an application's hosts reach one as a service of its own. Its
  ``subject_type`` stays ``Any`` and a subclass narrows it, since this package
  cannot name a type an application invented.

Types rather than ``subject_type`` alone, because taking keys does not by itself
mean *the store may ask you*: :class:`Refine` and ``kvcache_sim``'s reuse axis take
the same keys and must not be installed. A selector that is neither type is reached
from nowhere -- another selector consults it -- and being plain :class:`Selector`
is what keeps it out of both call sites.

Admission and SLO gates are neither: an answer that is not a ranked set of sources
does not belong in a :class:`Selection` at all. A gate rides *with* one instead --
a selector that refuses abstains (``Selection.of([])``), and what it would have
answered is simply not in the ranking.

Selectors compose two ways, and neither one is a selector holding another:

* :class:`FirstMatch` picks between alternatives -- ask each in order, take the
  first answer. It wraps either kind and *is* a plain :class:`Selector` whatever
  it wraps, so a chain mixing the two kinds is possible and harmless -- and
  thereby barred from the controller, the one place the mixture would matter. A
  chain whose links all take keys is a :class:`KeySelectorChain`, which is a
  :class:`KeySelector` and may be installed.
* :class:`Refine` funnels a single answer -- one selector's ranking, narrowed by
  each :class:`Refinement` behind it. That is how a test an application owns is
  applied to a ranking the store produced, with the composition in the object
  that holds both rather than inside either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Awaitable, Callable, Dict, Generic, Mapping, Optional, Protocol,
    Sequence, Tuple, TypeVar, get_args, get_origin,
)

from proposed.deployment import Key, VolumeId
from proposed.plane import ControlPlane
from proposed.view import View

__all__ = [
    "Ready", "Selection", "DecisionLog", "Selector", "KeySelector", "AnySelector",
    "NaiveKeySelector", "FirstMatch", "KeySelectorChain",
    "Refinement", "Refine", "AbstainOnSelf", "TakeHead",
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
            on its own, so a ``None`` selection leaves the controller's answer
            untouched.
        ready: optional gate awaited before the answer is released, for a selector
            that routes a requester to a peer which has not registered yet.
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
        payload: Optional[Mapping[VolumeId, _P]] = None,
    ) -> "Selection[_P]":
        """A selection ranking ``sources`` best-first."""
        return cls(
            sources=tuple(sources), ready=ready, payload=dict(payload or {})
        )

    @classmethod
    def priced(
        cls,
        candidates: Sequence[Tuple[VolumeId, _P]],
        *,
        ready: Optional[Ready] = None,
    ) -> "Selection[_P]":
        """Ranked ids and what each was priced at, from one ordered sequence.

        The safe form of :meth:`of` for a selector that prices: one argument, so
        the ranking and the prices cannot be built out of step with each other.
        """
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

    def narrow(
        self, located: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Restrict a directory answer to the selected sources, in rank order.

        A key none of the selected sources holds is left untouched: the selection
        is a preference, not a filter that can make data disappear.
        """
        if self.sources is None:
            return located
        scoped: Dict[str, Dict[str, Any]] = {}
        for key, volume_map in located.items():
            ranked = {
                vid: volume_map[vid] for vid in self.sources if vid in volume_map
            }
            scoped[key] = ranked if ranked else volume_map
        return scoped


class DecisionLog(Protocol):
    """Somewhere a selector can explain itself.

    Optional and never load-bearing: a selector must behave identically with none
    attached. Declared here so a selector need not name the simulator's trace.
    """

    def record(self, at: float, kind: str, message: str) -> None:
        ...


class Selector(ControlPlane, ABC, Generic[_S, _P]):
    """Rank the sources that should serve a subject, and say when they are usable.

    Written with the subject it takes and the price it hands each source back with
    (``AnySelector[Request, Plan]``), so both halves of ``select``'s signature are
    in the header a reader already looks at. ``None`` is the price of a selector
    that only ranks.

    Everything shared by every kind lives here -- ``select``, ``subject_type``,
    the :class:`~proposed.plane.ControlPlane` lifecycle -- so :class:`KeySelector`
    and :class:`AnySelector` cannot drift apart. Implement one of those, or this
    base directly for a selector nothing reaches from outside.

    A selector that must wait for a source to appear subscribes to the directory
    in its own :meth:`attach` (:meth:`proposed.deployment.Controller.subscribe`),
    reached through the view it is handed there. Nothing about that is declared
    here: a wakeup is the directory's to deliver, not a member every selector owes.
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
        header resolve. One that declares ``subject_type`` itself is left alone --
        :class:`Refine` computes it from its source and would lose the property.

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

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Keep the view this selector reads the directory through.

        A selector consulted inside ``locate_volumes`` runs on the same side as the
        service consulting it, so the sensor is the run's and not a per-call
        argument: one selector, one view, whoever asks.
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

    The store's own question, and the only kind a controller may install in
    ``locate_volumes``: being this type *is* the claim, which is what makes
    installing one checkable (``isinstance(control, KeySelector)`` at the seam).

    A type as well as a :attr:`subject_type`, because taking keys does not by
    itself mean the store may ask you -- a selector an application consults while
    routing takes the same keys and must stay out of the directory.
    """



class AnySelector(Selector[_S, _P]):
    """A selector whose subject is an **application payload**.

    An application question that happens to be a selection -- which host prefills,
    which host decodes, which of these priced candidates wins. :attr:`subject_type`
    stays ``Any`` here and a subclass narrows it: this package cannot name a type an
    application invented, and the run does not need the name -- it reaches one of
    these by *being* one. Being this type instead of :class:`KeySelector` is what
    keeps it out of the controller, where a subject the store cannot read would make
    ``locate_volumes`` answer a question it was not asked.

    One that an application's own hosts ask is given a service of its own by the
    run (:mod:`realsim.seams.control_plane_service`), so the subject and the answer
    both have to be values a wire could carry: the ranking is source ids, and what
    the winner was chosen with rides in :attr:`Selection.payload`.
    """


class NaiveKeySelector(KeySelector[None]):
    """Every holder, in directory order, usable now.

    Precisely the real directory's own answer, so this returns the empty
    :class:`Selection` rather than re-deriving it: installing it is byte-identical
    to installing no selector at all.
    """

    name = "naive"

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[None]:
        return Selection()


class FirstMatch(Selector[_S, _P]):
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

    Args:
        selectors: consulted left to right, each taking the chain's subject and
            pricing in its terms -- a chain answers as its links do, which is what
            the two parameters say. An empty chain is legal and abstains.
    """

    name = "first-match"

    def __init__(self, selectors: Sequence[Selector[_S, _P]]) -> None:
        self.selectors: Tuple[Selector[_S, _P], ...] = tuple(selectors)

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Hand the stack's ports to every wrapped selector, answering or not.

        One that senses through a view of its own must be brought up even if it
        never answers -- and one that subscribes to the directory does it here, so
        a link behind an earlier answer still has its wakeups.
        """
        for selector in self.selectors:
            selector.attach(view, transfer_cost)

    async def select(self, subject: _S, requester: str) -> Selection[_P]:
        """The first non-abstaining answer, or an abstention if there is none."""
        for selector in self.selectors:
            selection = await selector.select(subject, requester)
            if selection.sources is None or selection.sources:
                return selection
        return Selection.of([])


class KeySelectorChain(FirstMatch[Sequence[Key], _P], KeySelector[_P]):
    """A :class:`FirstMatch` every link of which selects over keys, so it does too.

    Installable in the controller, which a plain chain is not. The claim is checked
    at construction rather than trusted, so being this type says exactly what
    :class:`KeySelector` says: a ``locate_volumes`` hands its subject down every link,
    and one that read it as anything but keys would answer a question the directory
    did not ask.

    Args:
        selectors: consulted left to right; each must be a :class:`KeySelector`.
            Typed as the base's, since the claim is what is *checked* below -- an
            annotation would only restate what a caller can ignore.
    """

    name = "selector-chain"

    def __init__(self, selectors: Sequence[Selector[Sequence[Key], _P]]) -> None:
        super().__init__(selectors)
        wrong = [type(s).__name__ for s in self.selectors if not isinstance(s, KeySelector)]
        if wrong:
            raise TypeError(
                f"a KeySelectorChain selects over keys, so every link must be a KeySelector; "
                f"{', '.join(wrong)} {'is' if len(wrong) == 1 else 'are'} not"
            )


class Refinement(ControlPlane, ABC, Generic[_S, _P]):
    """Narrow a ranking some selector already produced.

    Not a :class:`Selector`: it has no subject of its own, and ``select`` has
    nowhere to put an incoming ranking. :class:`Refine` is what holds the selector,
    which is what keeps one selector from holding another. Its parameters are that
    selector's, borrowed -- the subject it is handed and the price it must not lose.

    Senses through the run's view like anything else in a chain (:meth:`attach`).
    """

    name = "refinement"

    #: What this refinement reads to decide: ``None`` until :meth:`attach`.
    view: Optional[View] = None

    def attach(self, view: Any, transfer_cost: Any) -> None:
        self.view = view

    @abstractmethod
    async def refine(
        self, selection: Selection[_P], subject: _S, requester: str
    ) -> Selection[_P]:
        """``selection``, narrowed. Handed the subject too, since a test may read it.

        Called only with a selection that names at least one source, so an
        implementation may index the head without checking.
        """


class Refine(Selector[_S, _P]):
    """One selector's ranking, put through each :class:`Refinement` in turn.

    A plain :class:`Selector` whatever it wraps, for :class:`FirstMatch`'s reason:
    narrowing a selector's ranking with an application's test asks something the
    store cannot read, and being neither subtype is what bars the result from the
    controller.

    A step that abstains (``Selection.of([])``) ends the funnel -- the steps behind
    it are not consulted and the abstention is the answer. ``Selection()``, every
    holder in directory order, is the one thing a step cannot narrow, so a source
    that answers with one in front of a step raises rather than silently handing
    back an unfiltered ranking.

    Args:
        source: produces the ranking. Either kind of selector, and what both
            parameters are read off: a funnel answers its source's question with
            its source's prices.
        steps: applied left to right, each narrowing that same ranking.
    """

    name = "refine"

    def __init__(
        self, source: Selector[_S, _P], *steps: Refinement[_S, _P]
    ) -> None:
        self.source = source
        self.steps: Tuple[Refinement[_S, _P], ...] = tuple(steps)

    @property
    def subject_type(self) -> Any:
        """Its source's: a funnel narrows an answer, it does not reinterpret one.

        Which is why a :class:`Refine` over a :class:`KeySelector` is still not one
        -- it takes keys and is barred from the directory all the same.
        """
        return self.source.subject_type

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Hand the stack's ports to the source and every step."""
        super().attach(view, transfer_cost)
        self.source.attach(view, transfer_cost)
        for step in self.steps:
            step.attach(view, transfer_cost)

    async def select(self, subject: _S, requester: str) -> Selection[_P]:
        """The source's ranking narrowed by every step, or the first abstention."""
        selection = await self.source.select(subject, requester)
        for step in self.steps:
            if selection.sources is None:
                raise ValueError(
                    f"{self.source.name} named every holder in directory order, "
                    f"which {step.name} cannot narrow: refine a source that ranks"
                )
            if not selection.sources:
                return selection
            selection = await step.refine(selection, subject, requester)
        return selection


class AbstainOnSelf(Refinement[Any, _P]):
    """Abstain when the ranking's head is the requester itself.

    A source is a peer, and a requester does not fetch what it already holds. The
    whole selection goes rather than just the head: the ranking preferred the
    requester, so nothing behind it is preferred to what the requester has.

    ``Refinement[Any, _P]``, as :class:`TakeHead` is: a step that reads the ranking
    and the requester and not the subject fits behind a source of any kind.
    """

    name = "abstain-on-self"

    async def refine(
        self, selection: Selection[_P], subject: Any, requester: str
    ) -> Selection[_P]:
        if selection.sources[0] == requester:
            return Selection.of([])
        return selection


class TakeHead(Refinement[Any, _P]):
    """Keep the best-ranked source and drop the rest, price and gate intact."""

    name = "take-head"

    async def refine(
        self, selection: Selection[_P], subject: Any, requester: str
    ) -> Selection[_P]:
        return selection.only(selection.sources[:1])
