"""One question, two subjects: which sources serve this, and when::

    Selector.select(subject, requester) -> Selection

The answer is always a ranked list of sources plus the moment they become usable
(and, for a selector handed candidates it did not price, what each one priced at).
What differs is the subject, and that difference is a type:

* :class:`Policy` -- the subject is **keys**, so the question is the store's:
  which volume serves these bytes. ``dedup_sim`` wants a reader routed to a *peer*
  about to hold the key, ``kvcache_sim`` the peer holding the longest reusable
  prefix. The only kind installable in the controller.
* :class:`Placement` -- the subject is an application payload, so the question is
  the application's: which peer to source a prefix from, which host prefills,
  which host decodes. Never installed in the controller; an application's hosts
  reach one as a service of its own.

Two subtypes rather than one generic interface because a controller consults a
:class:`Policy` inside ``locate_volumes``: with a single type, a selector whose
subject is not keys could be installed there, and :class:`Selection` would be a
union serving neither caller. The subtype is the marker.

Admission and SLO gates are neither: an answer that is not a ranked set of sources
does not belong in a :class:`Selection` at all. A gate rides *with* one instead --
a selector that refuses abstains (``Selection.of([])``), and what it would have
answered is simply not in the ranking.

Selectors compose two ways, and neither one is a selector holding another:

* :class:`FirstMatch` picks between alternatives -- ask each in order, take the
  first answer. It wraps either kind and *is* a plain :class:`Selector` whatever
  it wraps, so a chain mixing the two kinds is possible and harmless -- and
  thereby barred from the controller, the one place the mixture would matter. A
  chain whose links are all policies is a :class:`PolicyChain`, which is a
  :class:`Policy` and may be installed.
* :class:`Refine` funnels a single answer -- one selector's ranking, narrowed by
  each :class:`Refinement` behind it. That is how a test an application owns is
  applied to a ranking the store produced, with the composition in the object
  that holds both rather than inside either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple,
)

from proposed.plane import ControlPlane
from proposed.view import View

__all__ = [
    "Ready", "Selection", "DecisionLog", "Selector", "Policy", "Placement",
    "NaivePolicy", "FirstMatch", "PolicyChain",
    "Refinement", "Refine", "AbstainOnSelf", "TakeHead",
]

# A readiness gate: called with no arguments, awaited until the chosen source is
# usable. ``None`` means "usable now".
Ready = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Selection:
    """Ranked sources for one subject, plus when they become usable.

    Args:
        sources: volume ids, best first. ``None`` -- the default -- means *every
            holder, in directory order*, which is what the real directory returns
            on its own, so a ``None`` selection leaves the controller's answer
            untouched.
        ready: optional gate awaited before the answer is released, for a policy
            that routes a requester to a peer which has not registered yet.
        payload: ``source id -> what this selector holds about that source``,
            application-defined because this package cannot read an application's
            values. A ranking alone loses what produced it, and a selector handed
            alternatives somebody else priced has to give the winner's price back
            with it. Keyed by id rather than parallel to ``sources``, so a selection
            cannot be built out of step with itself; empty for a selector that only
            ranks.
    """

    sources: Optional[Tuple[str, ...]] = None
    ready: Optional[Ready] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        sources: Sequence[str],
        *,
        ready: Optional[Ready] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> "Selection":
        """A selection ranking ``sources`` best-first."""
        return cls(
            sources=tuple(sources), ready=ready, payload=dict(payload or {})
        )

    @property
    def winner(self) -> Optional[Any]:
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

    def only(self, sources: Sequence[str]) -> "Selection":
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
    """Somewhere a policy can explain itself.

    Optional and never load-bearing: a policy must behave identically with none
    attached. Declared here so a policy need not name the simulator's trace.
    """

    def record(self, at: float, kind: str, message: str) -> None:
        ...


class Selector(ControlPlane, ABC):
    """Rank the sources that should serve a subject, and say when they are usable.

    Everything shared by the two kinds lives here -- ``select`` and the
    :class:`~proposed.plane.ControlPlane` lifecycle -- so :class:`Policy` and
    :class:`Placement` cannot drift apart. Implement one of those; only the
    combinators below sit on the base itself.

    A selector that must wait for a source to appear subscribes to the directory
    in its own :meth:`attach` (:meth:`proposed.deployment.Controller.subscribe`),
    reached through the view it is handed there. Nothing about that is declared
    here: a wakeup is the directory's to deliver, not a member every selector owes.
    """

    name = "selector"

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
    async def select(self, subject: Any, requester: str) -> Selection:
        """Rank the sources that should serve ``subject`` for ``requester``.

        ``Any``, because what a subject *is* is the subtype's claim: keys for a
        :class:`Policy`, an application's own values for a :class:`Placement`.
        """


class Policy(Selector):
    """A selector whose subject is **keys**: which volume serves these bytes.

    The store's own question, and the only kind a controller may install in
    ``locate_volumes``. Adds no member to :class:`Selector`; being this type *is*
    the claim that ``subject`` is a set of keys, which is what makes installing one
    checkable (``isinstance(control, Policy)`` at the seam).
    """


class Placement(Selector):
    """A selector whose subject is an **application payload**.

    An application question that happens to be a selection -- which peer to source
    a prefix from, which host prefills, which host decodes. Adds no member to
    :class:`Selector`; being this type instead of :class:`Policy` is what keeps it
    out of the controller, where a subject the store cannot read would make
    ``locate_volumes`` answer a question it was not asked.

    One that an application's own hosts ask is given a service of its own by the
    run (:mod:`realsim.seams.placement_service`), so the subject and the answer
    both have to be values a wire could carry: the ranking is source ids, and what
    the winner was chosen with rides in :attr:`Selection.payload`.
    """


class NaivePolicy(Policy):
    """Every holder, in directory order, usable now.

    Precisely the real directory's own answer, so this returns the empty
    :class:`Selection` rather than re-deriving it: installing it is byte-identical
    to installing no policy at all.
    """

    name = "naive"

    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        return Selection()


class FirstMatch(Selector):
    """Ask each selector in order; the first one that answers is the answer.

    A :class:`Selection` can be empty in two ways, and they mean opposite things:

    * ``Selection()`` -- ``sources is None`` -- is *every holder, in directory
      order*, the decision :class:`NaivePolicy` makes. It **wins the chain**, and
      the selectors behind it are never consulted.
    * ``Selection.of([])`` names nobody. That is the **abstention**, and it falls
      through.

    An exhausted chain abstains in turn, which keeps chaining associative:
    ``FirstMatch([FirstMatch([a, b]), c])`` still reaches ``c``, as it could not if
    the inner chain's exhaustion arrived looking like a decision. A chain that
    should always answer ends with a :class:`NaivePolicy`. The winner is returned
    exactly as built, so a readiness gate rides along untouched.

    Args:
        selectors: consulted left to right. An empty chain is legal and abstains.
    """

    name = "first-match"

    def __init__(self, selectors: Sequence[Selector]) -> None:
        self.selectors: Tuple[Selector, ...] = tuple(selectors)

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Hand the stack's ports to every wrapped selector, answering or not.

        One that senses through a view of its own must be brought up even if it
        never answers -- and one that subscribes to the directory does it here, so
        a link behind an earlier answer still has its wakeups.
        """
        for selector in self.selectors:
            selector.attach(view, transfer_cost)

    async def select(self, subject: Any, requester: str) -> Selection:
        """The first non-abstaining answer, or an abstention if there is none."""
        for selector in self.selectors:
            selection = await selector.select(subject, requester)
            if selection.sources is None or selection.sources:
                return selection
        return Selection.of([])


class PolicyChain(FirstMatch, Policy):
    """A :class:`FirstMatch` every link of which selects over keys, so it does too.

    Installable in the controller, which a plain chain is not. The claim is checked
    at construction rather than trusted, so being this type says exactly what
    :class:`Policy` says: a ``locate_volumes`` hands its subject down every link,
    and one that read it as anything but keys would answer a question the directory
    did not ask.

    Args:
        selectors: consulted left to right; each must be a :class:`Policy`.
    """

    name = "policy-chain"

    def __init__(self, selectors: Sequence[Selector]) -> None:
        super().__init__(selectors)
        wrong = [type(s).__name__ for s in self.selectors if not isinstance(s, Policy)]
        if wrong:
            raise TypeError(
                f"a PolicyChain selects over keys, so every link must be a Policy; "
                f"{', '.join(wrong)} {'is' if len(wrong) == 1 else 'are'} not"
            )


class Refinement(ControlPlane, ABC):
    """Narrow a ranking some selector already produced.

    Not a :class:`Selector`: it has no subject of its own, and ``select`` has
    nowhere to put an incoming ranking. :class:`Refine` is what holds the selector,
    which is what keeps one selector from holding another.

    Senses through the run's view like anything else in a chain (:meth:`attach`).
    """

    name = "refinement"

    #: What this refinement reads to decide: ``None`` until :meth:`attach`.
    view: Optional[View] = None

    def attach(self, view: Any, transfer_cost: Any) -> None:
        self.view = view

    @abstractmethod
    async def refine(
        self, selection: Selection, subject: Any, requester: str
    ) -> Selection:
        """``selection``, narrowed. Handed the subject too, since a test may read it.

        Called only with a selection that names at least one source, so an
        implementation may index the head without checking.
        """


class Refine(Selector):
    """One selector's ranking, put through each :class:`Refinement` in turn.

    A plain :class:`Selector` whatever it wraps, for :class:`FirstMatch`'s reason:
    narrowing a policy's ranking with an application's test asks something the
    store cannot read, and being neither subtype is what bars the result from the
    controller.

    A step that abstains (``Selection.of([])``) ends the funnel -- the steps behind
    it are not consulted and the abstention is the answer. ``Selection()``, every
    holder in directory order, is the one thing a step cannot narrow, so a source
    that answers with one in front of a step raises rather than silently handing
    back an unfiltered ranking.

    Args:
        source: produces the ranking. Either kind of selector.
        steps: applied left to right.
    """

    name = "refine"

    def __init__(self, source: Selector, *steps: Refinement) -> None:
        self.source = source
        self.steps: Tuple[Refinement, ...] = tuple(steps)

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Hand the stack's ports to the source and every step."""
        super().attach(view, transfer_cost)
        self.source.attach(view, transfer_cost)
        for step in self.steps:
            step.attach(view, transfer_cost)

    async def select(self, subject: Any, requester: str) -> Selection:
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


class AbstainOnSelf(Refinement):
    """Abstain when the ranking's head is the requester itself.

    A source is a peer, and a requester does not fetch what it already holds. The
    whole selection goes rather than just the head: the ranking preferred the
    requester, so nothing behind it is preferred to what the requester has.
    """

    name = "abstain-on-self"

    async def refine(
        self, selection: Selection, subject: Any, requester: str
    ) -> Selection:
        if selection.sources[0] == requester:
            return Selection.of([])
        return selection


class TakeHead(Refinement):
    """Keep the best-ranked source and drop the rest, price and gate intact."""

    name = "take-head"

    async def refine(
        self, selection: Selection, subject: Any, requester: str
    ) -> Selection:
        return selection.only(selection.sources[:1])
