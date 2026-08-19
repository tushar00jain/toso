"""The fan-out tree a dedup decision reads."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterator, Mapping

from proposed import Key, LoadSensor, VolumeId
from proposed.dispatch import Action, Fold

from ._directory import Published

__all__ = ["FanoutSensor", "Retired", "Routed"]

_Region = tuple[Key, object]


@dataclass(frozen=True)
class Routed(Action):
    """``requester`` is routed through sources with exact required regions."""

    requester: VolumeId
    sources: tuple[VolumeId, ...]
    pending: tuple[VolumeId, ...] = ()
    required: tuple[tuple[VolumeId, tuple[_Region, ...]], ...] = ()
    #: What the directory was when ``pending`` was read off it
    #: (:meth:`~proposed.sensors.DirectorySensor.stamp`). While it is still that,
    #: ``pending`` describes live coverage exactly and readiness needs no further read.
    #: ``None`` says the caller does not vouch for it, and then nothing here is trusted.
    stamp: object = None


@dataclass(frozen=True)
class Retired(Action):
    """``requester`` no longer routes through ``source``."""

    requester: VolumeId
    source: VolumeId


class _Counts(Mapping):
    """``source -> how many are behind it``, over the sets naming them.

    A view rather than a rebuilt dict: a decision takes the whole mapping once
    (:data:`~proposed.selector.Balance`) and then asks about the handful of sources it
    priced, so counting on the lookup beats counting every set up front.
    """

    __slots__ = ("_behind",)

    def __init__(self, behind: Mapping[VolumeId, set[VolumeId]]) -> None:
        self._behind = behind

    def __getitem__(self, source: VolumeId) -> int:
        return len(self._behind[source])

    def __iter__(self) -> Iterator[VolumeId]:
        return iter(self._behind)

    def __len__(self) -> int:
        return len(self._behind)


class FanoutSensor(LoadSensor):
    """Who is folded in behind whom. Nobody's wait is here.

    Routes, retirements and publications move it through actions. A fold writes this
    state before its commit wakes anybody. Nothing here reads the directory's state,
    and nothing reads this: a route arrives carrying the stamp its pending flags were
    read at (:attr:`Routed.stamp`), and comparing that against the directory is left to
    the reader holding one (:class:`~dedup_sim.control._selector.Candidates`).

    Args:
        fanout_cap: readers one peer may be planned to feed -- 1 a chain, >= 2 a
            shallow tree. A ceiling and nothing else: which peer a reader takes is
            priced (:class:`~dedup_sim.control._selector.Candidates`), and this only
            says when one stops being offered at all.
    """

    def __init__(self, fanout_cap: int = 1) -> None:
        self.cap = fanout_cap
        # requester -> the sources it was routed through.
        self._route: Dict[VolumeId, tuple[VolumeId, ...]] = {}
        # requester -> source -> whether that source still owes it a region. A key is a
        # source that owed one when the route was read; the flag goes false where a
        # publication has since carried everything it owed. Both answers out of one
        # map: the flag is readiness, and the key set is what the route was read as.
        self._route_pending: Dict[VolumeId, Dict[VolumeId, bool]] = {}
        # Regions, not a Counter of them: one shared tuple per (requester, source)
        # against a dict cell per region, and the readiness probe counts them once.
        self._route_required: Dict[VolumeId, Dict[VolumeId, tuple[_Region, ...]]] = {}
        # requester -> the directory stamp its pending flags were read at.
        self._route_stamp: Dict[VolumeId, object] = {}
        # source -> the requesters behind it, carried rather than derived: every
        # decision reads the count, and re-deriving it walks the whole tree. This is
        # also the reverse edge a publication needs, so the two cannot disagree about
        # who is waiting on whom. Moved only by the members that move a route.
        self._load: Dict[VolumeId, set[VolumeId]] = {}
        self._folds: Dict[type, Fold] = {
            Published: self._published,
            Retired: self._retired,
            Routed: self._routed,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What it folds, by action type."""
        return MappingProxyType(self._folds)

    def _published(self, action: Published) -> None:
        """A reader's batch has landed: both edges of that one fact.

        The producer as a requester owes nothing more, and the producer as a *source*
        has delivered -- to whichever requesters it owed only regions named in
        :attr:`Published.landed`. A producer that published less than it promised leaves
        the legs it did not cover owed.
        """
        self._route_pending.pop(action.producer, None)
        self._route_required.pop(action.producer, None)
        self._route_stamp.pop(action.producer, None)
        landed = action.landed
        for requester in self._load.get(action.producer, ()):
            pending = self._route_pending.get(requester)
            required = self._route_required.get(requester)
            if pending is None or required is None or action.producer not in pending:
                continue
            pending[action.producer] = not (
                landed is not None
                and all(key in landed for key, _ in required[action.producer])
            )

    def planned(self, requester: str) -> tuple[VolumeId, ...]:
        """The sources ``requester`` is already folded in behind."""
        return self._route.get(requester, ())

    def routes(self) -> Mapping[VolumeId, tuple[VolumeId, ...]]:
        """``requester -> its sources``."""
        return MappingProxyType(self._route)

    def _routed(self, action: Routed) -> None:
        """Fold ``requester`` in behind ``sources``, and make it a source itself.

        Re-routing replaces every prior edge, so load and the cap stay aligned.
        """
        if len(set(action.sources)) != len(action.sources):
            raise ValueError("a route names each source once")
        required: Dict[VolumeId, tuple[_Region, ...]] = {}
        for source, regions in action.required:
            if source in required:
                raise ValueError("a route names each source requirement once")
            required[source] = regions
        if set(required) != set(action.sources) or any(
            not regions for regions in required.values()
        ):
            raise ValueError("a route requires non-empty regions for every source")
        if not set(action.pending).issubset(action.sources):
            raise ValueError("a pending source must belong to the route")
        pending = dict.fromkeys(action.pending, True)
        previous = self._route.get(action.requester, ())
        self._route_pending[action.requester] = pending
        self._route_required[action.requester] = required
        self._route_stamp[action.requester] = action.stamp
        # Unmoved route.
        if previous == action.sources:
            return
        for source in previous:
            self._drop(source, action.requester)
        self._route[action.requester] = action.sources
        for source in action.sources:
            self._load.setdefault(source, set()).add(action.requester)

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none."""
        return _Counts(self._load)

    def _retired(self, action: Retired) -> None:
        """Drop ``requester``'s route to a source nothing is coming from."""
        previous = self._route.get(action.requester, ())
        if action.source in previous:
            remaining = tuple(source for source in previous if source != action.source)
            self._route[action.requester] = remaining
            self._route_pending[action.requester].pop(action.source, None)
            self._route_required[action.requester].pop(action.source, None)
            self._drop(action.source, action.requester)

    def _drop(self, source: str, requester: VolumeId) -> None:
        """One reader fewer behind ``source``; absent at zero, as :meth:`named` says."""
        behind = self._load.get(source)
        if behind is None:
            return
        behind.discard(requester)
        if not behind:
            del self._load[source]

    def route_pending(self, producer: VolumeId) -> Mapping[VolumeId, bool]:
        """Each source that owed ``producer`` a region, and whether it still does."""
        return MappingProxyType(self._route_pending.get(producer, {}))

    def route_stamp(self, producer: VolumeId) -> object:
        """The directory :attr:`Routed.stamp` this route's pending flags were read at."""
        return self._route_stamp.get(producer)

    def route_required(
        self, producer: VolumeId
    ) -> Mapping[VolumeId, tuple[_Region, ...]]:
        """Exact regions each source must provide to ``producer``."""
        return MappingProxyType(self._route_required.get(producer, {}))
