"""The fan-out tree and pending puts a dedup decision reads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

from proposed import Key, LoadSensor, VolumeId
from proposed.dispatch import Action, Fold

__all__ = ["Asked", "FanoutSensor", "Published", "Retired", "Routed"]


@dataclass(frozen=True)
class Asked(Action):
    """``requester`` is about to read ``requests`` through, so it owes those puts."""

    requester: VolumeId
    requests: tuple[Any, ...]

    def __hash__(self) -> int:
        return hash((type(self), self.requester))


@dataclass(frozen=True)
class Published(Action):
    """``producer``'s promised read-through batch has landed."""

    producer: VolumeId


@dataclass(frozen=True)
class Routed(Action):
    """``requester`` is routed through ``sources``."""

    requester: VolumeId
    sources: tuple[VolumeId, ...]
    by_key: tuple[tuple[Key, tuple[VolumeId, ...]], ...] = ()
    pending: tuple[VolumeId, ...] = ()
    required: tuple[tuple[VolumeId, tuple[Any, ...]], ...] = ()


@dataclass(frozen=True)
class Retired(Action):
    """``requester`` no longer routes through ``source``."""

    requester: VolumeId
    source: VolumeId


class FanoutSensor(LoadSensor):
    """Who is folded in behind whom and which puts are owed. Nobody's wait is here.

    Asks, routes, retirements and publications move it through actions. A fold writes
    this state before its commit wakes anybody. Nothing here reads the directory's
    state, and nothing reads this.

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
        self._route_by_key: Dict[VolumeId, Dict[Key, tuple[VolumeId, ...]]] = {}
        self._route_pending: Dict[VolumeId, set[VolumeId]] = {}
        self._route_required: Dict[VolumeId, Dict[VolumeId, Counter]] = {}
        # source -> how many are behind it, carried rather than counted on demand:
        # every decision reads it, and re-deriving it walks the whole tree. Moved only
        # by the two members that move a route, so the two cannot disagree.
        self._load: Counter = Counter()
        # producer -> its one in-flight read-through batch, indexed by key.
        self._promised: Dict[VolumeId, Dict[Key, Any]] = {}
        self._promised_by_key: Dict[Key, Dict[VolumeId, Any]] = {}
        self._folds: Dict[type, Fold] = {
            Asked: self._asked,
            Published: self._published,
            Retired: self._retired,
            Routed: self._routed,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What it folds, by action type."""
        return MappingProxyType(self._folds)

    def _asked(self, action: Asked) -> None:
        """A reader is about to read this batch through: it owes those puts from now."""
        if action.requester in self._promised:
            raise ValueError(f"{action.requester} already has an in-flight publication")
        promised = {request.key: request for request in action.requests}
        if len(promised) != len(action.requests):
            raise ValueError("a publication names each key once")
        self._promised[action.requester] = promised
        for key, request in promised.items():
            self._promised_by_key.setdefault(key, {})[action.requester] = request

    def _published(self, action: Published) -> None:
        """A reader's batch has landed: settle the debt it owed."""
        promised = self._promised.pop(action.producer, {})
        for key in promised:
            producers = self._promised_by_key[key]
            del producers[action.producer]
            if not producers:
                del self._promised_by_key[key]
        self._route_by_key.pop(action.producer, None)
        self._route_pending.pop(action.producer, None)
        self._route_required.pop(action.producer, None)

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
        previous = self._route.get(action.requester, ())
        by_key = (
            dict(action.by_key)
            if action.by_key
            else {
                key: action.sources for key in self._promised.get(action.requester, {})
            }
        )
        # Unmoved route.
        if previous == action.sources:
            self._route_by_key[action.requester] = by_key
            self._route_pending[action.requester] = set(action.pending)
            self._route_required[action.requester] = {
                source: Counter(regions) for source, regions in action.required
            }
            return
        for source in previous:
            self._drop(source)
        self._route[action.requester] = action.sources
        self._route_by_key[action.requester] = by_key
        self._route_pending[action.requester] = set(action.pending)
        self._route_required[action.requester] = {
            source: Counter(regions) for source, regions in action.required
        }
        for source in action.sources:
            self._load[source] += 1

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none."""
        return MappingProxyType(self._load)

    def _retired(self, action: Retired) -> None:
        """Drop ``requester``'s route to a source nothing is coming from."""
        previous = self._route.get(action.requester, ())
        if action.source in previous:
            remaining = tuple(source for source in previous if source != action.source)
            self._route[action.requester] = remaining
            self._route_by_key[action.requester] = {
                key: tuple(source for source in sources if source != action.source)
                for key, sources in self._route_by_key.get(action.requester, {}).items()
            }
            self._route_pending[action.requester].discard(action.source)
            self._route_required[action.requester].pop(action.source, None)
            self._drop(action.source)

    def _drop(self, source: str) -> None:
        """One reader fewer behind ``source``; absent at zero, as :meth:`named` says."""
        if self._load[source] > 1:
            self._load[source] -= 1
        else:
            del self._load[source]

    def plan(self, producer: VolumeId) -> Mapping[Key, Any]:
        """``producer``'s in-flight requests, by key."""
        return MappingProxyType(self._promised[producer])

    def promised(self, keys: Sequence[Key]) -> Dict[Key, Mapping[VolumeId, Any]]:
        """Pending directory entries for ``keys``."""
        return {
            key: MappingProxyType(self._promised_by_key.get(key, {})) for key in keys
        }

    def in_flight(self) -> set[VolumeId]:
        """Producers with a publication outstanding."""
        return set(self._promised)

    def route_plan(self, producer: VolumeId) -> Mapping[Key, tuple[VolumeId, ...]]:
        """Per-key sources feeding ``producer``'s publication."""
        return MappingProxyType(self._route_by_key.get(producer, {}))

    def route_pending(self, producer: VolumeId) -> set[VolumeId]:
        """Sources ``producer`` is waiting to publish."""
        return set(self._route_pending.get(producer, set()))

    def route_required(self, producer: VolumeId) -> Mapping[VolumeId, Counter]:
        """Exact regions each source must provide to ``producer``."""
        return MappingProxyType(self._route_required.get(producer, {}))
