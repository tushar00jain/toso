"""The fan-out tree and pending puts a dedup decision reads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

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
    """``requester`` is routed through ``source``."""

    requester: VolumeId
    source: VolumeId


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
        # requester -> the source it was routed to, decided once and then reused.
        self._route: Dict[str, str] = {}
        # source -> how many are behind it, carried rather than counted on demand:
        # every decision reads it, and re-deriving it walks the whole tree. Moved only
        # by the two members that move a route, so the two cannot disagree.
        self._load: Counter = Counter()
        # producer -> its one in-flight read-through batch, indexed by key.
        self._promised: Dict[VolumeId, Dict[Key, Any]] = {}
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
            raise ValueError(
                f"{action.requester} already has an in-flight publication"
            )
        promised = {request.key: request for request in action.requests}
        if len(promised) != len(action.requests):
            raise ValueError("a publication names each key once")
        self._promised[action.requester] = promised

    def _published(self, action: Published) -> None:
        """A reader's batch has landed: settle the debt it owed."""
        self._promised.pop(action.producer, None)

    def planned(self, requester: str) -> Optional[str]:
        """The source ``requester`` is already folded in behind, if any."""
        return self._route.get(requester)

    def routes(self) -> Mapping[str, str]:
        """``requester -> its source``: the tree, and the edges a decision walks."""
        return MappingProxyType(self._route)

    def _routed(self, action: Routed) -> None:
        """Fold ``requester`` in behind ``source``, and make it a source itself.

        One requester, one edge: re-routing replaces it rather than adding a second, so
        no bookkeeping drift can hand out more slots than the cap.
        """
        previous = self._route.get(action.requester)
        # Unmoved route.
        if previous == action.source:
            return
        if previous is not None:
            self._drop(previous)
        self._route[action.requester] = action.source
        self._load[action.source] += 1

    def named(self) -> Mapping[str, int]:
        """``source -> requesters currently routed to it``. Absent means none."""
        return MappingProxyType(self._load)

    def _retired(self, action: Retired) -> None:
        """Drop ``requester``'s route to a source nothing is coming from."""
        if self._route.get(action.requester) == action.source:
            del self._route[action.requester]
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

    def covers(
        self, producer: VolumeId, requested: Mapping[Key, Any]
    ) -> bool:
        """Does ``producer``'s promised batch cover ``requested``?"""
        promised = self._promised.get(producer)
        return promised is not None and all(
            key in promised and _request_covers(promised[key], request)
            for key, request in requested.items()
        )


def _slice_covers(available: Optional[Any], requested: Optional[Any]) -> bool:
    """Whether one stored/requested tensor region contains another."""
    if available is None:
        return True
    if requested is None:
        return (
            all(offset == 0 for offset in available.offsets)
            and tuple(available.local_shape) == tuple(available.global_shape)
        )
    if tuple(available.global_shape) != tuple(requested.global_shape):
        return False
    return all(
        have_offset <= need_offset
        and need_offset + need_size <= have_offset + have_size
        for have_offset, have_size, need_offset, need_size in zip(
            available.offsets,
            available.local_shape,
            requested.offsets,
            requested.local_shape,
        )
    )


def _request_covers(available: Any, requested: Any) -> bool:
    """Whether one promised request contains another request for the same key."""
    return available.key == requested.key and _slice_covers(
        available.tensor_slice, requested.tensor_slice
    )


def _storage_covers(info: object, requested: Any) -> bool:
    """Whether one directory entry can serve ``requested`` alone."""
    tensor_slices = getattr(info, "tensor_slices", None)
    if tensor_slices is None:
        return True
    return any(
        _slice_covers(tensor_slice, requested.tensor_slice)
        for tensor_slice in tensor_slices
    )
