"""Promised directory entries, kept in the map the directory answers reads from.

A control plane that routes readers onto each other has to know what the directory
will say *after* the in-flight writes land, not only what it says now. That is a
projection of the directory, and the cheap place to keep it is the directory's own
``{key -> {volume_id -> StorageInfo}}`` store: a promise is then an ordinary entry,
found by the same lookup, and no join runs per request.

Two things make that safe:

* a promise is a :class:`Promised`, a ``StorageInfo`` subclass, so ``isinstance`` is
  the entire filter and an ordinary read subtracts it before answering, keeping
  whatever live entry it covers (:meth:`Projecting.live_map`);
* the live view of a key is cached and invalidated only by a **live** mutation of
  that key. Adding or clearing a promise cannot change which volumes hold the key,
  so a burst of ``G`` promises over ``K`` keys leaves the ordinary read ``O(K)``.

Mixed into whatever owns the store, which names it as :attr:`Projecting.entries`:
:class:`realsim.seams.controller_service.ControllerService` names the real
``Controller``'s ``keys_to_storage_volumes``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Set

from torchstore.controller import StorageInfo

__all__ = ["Projecting", "Promised"]

_Entries = MutableMapping[str, Dict[str, StorageInfo]]


@dataclass
class Promised(StorageInfo):
    """What ``owner`` will hold for this key once its write lands.

    A directory entry in every respect except that no ordinary read may see it: the
    volume holds nothing yet, so answering a ``get`` with it would send a reader to
    an empty volume, and counting it toward a DTensor's shards would call a
    half-written tensor complete.
    """

    #: The volume that promised it, and the only volume allowed to clear or replace
    #: it. Redundant with the slot it sits in, and kept so an entry read out of the
    #: map alone still says whose promise it is.
    owner: str = ""

    #: What the slot held live when the promise landed on it, or ``None`` where the
    #: volume held nothing. A volume that holds part of a key and promises the rest
    #: has one entry covering both, so :attr:`tensor_slices` is the union and this is
    #: the half an ordinary read may answer with.
    shadowed: StorageInfo | None = None


class Projecting:
    """Promise bookkeeping over a directory's own key-to-volume store."""

    def __init__(self) -> None:
        #: owner -> the keys it has outstanding promises on.
        self._promises: Dict[str, Set[str]] = {}
        #: key -> how many promises sit on it; absent means none, which is the fast
        #: path that keeps an unprojected read identical to upstream's.
        self._promised_on: Counter = Counter()
        #: key -> its live-only volume map, for keys carrying a promise. Dropped by
        #: :meth:`unpromise` on every live mutation of the key and by nothing else,
        #: because making or clearing a promise cannot change who holds the key.
        self._live: Dict[str, Dict[str, StorageInfo]] = {}

    @property
    def entries(self) -> _Entries:
        """The ``{key -> {volume_id -> StorageInfo}}`` store reads are answered from."""
        raise NotImplementedError

    # -- writing a projection ------------------------------------------------ #
    def project(self, owner: str, key: str, info: StorageInfo) -> None:
        """Record that ``owner`` will hold ``info`` for ``key``.

        A volume already holding part of ``key`` keeps that entry underneath the
        promise (:attr:`Promised.shadowed`) and the promise covers the union, so a
        reader routed onto it is offered both halves while an ordinary read still
        sees only what is really there. A promise cannot change a key's object type;
        one that disagrees is refused, as ``StorageInfo.update`` refuses it.
        """
        entries = self.entries
        if key not in entries:
            entries[key] = {}
        volumes = entries[key]
        held = volumes.get(owner)
        if isinstance(held, Promised):
            shadowed, counted = held.shadowed, True
        elif held is not None:
            if held.object_type != info.object_type:
                return
            shadowed, counted = held, False
        else:
            shadowed, counted = None, False
        if not counted:
            self._promised_on[key] += 1
            self._promises.setdefault(owner, set()).add(key)
        # Where nothing is shadowed this shares the caller's slice set rather than
        # copying it. Nothing mutates one: a put on this slot goes through
        # unpromise() first, so StorageInfo.update never reaches it.
        slices = (
            info.tensor_slices
            if shadowed is None
            else shadowed.tensor_slices | info.tensor_slices
        )
        volumes[owner] = Promised(info.object_type, slices, owner, shadowed)

    def clear_projections(self, owner: str) -> None:
        """Drop every promise ``owner`` still has outstanding.

        Idempotent, and the only thing that bounds how long a promise lives: a
        producer that publishes fewer keys than it promised leaves the rest here.
        """
        for key in tuple(self._promises.get(owner, ())):
            self._forget(key, owner)

    def projected_owners(self) -> Mapping[str, Set[str]]:
        """``owner -> its outstanding promised keys``."""
        return self._promises

    # -- reading around one ------------------------------------------------- #
    def live_map(self, key: str, volumes: Dict[str, StorageInfo]) -> Dict[str, Any]:
        """``volumes`` without its promises: what an ordinary read may answer with.

        Returns ``volumes`` itself where nothing is promised on ``key``, so a
        directory nobody is projecting onto reads exactly as it does upstream.
        """
        if not self._promised_on.get(key):
            return volumes
        cached = self._live.get(key)
        if cached is None:
            cached = {}
            for volume, info in volumes.items():
                if not isinstance(info, Promised):
                    cached[volume] = info
                elif info.shadowed is not None:
                    cached[volume] = info.shadowed
            self._live[key] = cached
        return cached

    def unpromise(self, key: str, owner: str) -> None:
        """A real put or delete by ``owner`` is about to land on ``key``.

        Two things, and both have to happen before the mutation runs. The slot is
        cleared of ``owner``'s promise, because a put must **replace** a promise
        rather than union with it (``StorageInfo.update`` unions) or the entry stays
        a promise forever and its owner is never a holder. And the key's live view is
        dropped, because a live mutation is the only thing that stales it.
        """
        self._forget(key, owner)
        self._live.pop(key, None)

    def _forget(self, key: str, owner: str) -> None:
        """Take ``owner``'s promise on ``key`` out of the store and the bookkeeping."""
        keys = self._promises.get(owner)
        if keys is None or key not in keys:
            return
        keys.discard(key)
        if not keys:
            del self._promises[owner]
        entries = self.entries
        if key in entries:
            volumes = entries[key]
            promise = volumes.get(owner)
            if isinstance(promise, Promised):
                # Restored, not dropped: the promise covered live slices too, and
                # deleting the slot would take real data with it.
                if promise.shadowed is not None:
                    volumes[owner] = promise.shadowed
                else:
                    del volumes[owner]
            if not volumes:
                del entries[key]
        count = self._promised_on[key] - 1
        if count:
            self._promised_on[key] = count
        else:
            del self._promised_on[key]
            self._live.pop(key, None)
