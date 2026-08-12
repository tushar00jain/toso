"""Which of a volume's keys go when it needs room: :class:`LeastRecentlyUsed`.

A bounded volume has to choose victims, and the choice is separable from the volume:
what the volume knows is *bytes and accesses*, and what a replacement policy turns
that into is a ranking. Keeping them apart is what lets a deployment run something
other than LRU -- importance-based, size-aware, TTL -- without touching the object
that stores the data.

This is the store's own, and deliberately not a control-plane decision: recency of a
volume's data is the one thing that volume cannot be wrong about, and asking a
cluster-wide service for it would be a round trip to be told what is already local.
:meth:`proposed.policy.Policy.evict` remains the way a control plane *overrides*
this, for the decisions it can make and a volume cannot -- that a key has three other
copies, that one is about to be read, that a version is dead.

Deterministic by construction: recency is a monotonic counter, never a clock, and
ties break on the key.

Folder-private, because the volume is the only thing that picks victims and a caller
wanting different behaviour passes its own object rather than subclassing this one:
the surface is ``note`` / ``forget`` / ``victims`` / ``held``, and nothing about it is
this module's to own. (``held`` is how the volume tells a ranking that everything it
knows about is gone -- a reset -- without the ranking having to own a second verb for
it.)
"""

from __future__ import annotations

from typing import Dict, Iterable, List

__all__ = ["LeastRecentlyUsed"]


class LeastRecentlyUsed:
    """Least-recently-used ranking over the keys one volume holds.

    Told about accesses (:meth:`note`) and removals (:meth:`forget`), asked for
    victims (:meth:`victims`). It holds no data -- only what it needs to rank.
    """

    def __init__(self) -> None:
        self._used: Dict[str, int] = {}
        self._bytes: Dict[str, int] = {}
        self._clock: int = 0

    def note(self, key: str, nbytes: int | None = None) -> None:
        """``key`` was just accessed, and (on a write) is this many bytes."""
        self._clock += 1
        self._used[key] = self._clock
        if nbytes is not None:
            self._bytes[key] = nbytes

    def forget(self, key: str) -> None:
        """``key`` is gone from the volume."""
        self._used.pop(key, None)
        self._bytes.pop(key, None)

    def victims(self, need_bytes: int) -> List[str]:
        """Coldest first, enough of them to free ``need_bytes``.

        Stops as soon as the total covers the need: a volume asked for one block's
        room should lose one block, not its whole working set.
        """
        victims: List[str] = []
        freed = 0
        for key in sorted(self._used, key=lambda k: (self._used[k], k)):
            if freed >= need_bytes:
                break
            victims.append(key)
            freed += self._bytes.get(key, 0)
        return victims

    def held(self) -> Iterable[str]:
        """Every key this ranking knows about."""
        return self._used.keys()
