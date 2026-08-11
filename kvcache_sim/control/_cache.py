"""Per-instance KV cache with eviction (K5).

Each serving instance owns one bounded cache of KV blocks. This is the capability
the weight-sync path never needed: the dedup design pins blocks within a version
window, but an inference cache is unbounded and long-lived, so it must evict.

The default policy is **LRU**, which exploits temporal proximity in request
utilization (see ../docs/torchstore_kvcache_design.md). ``capacity`` is measured in blocks; ``None``
means unbounded. Recency is a monotonic counter bumped on every access, so eviction
is fully deterministic (no wall-clock).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

__all__ = ["LRUCache"]


class LRUCache:
    """A bounded, LRU-evicting set of block keys held by one instance.

    This is the per-instance eviction bookkeeping (which keys to drop, and their
    recency); the authoritative record of *presence* is the real directory, which
    the caller keeps in sync by publishing admitted keys and dropping evicted ones.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        self.capacity = capacity            # in blocks; None => unbounded
        self._recency: Dict[str, int] = {}  # key -> last-access clock
        self._clock: int = 0

    def __contains__(self, key: str) -> bool:
        return key in self._recency

    def held(self) -> Set[str]:
        """Return the set of block keys currently cached."""
        return set(self._recency)

    def __len__(self) -> int:
        return len(self._recency)

    def touch(self, keys: List[str]) -> None:
        """Mark ``keys`` as most-recently used (a cache hit on a prefix)."""
        for k in keys:
            if k in self._recency:
                self._clock += 1
                self._recency[k] = self._clock

    def admit(self, keys: List[str]) -> List[str]:
        """Insert/refresh ``keys``; evict the coldest until within capacity.

        Returns the list of evicted keys (so the caller can drop them from the
        directory). Eviction ties break on the block key for determinism.
        """
        for k in keys:
            self._clock += 1
            self._recency[k] = self._clock

        evicted: List[str] = []
        if self.capacity is not None:
            while len(self._recency) > self.capacity:
                # coldest = lowest recency; tie-break on key for determinism.
                victim = min(self._recency, key=lambda k: (self._recency[k], k))
                del self._recency[victim]
                evicted.append(victim)
        return evicted
