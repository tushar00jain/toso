"""The one derived directory read the KV-cache scheduler needs.

:class:`~proposed.view.View` stops at "who holds this key". A KV-cache scheduler
asks something one step further on: *how many leading blocks of this prompt does
each instance hold contiguously?* -- because a cache is only useful as a
contiguous prefix. That is a KV-cache notion (a block-key chain), not a store
notion, so it is a subclass here rather than a field on the base view.

:class:`PinnedKVView` is the second half of the same idea. A routing decision
reads the prefix runs several times -- once for the candidate loop's local
matches and once per candidate when it asks the source
:class:`~proposed.policy.Policy` which peer would serve the gap -- and every one of
those reads must see the *same* directory state or the decision is incoherent.
Pinning the snapshot for the duration of one decision makes that explicit
(and means the directory is walked once per request, not once per read).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from proposed.view import View

__all__ = ["KVView", "PinnedKVView"]


class KVView(View):
    """A :class:`~proposed.view.View` plus per-instance prefix-run lengths."""

    async def prefix_lengths(self, block_keys: Sequence[str]) -> Dict[str, int]:
        """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

        Computed from the real ``locate_volumes`` result
        (``{key -> {volume_id -> StorageInfo}}``); the run stops at the first
        missing block, and instances holding none of the first block are omitted.
        """
        keys = list(block_keys)
        if not keys:
            return {}
        located = await self.locate(keys)
        counts: Dict[str, int] = {}
        for inst in sorted(located.get(keys[0], {})):
            n = 0
            for key in keys:
                if inst in located.get(key, {}):
                    n += 1
                else:
                    break
            counts[inst] = n
        return counts

    def pin(self, block_keys: Sequence[str]) -> "PinnedKVView":
        """A view of this one directory snapshot, for one routing decision."""
        return PinnedKVView(self, block_keys)


class PinnedKVView(KVView):
    """A :class:`KVView` whose prefix runs are read once and then reused.

    Everything else (``locate``, topology, the clock) delegates to the view it
    was pinned from.
    """

    def __init__(self, base: KVView, block_keys: Sequence[str]) -> None:
        super().__init__(base.directory, base.topology)
        self._base = base
        self._keys: List[str] = list(block_keys)
        self._counts: Optional[Dict[str, int]] = None

    async def prefix_lengths(
        self, block_keys: Optional[Sequence[str]] = None
    ) -> Dict[str, int]:
        """The pinned snapshot's prefix runs (the argument is the pinned keys)."""
        assert block_keys is None or list(block_keys) == self._keys, (
            "a pinned view answers for the keys it was pinned to; pin a new one "
            "for a different request"
        )
        if self._counts is None:
            self._counts = await self._base.prefix_lengths(self._keys)
        return self._counts
