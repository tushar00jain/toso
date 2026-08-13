"""The one derived directory read the KV-cache scheduler needs.

:class:`~proposed.view.View` stops at "who holds this key". A KV-cache scheduler
asks one step further on: *how many leading blocks of this prompt does each
instance hold contiguously?* -- because a cache is only useful as a contiguous
prefix. That is a KV-cache notion, not a store notion, so it is a subclass here
rather than a field on the base view.

:meth:`KVView.pinned` is the second half of the same idea. A routing decision reads
the prefix runs several times -- once for the candidate loop's local matches, once
per candidate when it asks the source :class:`~proposed.policy.KeySelector` which peer
would serve the gap -- and all of them must see the *same* directory state or the
decision is incoherent. Pinning also means the directory is walked once per
request, not once per read.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    AbstractSet, Dict, Iterator, List, Optional, Sequence, Tuple,
)

from proposed import View

__all__ = ["prefix_lengths_of", "KVView"]


def _longest_prefix_run(block_keys: Sequence[str], present: AbstractSet[str]) -> int:
    """Return how many leading blocks of ``block_keys`` are in ``present``.

    The prefix match stops at the first missing block (a cache is only useful as a
    contiguous prefix), matching block-by-block prefix comparison.
    """
    n = 0
    for k in block_keys:
        if k in present:
            n += 1
        else:
            break
    return n


def prefix_lengths_of(
    located: Dict[str, Dict[str, object]], block_keys: Sequence[str]
) -> Dict[str, int]:
    """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

    Split from the read that feeds it: :meth:`KVView.prefix_lengths` reads the
    directory (or serves a pinned snapshot), while
    :class:`~kvcache_sim.control._source.LongestPrefixPolicy` may be attached to a
    plain :class:`~proposed.view.View` and reads it itself. One definition either
    way.
    """
    keys = list(block_keys)
    if not keys:
        return {}
    counts: Dict[str, int] = {}
    for inst in sorted(located.get(keys[0], {})):
        held = {key for key in keys if inst in located.get(key, {})}
        counts[inst] = _longest_prefix_run(keys, held)
    return counts


class KVView(View):
    """A :class:`~proposed.view.View` plus per-instance prefix-run lengths."""

    #: The keys one decision pinned and the runs it read for them, while
    #: :meth:`pinned` holds; ``None`` outside such a decision.
    _snapshot: Optional[Tuple[List[str], Dict[str, int]]] = None

    def prefix_lengths(self, block_keys: Sequence[str]) -> Dict[str, int]:
        """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

        Computed from the real ``locate_volumes`` result
        (``{key -> {volume_id -> StorageInfo}}``); the run stops at the first
        missing block, and instances holding none of the first block are omitted.
        Served from the pinned snapshot while one is held.
        """
        keys = list(block_keys)
        if self._snapshot is not None:
            pinned_keys, counts = self._snapshot
            assert keys == pinned_keys, (
                "a pinned view answers for the keys it was pinned to; one decision "
                "reads one snapshot"
            )
            return counts
        return prefix_lengths_of(self.locate(keys), keys)

    @contextmanager
    def pinned(self, block_keys: Sequence[str]) -> Iterator[None]:
        """Read the directory once, and serve that snapshot for the block.

        Scoped state on the view rather than a snapshot object passed around,
        because every selector a decision consults senses through this same view
        (:meth:`~proposed.policy.Selector.attach`) and would otherwise read past the
        snapshot into the live directory.

        Sound because one decision cannot be interleaved with another: the directory
        read underneath it is a plain synchronous method
        (:meth:`~proposed.deployment.Controller.locate_raw`), so there is no
        suspension point between the pin and its release. Should one ever appear,
        the assertions fire -- here on a second decision entering, in
        :meth:`prefix_lengths` on a read of other keys arriving inside one.
        """
        assert self._snapshot is None, "a decision already holds this view's snapshot"
        keys = list(block_keys)
        self._snapshot = (keys, prefix_lengths_of(self.locate(keys), keys))
        try:
            yield
        finally:
            self._snapshot = None
