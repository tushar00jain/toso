"""Block-location index modelling the TorchStore ``Controller`` directory.

Mirrors the controller's storage index restricted to KV-cache mode: ``block_key ->
set[instance_id]`` (which instances currently cache each block), matching
``keys_to_storage_volumes: key -> {volume_id -> StorageInfo}``. Metadata only -- no
KV bytes live here; a "block present on an instance" is a directory entry created by
:meth:`notify_put` and removed by :meth:`notify_delete` (on eviction).

As in ``dedup_sim`` we *attempt* the real controller import for faithfulness, but a
single-threaded DES cannot drive its ``@endpoint async`` Monarch-actor methods, so
we use the faithful shim below. Method names mirror the controller (``locate`` ~
``locate_volumes``, ``notify_put`` ~ ``notify_put_batch``, ``notify_delete``,
``keys``) so a later swap is mechanical.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set

from sim_common.controller_probe import HAVE_REAL  # noqa: F401

from .model import BlockKey

# ``HAVE_REAL`` (imported above) records whether the real ``torchstore.controller``
# .``Controller`` is importable -- the silenced probe lives in
# ``sim_common.controller_probe`` (see that module for why the import is silenced).
#
# We deliberately use the shim regardless of HAVE_REAL: the real controller's
# endpoints are async Monarch-actor methods needing an actor runtime. HAVE_REAL is
# recorded only for the demo banner.
USING_REAL = False


class BlockIndex:
    """Faithful in-memory shim of the controller's block-location directory."""

    def __init__(self) -> None:
        self._index: Dict[BlockKey, Set[str]] = defaultdict(set)

    def notify_put(self, block_key: BlockKey, instance_id: str) -> None:
        """Record that ``instance_id`` now caches ``block_key`` (idempotent)."""
        self._index[block_key].add(instance_id)

    def notify_delete(self, block_key: BlockKey, instance_id: str) -> None:
        """Drop one ``block -> instance`` mapping (block evicted on that instance)."""
        if block_key in self._index:
            self._index[block_key].discard(instance_id)
            if not self._index[block_key]:
                del self._index[block_key]

    def locate(self, block_keys: Iterable[BlockKey]) -> Dict[BlockKey, Set[str]]:
        """Return ``{block_key -> set[instance_id]}`` for the given keys.

        Mirrors ``Controller.locate_volumes``; a copy is returned so callers cannot
        mutate the index.
        """
        out: Dict[BlockKey, Set[str]] = {}
        for k in block_keys:
            if k in self._index:
                out[k] = set(self._index[k])
        return out

    def instances_with_prefix(self, block_keys: List[BlockKey]) -> Dict[str, int]:
        """Return ``{instance_id -> matched prefix length (in blocks)}``.

        The cache-aware scheduler's core query: for each instance, how many *leading* blocks of
        ``block_keys`` it holds contiguously. Instances with no match are omitted.
        """
        loc = self.locate(block_keys)
        # Per instance, count the leading run present.
        counts: Dict[str, int] = {}
        # Candidate instances = any holding the first block (others match 0).
        for inst in sorted(loc.get(block_keys[0], set())) if block_keys else []:
            n = 0
            for k in block_keys:
                if inst in self._index.get(k, ()):
                    n += 1
                else:
                    break
            counts[inst] = n
        return counts

    def keys(self) -> List[BlockKey]:
        """Return all cached block keys (mirrors ``Controller.keys``)."""
        return list(self._index.keys())
