"""Storage-volume index modelling the TorchStore ``Controller`` directory.

Per the spec we first *attempt* to import the real controller. It imports
cleanly (with Monarch), but its endpoints (`locate_volumes`, `notify_put_batch`,
`keys`) are ``@endpoint async`` Monarch-actor methods that require an actor
runtime to drive, and they operate on torchstore-internal ``Request`` /
``TensorSlice`` / ``Trie`` types rather than plain 1-D integer regions.

A single-threaded discrete-event sim cannot drive those actor endpoints without
spawning Monarch actors (explicitly out of scope: "Do not pull in Monarch /
launch actors"). So we use the faithful **shim** below. It mirrors the real
controller's storage index semantics -- ``key -> {volume_id -> set[Region]}``,
matching ``keys_to_storage_volumes: key -> {volume_id -> StorageInfo}`` -- with
matching method names (``locate`` ~ ``locate_volumes``, ``notify_put`` ~
``notify_put_batch``, ``keys``) so a later swap to the real controller is
mechanical.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from .model import Region

# Attempt the real import for documentation/faithfulness (see module docstring).
# The import (via Monarch) is noisy on stderr; silence it -- we do not use the
# real controller anyway, we only record whether it is importable.
import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

try:  # pragma: no cover - depends on Monarch being installed
    with contextlib.redirect_stderr(io.StringIO()), \
            contextlib.redirect_stdout(io.StringIO()):
        with open(os.devnull, "w") as _devnull:
            _saved = os.dup(2)
            os.dup2(_devnull.fileno(), 2)
            try:
                from torchstore.controller import Controller as _RealController  # noqa: F401,E501
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
    HAVE_REAL = True
except Exception:  # pragma: no cover
    HAVE_REAL = False

# We deliberately use the shim regardless of HAVE_REAL: the real controller's
# endpoints are async Monarch-actor methods that need an actor runtime, which a
# plain single-thread sim cannot provide. HAVE_REAL is recorded for reference.
USING_REAL = False


class StoreIndex:
    """Faithful in-memory shim of the controller's storage index.

    Maps ``key -> {volume_id -> set[Region]}``. Metadata only -- no bytes ever
    live here; a "region present on a volume" is a directory entry created by
    :meth:`notify_put`.
    """

    def __init__(self) -> None:
        self._index: Dict[str, Dict[str, Set[Region]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def notify_put(self, key: str, volume_id: str, region: Region) -> None:
        """Record that ``volume_id`` now holds ``region`` for ``key``.

        Mirrors ``Controller.notify_put_batch`` / ``_notify_put``: idempotent,
        additive union into the per-key, per-volume region set.
        """
        self._index[key][volume_id].add(region)

    def locate(self, key: str) -> Dict[str, Set[Region]]:
        """Return ``{volume_id -> set[Region]}`` for ``key`` (empty if absent).

        Mirrors ``Controller.locate_volumes`` (single key). A copy is returned
        so callers cannot mutate the index.
        """
        if key not in self._index:
            return {}
        return {vol: set(regs) for vol, regs in self._index[key].items()}

    def keys(self) -> List[str]:
        """Return all known keys (mirrors ``Controller.keys``)."""
        return list(self._index.keys())

    def notify_delete(self, key: str, volume_id: str) -> None:
        """Drop one ``key -> volume`` mapping (mirrors ``notify_delete``)."""
        if key in self._index and volume_id in self._index[key]:
            del self._index[key][volume_id]
            if not self._index[key]:
                del self._index[key]
