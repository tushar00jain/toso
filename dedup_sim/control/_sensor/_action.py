"""What this capability dispatches: :class:`Asked`.

The plane dispatches this one to itself. The landed put folded beside it is
:class:`proposed.dispatch.Stored`, declared by the store's own surface because any
reader could report it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from proposed import Key, VolumeId
from proposed.dispatch import Action

__all__ = ["Asked"]


@dataclass(frozen=True)
class Asked(Action):
    """``requester`` is about to read ``keys`` through, so it owes those puts.

    The plane dispatches this before consulting the ranking that could offer the
    requester as a peer, so a peer is offered only once it has promised -- which is
    what bounds the wait on it (:func:`~dedup_sim.control._answer.committed`).
    """

    requester: VolumeId
    keys: Tuple[Key, ...]
