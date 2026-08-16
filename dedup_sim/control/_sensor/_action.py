"""What this capability dispatches: :class:`Asked`.

A :class:`proposed.dispatch.Action`, folded by every sensor here that folds its type and
committed once (:class:`proposed.dispatch.Dispatcher`). This one the plane dispatches to
itself; the landed put it folds beside it is :class:`proposed.dispatch.Stored`, which the
store's own surface declares because any reader could report it. Nothing here says which
sensor folds what, because an action does not know who folds it.

A frozen value, so it crosses a process boundary unchanged.
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

    Asking is the promise, and the plane dispatches it before consulting the ranking that
    could offer this requester as a peer: that is what makes a requester offered only
    after it has promised, and so what bounds the wait on it
    (:func:`~dedup_sim.control._answer.committed`).
    """

    requester: VolumeId
    keys: Tuple[Key, ...]
