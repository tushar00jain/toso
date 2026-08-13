"""Decisions taken but not yet carried out: :class:`Reservations`, :class:`RoutedPulls`.

A control plane that predicts has to remember what it has already promised, because
the next decision is made against a cluster that has not finished doing the last
one. Both records here are *self-expiring*, and expiry runs on the **read** rather
than on the write: a routing decision reads before it writes, so sweeping on write
would serve entries whose event had since occurred -- and prediction would count a
request twice, once as a reservation and once through the observed decode state
that had superseded it.

Folder-private: one control plane's own bookkeeping, not a surface.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

__all__ = ["Reservation", "Reservations", "RoutedPulls"]


class Reservation(NamedTuple):
    """A prefill the scheduler has committed to, and the decode it will join."""

    prefill_done: float
    decode_id: str
    output_tokens: int


class Reservations:
    """Prefills promised and not yet finished, oldest first.

    Held because a request routed now shares its decode instance with requests
    whose prefill has not completed yet: they are invisible to the observed decode
    state and would otherwise be predicted as absent.
    """

    def __init__(self) -> None:
        self._held: List[Reservation] = []

    def reserve(self, prefill_done: float, decode_id: str, output_tokens: int) -> None:
        """Record a committed prefill and the decode it is bound for."""
        self._held.append(Reservation(prefill_done, decode_id, output_tokens))

    def pending(self, now: float) -> Sequence[Reservation]:
        """Reservations whose prefill has not completed as of ``now``.

        Drops the rest as it goes: once a prefill has finished, the data plane has
        reported the decode batch it joined, so keeping the prediction would count
        that request a second time.
        """
        self._held = [r for r in self._held if r.prefill_done >= now]
        return self._held


class RoutedPulls:
    """Pulls priced against a chosen peer, waiting for the store to ask about them.

    A pull is routed when the request is planned and fetched later, and in between
    the directory will ask who should serve it. This is the note that lets the
    answer be the decision already made rather than a second, differently-derived
    one.
    """

    def __init__(self) -> None:
        self._pending: List[Tuple[str, Tuple[str, ...], str]] = []

    def route(self, requester: str, keys: Sequence[str], peer: str) -> None:
        """Remember that ``requester``'s pull of ``keys`` was priced against ``peer``."""
        self._pending.append((requester, tuple(keys), peer))

    def claim(self, requester: str, keys: Sequence[str]) -> Optional[str]:
        """The peer ``requester``'s pull of ``keys`` was priced against, consumed.

        ``None`` if this caller has no routed pull of exactly these keys -- "I
        decided nothing about this", not a refusal; the caller falls back to the
        ranking.

        Matched oldest first, so two pulls in flight to one instance resolve in a
        fixed order, and consumed on match, so a later fetch cannot claim it.

        The keys must match *exactly*. A pull is all-or-nothing, so a fetch asks for
        precisely what it was told to; a smaller set is a different pull, and
        handing it this peer would charge a locality tier chosen for someone else.
        """
        wanted = set(keys)
        for i, (inst, planned, peer) in enumerate(self._pending):
            if inst == requester and wanted == set(planned):
                del self._pending[i]
                return peer
        return None
