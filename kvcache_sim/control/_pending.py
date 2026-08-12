"""Decisions taken but not yet carried out: :class:`Reservations`, :class:`RoutedPulls`.

A coordinator that predicts has to remember what it has already promised, because
the next decision is made against a cluster that has not finished doing the last
one. Two such records exist here, and both are *self-expiring*: an entry is only
worth keeping until the thing it describes happens.

Which is the whole reason they are objects. Expiry that lives in the method that
*adds* an entry runs at the wrong moment -- a routing decision reads the record
before it writes to it, so every read saw a record last cleaned one decision ago,
carrying entries whose event had since occurred. Prediction then counted a request
twice: once as a reservation it still believed in, and once through the observed
decode state that had already superseded it. Here, expiry belongs to the record and
runs when it is *read*, which is the only moment its answer has to be right.

Neither holds anything the coordinator could look up elsewhere: a reservation is a
prediction nobody else has made yet, and a routed pull is a decision the store has
not asked about yet. That is what distinguishes them from a mirror -- nothing
external can contradict them, only fulfil them.

Folder-private: this is the shape of one control plane's own bookkeeping, not a
surface. What ``proposed`` would gain from it, if anything, is the idea rather than
the code.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

__all__ = ["Reservation", "Reservations", "RoutedPulls"]


class Reservation(NamedTuple):
    """A prefill this coordinator has committed to, and the decode it will join."""

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
        that request a second time. Expiry runs here, on the read, because that is
        where being stale would be wrong.
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

        ``None`` if this caller has no routed pull of exactly these keys, which the
        caller should read as "I decided nothing about this" -- not as a refusal;
        the ranking then answers, which is the right answer for a pull nobody
        priced.

        Matched oldest first, so two pulls in flight to one instance resolve in a
        fixed order, and consumed on match: a pull is answered once, and an entry
        that stayed would be claimed by somebody else's fetch later.

        The keys must match *exactly*. They used to only have to be covered by what
        was planned, because a fetch would drop blocks that had been evicted since
        routing and ask for the rest -- but a pull is all-or-nothing now, so a
        fetch asks for precisely what it was told to. A smaller set no longer means
        "this pull, minus what vanished"; it means a different pull, and handing it
        this peer would charge it a locality tier chosen for someone else.
        """
        wanted = set(keys)
        for i, (inst, planned, peer) in enumerate(self._pending):
            if inst == requester and wanted == set(planned):
                del self._pending[i]
                return peer
        return None
