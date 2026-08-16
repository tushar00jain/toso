"""Decisions taken but not yet carried out: :class:`ReservationSensor`,
:class:`RoutedPullSensor`.

A control plane that predicts has to remember what it has already promised, because
the next decision is made against a cluster that has not finished doing the last
one. Each entry stands for one thing that is going to happen, so each **expires when
that thing does**, folded from the action that says so
(:class:`~kvcache_sim.control._sensor.PrefillFinished`,
:class:`~kvcache_sim.control._sensor.FetchAnswered`) like every other write here.

A read filters as well, and that is what keeps the prediction honest rather than the
expiry: a reservation is read against the clock of the decision reading it, so one whose
prefill has landed is not offered even if no report has arrived yet -- which would count
a request twice, once as a reservation and once through the observed decode state that
had superseded it.

Grouped in one module because that is the one idea behind both, and because both are
written by the same decision: the one that took them
(:class:`~kvcache_sim.control._sensor.Committed`, folded here and by the cluster sensor,
each into its own state).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from proposed import Sensor
from proposed.dispatch import Fold

from ._action import Committed, FetchAnswered, PrefillFinished

__all__ = ["Reservation", "ReservationSensor", "RoutedPullSensor"]


class Reservation(NamedTuple):
    """A prefill the scheduler has committed to, and the decode it will join."""

    prefill_done: float
    decode_id: str
    output_tokens: int


class ReservationSensor(Sensor):
    """Prefills promised and not yet finished, oldest first.

    Held because a request routed now shares its decode instance with requests
    whose prefill has not completed yet: they are invisible to the observed decode
    state and would otherwise be predicted as absent.

    Read through the scheduler's view
    (:class:`~kvcache_sim.control._view.ReservedView`): the decision that promises a
    prefill and the host that finishes one write it, and the decode-side prediction in
    between reads it.

    **Only a run that predicts holds one**, and that is the whole of the condition:
    this sensor is composed exactly when decode occupancy is rolled forward
    (:meth:`~kvcache_sim.control.scheduler._Scheduler.attach`), so a run that does not
    predict has no reservation sensor to compose onto its dispatcher, and the same
    :class:`~kvcache_sim.control._sensor.Committed` every decision dispatches reserves
    nothing. Nothing tests a flag, because a sensor cannot see the scheduler's.
    """

    def __init__(self) -> None:
        self._held: List[Reservation] = []
        self._folds: Dict[type, Fold] = {
            Committed: self._committed,
            PrefillFinished: self._prefill_finished,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """:class:`proposed.dispatch.Reducer` -- what it folds, by action type."""
        return MappingProxyType(self._folds)

    def _committed(self, action: Committed) -> None:
        """Stand in for the accepted decision's decode until its prefill lands."""
        self.reserve(
            action.response.plan.done_time,
            action.response.decode,
            action.output_tokens,
        )

    def _prefill_finished(self, action: PrefillFinished) -> None:
        """Drop what a host has now been seen to finish.

        The clock a host reports is one it has reached, and a clock only goes forward, so
        this drops exactly what every later :meth:`pending` would filter out -- which is
        what keeps a run of any length from carrying every prefill it ever promised.
        """
        self._held = [r for r in self._held if r.prefill_done >= action.now]

    def reserve(self, prefill_done: float, decode_id: str, output_tokens: int) -> None:
        """Record a committed prefill and the decode it is bound for."""
        self._held.append(Reservation(prefill_done, decode_id, output_tokens))

    def pending(self, now: float) -> Sequence[Reservation]:
        """Reservations whose prefill has not completed as of ``now``.

        Filtered at the reading decision's own clock, not at whatever a host last
        reported: a prefill that has finished has had the decode batch it joined
        reported, so offering it here would count that request a second time.
        """
        return [r for r in self._held if r.prefill_done >= now]


class RoutedPullSensor(Sensor):
    """Pulls priced against a chosen peer, waiting for the store to ask about them.

    A pull is routed when the request is planned and fetched later, and in between
    the host that fetches will ask who should serve it. This is the note that lets
    the answer be the decision already made rather than a second,
    differently-derived one.

    Read through the scheduler's view (:class:`~kvcache_sim.control._view.RoutedView`):
    the decision that prices a pull writes it and the answer that spends one writes it
    again, while the one link that answers a fetch from it only reads.
    """

    def __init__(self) -> None:
        self._pending: List[Tuple[str, Tuple[str, ...], str]] = []
        self._folds: Dict[type, Fold] = {
            Committed: self._committed,
            FetchAnswered: self._answered,
        }

    @property
    def folds(self) -> Mapping[type, Fold]:
        """:class:`proposed.dispatch.Reducer` -- what it folds, by action type."""
        return MappingProxyType(self._folds)

    def _committed(self, action: Committed) -> None:
        """Remember the peer, for a decision that actually priced one.

        The test is on the action's own payload -- most accepted plans recompute the
        gap instead, and a plan with no source has no pull for a fetch to answer.
        """
        plan = action.response.plan
        if plan.reuse_source is not None and plan.pull_keys:
            self.route(action.response.prefill, plan.pull_keys, plan.reuse_source)

    def _answered(self, action: FetchAnswered) -> None:
        """Spend the memo that fetch was answered from.

        One answer per priced pull, so a second fetch of the same keys falls through to
        the ranking rather than reading a peer whose transfer nothing charged for. A
        fetch nothing priced spends nothing, which is why the plane dispatches this
        without having to know which link answered.
        """
        found = self._match(action.requester, action.keys)
        if found is not None:
            del self._pending[found]

    def route(self, requester: str, keys: Sequence[str], peer: str) -> None:
        """Remember that ``requester``'s pull of ``keys`` was priced against ``peer``."""
        self._pending.append((requester, tuple(keys), peer))

    def peer(self, requester: str, keys: Sequence[str]) -> Optional[str]:
        """The peer ``requester``'s pull of ``keys`` was priced against.

        ``None`` if this caller has no routed pull of exactly these keys -- "I
        decided nothing about this", not a refusal; the caller falls back to the
        ranking.

        The keys must match *exactly*. A pull is all-or-nothing, so a fetch asks for
        precisely what it was told to; a smaller set is a different pull, and
        handing it this peer would charge a locality tier chosen for someone else.
        """
        found = self._match(requester, keys)
        return None if found is None else self._pending[found][2]

    def _match(self, requester: str, keys: Sequence[str]) -> Optional[int]:
        """Where ``requester``'s pull of exactly ``keys`` sits, oldest first.

        One rule for the read and the expiry, so the memo an answer came from is the
        memo spending it drops: two pulls in flight to one instance resolve in a fixed
        order either way.
        """
        wanted = set(keys)
        for i, (inst, planned, _peer) in enumerate(self._pending):
            if inst == requester and wanted == set(planned):
                return i
        return None
