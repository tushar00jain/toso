"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

One member, and it is everything a reader asks: *which volumes serve this key for me,
and when are they usable* (:meth:`Dedup.sources`). That the put landed is not a second
question and not a report to this plane: it is one action a reader commits
(:class:`proposed.dispatch.Stored`), folded by this plane's own state
(:attr:`Dedup.dispatcher`).

The ranking is :mod:`dedup_sim.control._selector`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Unpack

from proposed import (
    ControlPlane, DecisionLog, DirectorySensor, Dispatcher, Environment, Key,
    endpoint, Selection, Sensor,
)
from proposed.selector import (
    Balance, FirstMatch, Fold, NaiveKeySelector, Ordered, pipe, Selector, WithFold,
)

from ._selector import Candidates, CHAIN, SPREAD
from ._sensor import Asked, FanoutSensor

__all__ = ["Dedup"]


def _soonest(dims: Tuple[float, int]) -> float:
    """Spread's fold: with the fabric free the score is one read, so a reader queued
    behind ``queued`` others waits that many times over."""
    score, queued = dims
    return score * (1 + queued)


class Dedup(ControlPlane):
    """Dedup's whole control plane: one member over one source chain.

    Args:
        fanout_cap: peers one source may be planned to feed -- 1 a chain, >= 2 a
            shallow tree, 1x fabric either way.
        spread: price the fabric at nothing
            (:data:`~dedup_sim.control._selector.SPREAD`) and fold the queue at a
            source into the score instead (:func:`_soonest`). Off by default, and a
            trade: two readers of one key go to two replicas rather than chaining the
            second behind the first, which costs a second hop off the holders and buys
            back the wallclock of that chain hop.
        trace: optional :class:`~proposed.selector.DecisionLog` for each routing
            decision. Records only; no metric turns on it.
    """

    sensors = (DirectorySensor, FanoutSensor)

    def __init__(
        self,
        *,
        fanout_cap: int = 1,
        spread: bool = False,
        trace: Optional[DecisionLog] = None,
    ) -> None:
        self._cap = fanout_cap
        self._spread = spread
        self._trace = trace
        self.dispatcher: Optional[Dispatcher] = None
        self._chain: Optional[Selector[Sequence[Key], Unpack[Tuple[Any, ...]]]] = None

    def attach(
        self,
        environment: Environment,
        sensors: Optional[Mapping[type, Sensor]] = None,
    ) -> "Dedup":
        """Build this plane's sensor and attach the chain that reads it.

        The sensor is built here, never accepted from a caller: two planes sharing one
        would each answer for the other's decisions -- a requester handed a peer the
        other plane planned, then waiting on a put only the other plane hears about.

        Sorted rather than cut to the winner: a reader reads its preference down, so a
        source ranked behind the head still serves the read if the head evicted the key
        before the reader got there.
        """
        sensor = FanoutSensor(fanout_cap=self._cap)
        available = dict(sensors or {})
        available[type(sensor)] = sensor
        super().attach(environment, available)
        self.dispatcher = Dispatcher()
        for observed in self._sensed.values():
            self.dispatcher.compose(observed)
        # Which volumes serve a read: every holder of the key and every peer already
        # planned to hold it, priced together in seconds, so which one wins is
        # arithmetic off the score rather than an order the caller has to know.
        # ``None`` names no arity to read off it, so the key this leaves alone is said
        # here: the score Candidates prices, then the readers Balance counts.
        fold: Optional[Fold[float, int]] = _soonest if self._spread else None
        self._chain = pipe(
            FirstMatch([
                pipe(
                    Balance(Candidates(SPREAD if self._spread else CHAIN)),
                    WithFold(fold),
                ),
                # Tail: an unroutable ask gets the directory's own answer, not a hole.
                NaiveKeySelector(),
            ]),
            Ordered,
        ).attach(environment, self._sensed)
        return self

    # -- what a reader asks -------------------------------------------------- #
    @endpoint
    async def sources(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[Unpack[Tuple[Any, ...]]]:
        """Which volumes serve ``keys`` for ``requester``, once they are usable."""
        # The wait is spent here, not handed back: a caller that read before these
        # sources held the key would go to a volume with nothing to serve.
        return await (await self._decide(keys, requester)).settled()

    async def _decide(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[Unpack[Tuple[Any, ...]]]:
        """The whole decision with the gate unspent, awaitable without parking.

        The chain keys each source ``(score, queued)``: the seconds
        :class:`~dedup_sim.control._selector.Candidates` priced it at, then the readers
        :data:`~proposed.selector.Balance` found already routed to it. Compared as they
        stand, the fabric decides and the queue only settles a tie the score cannot:
        queueing behind a peer is already in that peer's own wait, so charging it again
        would price one delay twice. The tie-break keeps two replicas of one key
        alternating rather than falling back to id order.
        """
        keys = list(keys)
        # Asking is what makes this requester a peer, so its debt is dispatched before
        # the ranking is consulted -- that debt is what bounds the wait on whichever
        # source is named. Dispatched without suspending, so the debt and the decision
        # priced against it are one turn.
        self.dispatcher.dispatch_sync(Asked(requester, tuple(keys)))
        ranking = self._chain.select(keys, requester)
        return self._committed(keys, requester, ranking)

    def _committed(
        self, keys: Sequence[Key], requester: str, ranking: Selection
    ) -> Selection:
        """``ranking``, routed to its head and gated until that head is usable.

        A source that owes the keys is gated on its pending registration. A source
        that neither owes nor holds them has evicted them and is retired.
        """
        # No suspension splits the sensor read-modify-write sequence.
        source = ranking.head
        if source is None:
            return ranking
        fanout = self.sensor(FanoutSensor)
        if fanout.planned(requester) != source:
            fanout.route(requester, source)
            if self._trace is not None:
                self._trace.record(self.env.now(), "route", f"{requester} <- {source}")
        facts = [(source, key) for key in keys]
        if fanout.owes(facts):
            # Pending publication.
            return replace(
                ranking,
                ready=self.dispatcher.gate(
                    lambda: len(self._registered(facts)) == len(facts)
                ),
            )
        if len(self._registered(facts)) == len(facts):
            # Already holds every key.
            return ranking
        # Evicted after publication.
        fanout.retire(requester, source)
        if self._trace is not None:
            self._trace.record(self.env.now(), "retire", f"{source} holds nothing")
        return Selection()

    def _registered(
        self, facts: Sequence[Tuple[str, Key]]
    ) -> List[Tuple[str, Key]]:
        """Facts the directory holds now; gates must re-read this live."""
        directory = self.sensor(DirectorySensor)
        by_key = directory.holders([key for _volume, key in facts], live=True)
        return [
            (volume, key)
            for volume, key in facts
            if volume in by_key[key]
        ]
