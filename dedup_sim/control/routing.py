"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

One member, and it is everything a reader asks: *which volumes serve this batch for me,
and when are they usable* (:meth:`Dedup.sources`). That the put landed is not a second
question and not a report to this plane: it is one action a reader commits
(:class:`dedup_sim.control._sensor.Published`), folded by this plane's own state
(:attr:`Dedup.dispatcher`).

The ranking is :mod:`dedup_sim.control._selector`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Tuple, Unpack

from proposed import (
    ControlPlane,
    DecisionLog,
    DirectorySensor,
    Dispatcher,
    Environment,
    Key,
    VolumeId,
    endpoint,
    Selection,
    Sensor,
)
from proposed.selector import (
    Balance,
    FirstMatch,
    Fold,
    NaiveKeySelector,
    Ordered,
    pipe,
    Selector,
    WithFold,
)
from .._planning import plan_fetch, requirements_met
from ._selector import Candidates, CHAIN, SPREAD
from ._sensor import Asked, FanoutSensor, Published, Routed

__all__ = ["Dedup", "ReadPlan"]


@dataclass(frozen=True)
class ReadPlan:
    """Per-key source preferences and their readiness."""

    by_key: Mapping[Key, tuple[VolumeId, ...]]
    sources: tuple[VolumeId, ...]
    ready: Optional[Callable[[], Awaitable[None]]] = None

    @property
    def head(self) -> Optional[VolumeId]:
        return self.sources[0] if self.sources else None

    async def settled(self) -> "ReadPlan":
        if self.ready is not None:
            await self.ready()
            return replace(self, ready=None)
        return self


def _soonest(dims: Tuple[float, int]) -> float:
    """Spread's fold: with the fabric free the score is one read, so a reader queued
    behind ``queued`` others waits that many times over."""
    score, queued = dims
    return score * (1 + queued)


class Dedup(ControlPlane):
    """Dedup's whole control plane: one member over source chains.

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
            FirstMatch(
                [
                    pipe(
                        Balance(Candidates(SPREAD if self._spread else CHAIN)),
                        WithFold(fold),
                    ),
                    # Tail: an unroutable ask gets the directory's own answer, not a hole.
                    NaiveKeySelector(),
                ]
            ),
            Ordered,
        ).attach(environment, self._sensed)
        return self

    # -- what a reader asks -------------------------------------------------- #
    @endpoint
    async def sources(self, requests: Sequence[Any], requester: str) -> ReadPlan:
        """Which volumes serve ``requests`` for ``requester``, once they are usable."""
        # The wait is spent here, not handed back: a caller that read before these
        # sources held the key would go to a volume with nothing to serve.
        return await (await self._decide(requests, requester)).settled()

    async def _decide(self, requests: Sequence[Any], requester: str) -> ReadPlan:
        """The whole decision with the gate unspent, awaitable without parking.

        The chain keys each source ``(score, queued)``: the seconds
        :class:`~dedup_sim.control._selector.Candidates` priced it at, then the readers
        :data:`~proposed.selector.Balance` found already routed to it. Compared as they
        stand, the fabric decides and the queue only settles a tie the score cannot:
        queueing behind a peer is already in that peer's own wait, so charging it again
        would price one delay twice. The tie-break keeps two replicas of one key
        alternating rather than falling back to id order.
        """
        requests = list(requests)
        keys = [request.key for request in requests]
        # Asking is what makes this requester a peer, so its debt is dispatched before
        # the ranking is consulted -- that debt is what bounds the wait on whichever
        # source is named. Dispatched without suspending, so the debt and the decision
        # priced against it are one turn.
        self.dispatcher.dispatch_sync(Asked(requester, tuple(requests)))
        directory = self.sensor(DirectorySensor)
        with directory.pinned(keys):
            ranking = self._chain.select(keys, requester)
            return self._committed(requester, ranking)

    def _committed(self, requester: str, ranking: Selection) -> ReadPlan:
        """Narrow ``ranking`` with TorchStore's planner and gate pending sources."""
        # No suspension splits the sensor read-modify-write sequence.
        fanout = self.sensor(FanoutSensor)
        directory = self.sensor(DirectorySensor)
        requested = tuple(fanout.plan(requester).values())
        live = directory.locate([request.key for request in requested])
        order = ranking.sources
        if order is None:
            order = tuple(
                dict.fromkeys(
                    source for by_source in live.values() for source in by_source
                )
            )
        allowed = set(order)
        promised = {
            key: {
                producer: request
                for producer, request in producers.items()
                if producer != requester and producer in allowed
            }
            for key, producers in fanout.promised(
                [request.key for request in requested]
            ).items()
        }
        fetch = plan_fetch(requested, live, promised, order)
        sources = fetch.sources
        previous = fanout.planned(requester)
        self.dispatcher.dispatch_sync(
            Routed(
                requester,
                sources,
                tuple(fetch.by_key.items()),
                tuple(source for source in sources if source in fetch.pending),
                tuple(
                    (source, tuple(fetch.required[source].elements()))
                    for source in sources
                ),
            )
        )
        if previous != sources:
            if self._trace is not None:
                self._trace.record(
                    self.env.now(), "route", f"{requester} <- {','.join(sources)}"
                )
        required = {source: fetch.required[source] for source in fetch.pending}
        gate = self.dispatcher.gate(
            lambda: requirements_met(
                requested,
                directory.locate_live([request.key for request in requested]),
                required,
            ),
            (Published(source) for source in fetch.pending),
        )
        return ReadPlan(fetch.by_key, sources, gate)
