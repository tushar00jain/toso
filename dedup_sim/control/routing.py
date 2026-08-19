"""Publication-aware dedup routing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Tuple, Unpack

from proposed import (
    ControlPlane,
    DecisionLog,
    Dispatcher,
    Environment,
    Selection,
    Sensor,
    endpoint,
)
from proposed.selector import Balance, Fold, Ordered, pipe, Selector, WithFold

from ._selector import Candidates, CHAIN, Serving, SPREAD
from ._sensor import Asked, DedupDirectorySensor, FanoutSensor, Pub, Published, Routed

__all__ = ["Dedup", "ReadPlan"]


@dataclass(frozen=True)
class ReadPlan:
    """A ranked source preference and its publication."""

    sources: tuple[str, ...]
    publication: Pub
    ready: Optional[Callable[[], Awaitable[None]]] = None

    @property
    def head(self) -> Optional[str]:
        return self.sources[0] if self.sources else None

    async def settled(self) -> "ReadPlan":
        if self.ready is not None:
            await self.ready()
            return replace(self, ready=None)
        return self


def _soonest(dims: Tuple[float, int]) -> float:
    score, queued = dims
    return score * (1 + queued)


def _gate_publications(
    requests: Sequence[Any],
    sources: tuple[str, ...],
    live: Mapping[str, set[str]],
    pending: Mapping[str, set[Pub]],
) -> frozenset[Pub]:
    rank = {source: index for index, source in enumerate(sources)}
    sliced = {request.key for request in requests if request.tensor_slice is not None}
    gates: set[Pub] = set()
    for key in live.keys() | pending.keys():
        key_pubs = pending.get(key, set())
        if key in sliced:
            gates.update(pub for pub in key_pubs if pub[0] in rank)
            continue
        live_ranks = [rank[volume] for volume in live.get(key, set()) if volume in rank]
        pub_ranks = [rank[pub[0]] for pub in key_pubs if pub[0] in rank]
        if not pub_ranks:
            continue
        head = min((*live_ranks, *pub_ranks))
        if head not in live_ranks:
            gates.update(pub for pub in key_pubs if rank.get(pub[0]) == head)
    return frozenset(gates)


class Dedup(ControlPlane):
    """Rank live and pending sources for one read-through batch."""

    sensors = (DedupDirectorySensor, FanoutSensor)

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
        self._chain: Optional[Selector[Serving, Unpack[Tuple[Any, ...]]]] = None

    def attach(
        self,
        environment: Environment,
        sensors: Optional[Mapping[type, Sensor]] = None,
    ) -> "Dedup":
        available = dict(sensors or {})
        fanout = FanoutSensor(fanout_cap=self._cap)
        available[type(fanout)] = fanout
        super().attach(environment, available)
        self.dispatcher = Dispatcher()
        for observed in self._sensed.values():
            self.dispatcher.compose(observed)
        fold: Optional[Fold[float, int]] = _soonest if self._spread else None
        self._chain = pipe(
            Balance(Candidates(SPREAD if self._spread else CHAIN)),
            WithFold(fold),
            Ordered,
        ).attach(environment, self._sensed)
        return self

    @endpoint
    async def sources(self, requests: Sequence[Any], requester: str) -> ReadPlan:
        return await (await self._decide(requests, requester)).settled()

    async def _decide(self, requests: Sequence[Any], requester: str) -> ReadPlan:
        batch = tuple(requests)
        directory = self.sensor(DedupDirectorySensor)
        publication = directory.declare(requester, batch)
        self.dispatcher.dispatch_sync(Asked(publication))
        live, pending = directory.serving_union(batch)
        serving = Serving(
            frozenset().union(*live.values()),
            frozenset().union(*pending.values()),
        )
        ranking = self._chain.select(serving, requester)
        return self._committed(
            requester, publication, batch, live, pending, ranking
        )

    def _committed(
        self,
        requester: str,
        publication: Pub,
        requests: Sequence[Any],
        live: Mapping[str, set[str]],
        pending: Mapping[str, set[Pub]],
        ranking: Selection,
    ) -> ReadPlan:
        assert ranking.sources is not None, "the dedup chain names an explicit order"
        sources = ranking.sources
        gate_pubs = _gate_publications(requests, sources, live, pending)
        fanout = self.sensor(FanoutSensor)

        def source_arrival(source: str) -> float:
            arrivals = [
                fanout.arrival(pub)
                for pub in gate_pubs
                if pub[0] == source and fanout.arrival(pub) is not None
            ]
            return max(arrivals, default=0.0)

        arrival = max(
            (
                source_arrival(source)
                + self.env.read_time(source, requester, 1)
                for source in sources[:1]
            ),
            default=0.0,
        )
        if self._trace is not None:
            self._trace.record(
                self.env.now(), "route", f"{requester} <- {','.join(sources)}"
            )
        self.dispatcher.dispatch_sync(
            Routed(publication, sources, gate_pubs, arrival)
        )
        directory = self.sensor(DedupDirectorySensor)
        gate = self.dispatcher.gate(
            lambda: not any(directory.is_in_flight(pub) for pub in gate_pubs),
            (Published(*pub) for pub in gate_pubs),
        )
        return ReadPlan(sources, publication, gate)
