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
from proposed.selector import Fold, Ordered, pipe, Selector, WithFold
from torchstore import Publication

from ._selector import Candidates, CHAIN, SourceBalance, SPREAD
from ._sensor import Asked, DedupDirectorySensor, FanoutSensor, Published, Routed

__all__ = ["Dedup", "ReadPlan"]


@dataclass(frozen=True)
class ReadPlan:
    """A ranked source preference and its publication."""

    sources: tuple[str, ...]
    publication: Publication
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
    chosen: Sequence[Publication],
) -> frozenset[Publication]:
    return frozenset(publication for publication in chosen if publication[0] != 0)


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
        self._chain: Optional[
            Selector[frozenset[Publication], Unpack[Tuple[Any, ...]]]
        ] = None

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
            SourceBalance(Candidates(SPREAD if self._spread else CHAIN)),
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
        serving = directory.serving_union(batch)
        ranking = self._chain.select(serving, requester)
        return self._committed(requester, publication, batch, ranking)

    def _committed(
        self,
        requester: str,
        publication: Publication,
        requests: Sequence[Any],
        ranking: Selection,
    ) -> ReadPlan:
        assert ranking.sources is not None, "the dedup chain names an explicit order"
        directory = self.sensor(DedupDirectorySensor)
        chosen = list(directory.greedy_cover(requests, ranking.sources))
        sources = tuple(dict.fromkeys(volume for _pub, volume in chosen))
        fanout = self.sensor(FanoutSensor)
        gate_pubs = _gate_publications(chosen)
        contributions: list[float] = []
        for pub, volume in chosen:
            arrival = 0.0 if pub == 0 else fanout.arrival((pub, volume))
            assert arrival is not None, "a ranked pending source has an arrival"
            contributions.append(arrival + self.env.read_time(volume, requester, 1))
        arrival = max(contributions, default=0.0)
        if self._trace is not None:
            self._trace.record(
                self.env.now(), "route", f"{requester} <- {','.join(sources)}"
            )
        self.dispatcher.dispatch_sync(
            Routed(publication, sources, gate_pubs, arrival)
        )
        gate = self.dispatcher.gate(
            lambda: not any(directory.is_in_flight(pub) for pub in gate_pubs),
            (Published(pub) for pub in gate_pubs),
        )
        return ReadPlan(sources, publication, gate)
