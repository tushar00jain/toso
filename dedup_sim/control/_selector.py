"""Rank live volumes and pending publication volumes together."""

from __future__ import annotations

from proposed import Selection
from proposed.selector import Annotate, Selector
from torchstore import Publication

from ._sensor import FanoutSensor

__all__ = ["Candidates", "SourceBalance", "CHAIN", "SPREAD"]

CHAIN = 10.0
SPREAD = 0.0


def _source_load(selector, _serving):
    load = selector.sensor(FanoutSensor).named()
    return lambda source: load.get(source[1], 0)


SourceBalance = Annotate(_source_load, senses=(FanoutSensor,))


class Candidates(Selector[frozenset[Publication], float]):
    """Every serving source, priced by arrival plus one transfer."""

    sensors = (FanoutSensor,)

    def __init__(self, fabric: float = CHAIN, payload_bytes: int = 1) -> None:
        self.fabric = fabric
        self.payload_bytes = payload_bytes

    def select(
        self, serving: frozenset[Publication], requester: str
    ) -> Selection[float]:
        fanout = self.sensor(FanoutSensor)
        load = fanout.named()
        priced: list[tuple[Publication, float]] = []
        for source in serving:
            pub, volume = source
            if volume == requester:
                continue
            if pub == 0:
                wait = 0.0
            else:
                if load.get(volume, 0) >= fanout.cap:
                    continue
                arrival = fanout.arrival(source)
                if arrival is None:
                    continue
                wait = arrival
            hop = self.env.read_time(volume, requester, self.payload_bytes)
            priced.append((source, wait + hop + self.fabric * hop))
        if not priced:
            return Selection.abstain()
        priced.sort(key=lambda candidate: (candidate[1], candidate[0]))
        return Selection.priced(priced)
