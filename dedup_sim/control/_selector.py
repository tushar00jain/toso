"""Rank live volumes and pending publication volumes together."""

from __future__ import annotations

from dataclasses import dataclass
from proposed import Selection, VolumeId
from proposed.selector import Selector

from ._sensor import DedupDirectorySensor, FanoutSensor, Pub

__all__ = ["Candidates", "Serving", "CHAIN", "SPREAD"]

CHAIN = 10.0
SPREAD = 0.0


@dataclass(frozen=True)
class Serving:
    """The store candidates read for one decision."""

    live: frozenset[VolumeId]
    pending: frozenset[Pub]


class Candidates(Selector[Serving, float]):
    """Every serving volume, priced by arrival plus one transfer."""

    sensors = (DedupDirectorySensor, FanoutSensor)

    def __init__(self, fabric: float = CHAIN, payload_bytes: int = 1) -> None:
        self.fabric = fabric
        self.payload_bytes = payload_bytes

    def select(self, serving: Serving, requester: str) -> Selection[float]:
        fanout = self.sensor(FanoutSensor)
        pending: dict[VolumeId, list[float]] = {}
        for pub in serving.pending:
            arrival = fanout.arrival(pub)
            if arrival is not None:
                pending.setdefault(pub[0], []).append(arrival)

        priced: list[tuple[VolumeId, float]] = []
        candidates = serving.live | pending.keys()
        for volume in candidates:
            if volume == requester:
                continue
            live = volume in serving.live
            arrivals = pending.get(volume, ())
            if not live and not arrivals:
                continue
            if not live and fanout.named().get(volume, 0) >= fanout.cap:
                continue
            wait = 0.0 if live else max(arrivals)
            hop = self.env.read_time(volume, requester, self.payload_bytes)
            priced.append((volume, wait + hop + self.fabric * hop))
        if not priced:
            return Selection.abstain()
        return Selection.priced(priced)
