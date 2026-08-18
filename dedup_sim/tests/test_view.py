"""Dedup selectors resolve their domain sensors through the shared map."""

from __future__ import annotations

import pytest

from dedup_sim.control._selector import Candidates
from dedup_sim.control._sensor import Asked, FanoutSensor, Routed
from proposed import DirectorySensor, Dispatcher, Endpoint, Environment
from proposed.selector import Balance, FirstMatch, Ordered


class _Holds:
    def __init__(self, volume: str) -> None:
        self.volume = volume

    def locate_raw(self, keys, missing_ok: bool = False):
        return {key: {self.volume: None} for key in keys}


_TOPOLOGY = {
    v: Endpoint(id=v, host=v, node=v) for v in ("origin", "r0", "r1")
}


class _Profile:
    def read_time(self, src: Endpoint, dst: Endpoint, nbytes: int) -> float:
        if src.id == dst.id:
            return 0.0
        return 10.0 if src.id == "origin" else 1.0


_PROFILE = _Profile()


def test_a_fanout_nobody_supplied_raises():
    directory = DirectorySensor(_Holds("origin"))
    with pytest.raises(RuntimeError, match="FanoutSensor"):
        Candidates().attach(
            Environment(_TOPOLOGY, _PROFILE), {DirectorySensor: directory}
        )


def test_the_ranking_keeps_fanout_under_reranking():
    fanout = FanoutSensor(fanout_cap=1)
    dispatcher = Dispatcher()
    dispatcher.compose(fanout)
    dispatcher.dispatch_sync(Routed("r0", "origin"))
    dispatcher.dispatch_sync(Asked("r0", ("K",)))
    ranking = Candidates()
    chain = Ordered(FirstMatch([Balance(ranking)]))
    chain.attach(
        Environment(_TOPOLOGY, _PROFILE),
        {
            DirectorySensor: DirectorySensor(_Holds("origin")),
            FanoutSensor: fanout,
        },
    )

    assert ranking.sensor(FanoutSensor) is fanout
    assert chain.select(["K"], "r1").sources == ("r0", "origin")
