"""Selector sensor declarations and coherent KV directory reads."""

from __future__ import annotations

import pytest

from kvcache_sim.control._selector import (
    LocalOnly, LongestPrefixKeySelector, RoutedPull,
)
from kvcache_sim.control._sensor import ClusterSensor, RoutedPullSensor
from proposed import DirectorySensor, Environment
from realsim.adapters.real_controller import RealControllerAdapter
from torchstore.transport import Request


def test_a_selector_requires_each_declared_sensor():
    with pytest.raises(RuntimeError, match="RoutedPullSensor"):
        RoutedPull().attach(Environment({}))


def test_a_selector_retains_only_declared_sensor_types():
    routed = RoutedPullSensor()
    cluster = ClusterSensor(["s0"])
    selector = RoutedPull().attach(
        Environment({}), {RoutedPullSensor: routed, ClusterSensor: cluster}
    )
    assert selector.sensor(RoutedPullSensor) is routed
    with pytest.raises(RuntimeError, match="did not declare"):
        selector.sensor(ClusterSensor)


def test_a_ranking_that_declares_nothing_needs_no_sensor():
    selector = LocalOnly().attach(Environment({}))
    assert selector.select(["k"], "r").sources == ()


def test_directory_pin_is_shared_by_every_selector_read(monkeypatch):
    service = RealControllerAdapter().service
    service.notify_put_batch(
        [Request.from_any(key, None).meta_only() for key in ("k0", "k1")],
        "s0",
        pending=False,
    )
    walks = 0
    locate = service._locate

    def counted(*args, **kwargs):
        nonlocal walks
        walks += 1
        return locate(*args, **kwargs)

    monkeypatch.setattr(service, "_locate", counted)
    directory = DirectorySensor(service)
    selector = LongestPrefixKeySelector().attach(
        Environment({}), {DirectorySensor: directory}
    )
    keys = ["k0", "k1"]
    with directory.pinned(keys):
        assert walks == 1
        assert selector.select(keys, "r").key == {"s0": (-2,)}
        assert walks == 1
    selector.select(keys, "r")
    assert walks == 2
