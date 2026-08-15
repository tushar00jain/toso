"""What a view composed of the wrong sensors does: raise, and where.

A decision senses through one view, and each sensor on it is composed in by the
plane that owns it (:meth:`kvcache_sim.control.scheduler._Scheduler.attach`). Two
ways to get that wrong, and both have to be loud, because the quiet versions are a
run that looks healthy: a sensor nobody composed answering as if it were empty, and
a keyword nobody claims being dropped on the floor.

And what a selector's declared views buy (:attr:`~proposed.selector.Selector.sensors`):
it senses those and nothing else, off the same pin the decision consulting it holds.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest kvcache_sim/tests/test_view.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from kvcache_sim.control._selector import (
    ByBatch, LongestPrefixKeySelector, RoutedPull,
)
from kvcache_sim.control._sensor import (
    ClusterSensor, ReservationSensor, RoutedPullSensor,
)
from kvcache_sim.control._view import ClusterView, KVView, PrefixView, RoutedView
from proposed.view import View


def _view(**sensors):
    """A view over no directory: nothing here reads one."""
    return View(None, {}).derived(KVView, **sensors)


class _Walks:
    """A directory in which ``s0`` holds every key, counting the walks it is asked for."""

    def __init__(self) -> None:
        self.walks = 0

    def locate_raw(self, keys, missing_ok: bool = False):
        self.walks += 1
        return {key: {"s0": None} for key in keys}


@pytest.mark.parametrize(
    "attribute, said",
    [("cluster", "cluster"), ("reserved", "reservation"), ("routed", "routed-pull")],
)
def test_a_sensor_nobody_composed_raises_and_names_itself(attribute, said):
    """An empty answer would be a wrong one: every host idle, nothing promised.

    Named in the message because a view carries several and the reader is a decision
    that asked for one of them.
    """
    with pytest.raises(RuntimeError, match=said):
        getattr(_view(), attribute)


def test_a_keyword_no_sensor_claims_fails_where_the_view_is_built():
    """Each view pops its own and passes the rest up; the base takes none.

    So a misspelling cannot compose a view that silently senses less than it was
    asked for -- it is refused at the one place a view is assembled.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _view(clustre=ClusterSensor(["s0"]))


def test_a_view_may_compose_one_sensor_and_only_that_one():
    """The point of composing rather than subclassing one view that has everything.

    A caller that reads the cluster and nothing else says so, and gets a view whose
    other reads do not exist rather than one that answers them emptily.
    """
    one = View(None, {}).derived(ClusterView, cluster=ClusterSensor(["s0"]))
    assert one.cluster.busy_until == {"s0": 0.0}
    assert not isinstance(one, PrefixView)
    assert not hasattr(one, "prefix_lengths")


def test_a_view_composing_no_sensor_is_the_base_view():
    """Nothing is added by asking for nothing, so a plane that senses the directory
    alone needs no view of its own."""
    assert type(View(None, {}).derived(View)) is View


def test_a_selector_senses_the_views_it_declared_and_nothing_else():
    """The header is the list, and what it leaves out is unreachable, not empty.

    A ranking cannot read past what it declared by accident: the attribute is not on
    the view it was attached to at all.
    """
    view = _view(
        cluster=ClusterSensor(["s0"]), reserved=None, routed=RoutedPullSensor(),
    )
    assert RoutedPull.sensors == (RoutedView,)
    sensed = view.subset(*RoutedPull.sensors)
    assert sensed.routed is view.routed              # the run's one sensor
    with pytest.raises(AttributeError):
        sensed.cluster                               # not declared, so not there
    assert not hasattr(sensed, "prefix_lengths")


def test_a_ranking_that_declares_nothing_senses_nothing():
    """``()`` is a claim as much as a view is, and it holds the same way.

    Attached to a view composing no sensor at all, so any sensor read on it raises and
    this ranking still answers -- which is what makes "reads nothing" a fact.
    """
    bare = View(None, {})
    assert ByBatch.sensors == ()
    assert ByBatch().attach(bare.subset(*ByBatch.sensors)).view is bare
    with pytest.raises(AttributeError):
        bare.cluster
    keyed = asyncio.run(ByBatch().attach(bare).select([("s0", 3), ("s1", 1)], "r"))
    assert keyed.sort().sources == ("s1", "s0")      # the ranking keys, the fold orders


def test_a_subset_composing_a_sensor_this_view_never_carried_raises():
    """Declaring a view is not a way to reach past the one a caller was given."""
    cluster_only = View(None, {}).derived(ClusterView, cluster=ClusterSensor(["s0"]))
    with pytest.raises(RuntimeError, match="routed-pull"):
        cluster_only.subset(RoutedView).routed


def test_a_subset_reads_the_pin_and_walks_the_directory_no_further():
    """One decision, one directory -- through every subset of the view it pinned.

    The pin is the root view's cell, shared by reference, so a ranking sensing through
    the view it declared is inside the decision's snapshot rather than beside it.
    """
    directory = _Walks()
    view = View(directory, {}).derived(KVView)
    sensed = view.subset(*LongestPrefixKeySelector.sensors)
    keys = ["k0", "k1"]
    with view.pinned(keys):
        assert directory.walks == 1                  # the decision's one walk
        assert sensed.prefix_lengths(keys) == {"s0": 2}
        assert directory.walks == 1                  # and the subset added none
    assert sensed.prefix_lengths(keys) == {"s0": 2}
    assert directory.walks == 2                      # live again once released


def test_the_sensors_a_run_composes_answer_once_composed():
    """The whole point of the raise above: composed, they answer for themselves."""
    view = _view(
        cluster=ClusterSensor(["s0"]),
        reserved=ReservationSensor(),
        routed=None,
    )
    assert view.cluster.busy_until == {"s0": 0.0}
    assert list(view.reserved.pending(now=0.0)) == []
    with pytest.raises(RuntimeError, match="routed-pull"):
        view.routed
