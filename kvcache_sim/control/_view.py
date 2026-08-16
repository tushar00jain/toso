"""What one KV-cache decision senses, composed: :class:`KVView`.

One class per read a KV-cache decision makes, and a selector takes the one it needs:

* :class:`PrefixView`: how many leading blocks of a prompt an instance holds
  contiguously. The base view stops at "who holds this key", and a cache is only
  useful as a contiguous prefix -- a KV-cache notion, not a store notion;
* :class:`ClusterView`: the predicted prefill queues and observed decode batches
  (:class:`~kvcache_sim.control._sensor.ClusterSensor`), which is what ranks, prices
  and gates a candidate host;
* :class:`ReservedView`: the prefills this plane promised and has not seen land
  (:class:`~kvcache_sim.control._sensor.ReservationSensor`), read by the decode-side
  prediction of a run that rolls occupancy forward, and composed in only by such a run;
* :class:`RoutedView`: the pulls a decision priced against a peer
  (:class:`~kvcache_sim.control._sensor.RoutedPullSensor`), read by the fetch chain's
  head link alone (:class:`~kvcache_sim.control._selector.RoutedPull`);
and one more the store's own surface declares, because what it holds is a *volume's*
load rather than anything KV-shaped: :class:`~proposed.view.LoadView`, carrying this
plane's :class:`~kvcache_sim.control._sensor.SourceLoad`.

The five are disjoint -- no view touches another's sensor -- which is what makes
sensing any of them ambiently safe. All five are *observed state*, so nothing that
ranks, prices or gates is handed a sensor, and a **write** never comes this way: it is
an action dispatched, whether a host reported it or this plane's own decision did.

Pinning is the second half of the prefix idea. A routing decision reads the prefix runs
several times -- once for the candidate loop's
local matches, once per candidate asking which peer would serve the gap -- and all of
them must see the *same* directory state or the decision is incoherent. The pin is on
the directory read those runs derive from, so the scheduler names the keys once
(:meth:`~kvcache_sim.control.scheduler._Scheduler.decide`) and nothing here carries a
snapshot of its own. The other four are live: they move only when a fact is folded or a
decision commits, and neither happens inside a pin.
"""

from __future__ import annotations

from typing import AbstractSet, Dict, Sequence

from proposed import LoadView, Sensed, SensorView, View

__all__ = [
    "prefix_lengths_of",
    "PrefixView",
    "ClusterView",
    "ReservedView",
    "RoutedView",
    "KVView",
]


def _longest_prefix_run(block_keys: Sequence[str], present: AbstractSet[str]) -> int:
    """Return how many leading blocks of ``block_keys`` are in ``present``.

    The prefix match stops at the first missing block (a cache is only useful as a
    contiguous prefix), matching block-by-block prefix comparison.
    """
    n = 0
    for k in block_keys:
        if k in present:
            n += 1
        else:
            break
    return n


def prefix_lengths_of(
    located: Dict[str, Dict[str, object]], block_keys: Sequence[str]
) -> Dict[str, int]:
    """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

    Split from the read that feeds it, so a caller holding a plain
    :class:`~proposed.view.View` and one holding a :class:`PrefixView` share the
    definition.
    """
    keys = list(block_keys)
    if not keys:
        return {}
    counts: Dict[str, int] = {}
    for inst in sorted(located.get(keys[0], {})):
        held = {key for key in keys if inst in located.get(key, {})}
        counts[inst] = _longest_prefix_run(keys, held)
    return counts


class PrefixView(View):
    """Prefix runs, off this view's own directory.

    Derived from the directory read rather than held, so it is never absent.
    """

    def prefix_lengths(self, block_keys: Sequence[str]) -> Dict[str, int]:
        """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

        Off the real ``locate_volumes`` result (``{key -> {volume_id -> StorageInfo}}``);
        the run stops at the first missing block, and instances holding none of the
        first block are omitted. A pure function of the directory read, so it is
        coherent for the whole of a decision that pinned these keys without knowing it
        was pinned.
        """
        keys = list(block_keys)
        return prefix_lengths_of(self.locate(keys), keys)


class ClusterView(SensorView):
    """The cluster this capability decides against: :attr:`cluster`.

    ``busy_until``, ``occupancy``, ``predict_occupancy``, described where the sensor is
    (:class:`~kvcache_sim.control._sensor.ClusterSensor`).
    """

    cluster = Sensed()


class ReservedView(SensorView):
    """The prefills this plane promised and has not seen land: :attr:`reserved`.

    Read against the reading decision's own clock, so what has come true is not offered
    (:meth:`~kvcache_sim.control._sensor.ReservationSensor.pending`).
    """

    reserved = Sensed("reservation")


class RoutedView(SensorView):
    """The pulls this plane has priced against a peer: :attr:`routed`.

    Read by the one link that answers a fetch from them
    (:class:`~kvcache_sim.control._selector.RoutedPull`).
    """

    routed = Sensed("routed-pull")


class KVView(PrefixView, ClusterView, ReservedView, RoutedView, LoadView):
    """Every read a KV-cache decision makes, over the run's ports."""
