"""What one KV-cache decision senses, composed: :class:`KVView`.

Each class here is one read, and a :class:`~proposed.view.View` in its own right: a
view is assembled by naming the reads a decision makes, and a capability needing one
of them composes that one alone (``view.derived(ClusterView, cluster=s)``). A selector
takes the one it needs:

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

Each names its sensor with a :class:`~proposed.view.Sensed` attribute, so composing one
in is a name in a class statement and nothing else moves. They are disjoint: different
selectors read different ones and none of them touches another's, which is what makes
sensing any of them ambiently safe.

All five are *observed state* -- this plane's own sensors as much as the directory --
so whatever ranks, prices or gates senses it here instead of being handed the sensor. A
*write* does not come this way at all: every one is an action dispatched into the
plane's :class:`proposed.dispatch.Dispatcher`, whether a host reported it or the plane's
own decision did.

Pinning (:meth:`~proposed.view.View.pinned`) is the second half of the prefix idea. A
routing decision reads the prefix runs several times -- once for the candidate loop's
local matches, once per candidate when it asks the source
:class:`~proposed.selector.KeySelector` which peer would serve the gap -- and all of
them must see the *same* directory state or the decision is incoherent. The pin is on
the directory read those runs are derived from, so the scheduler names the keys once
(:meth:`~kvcache_sim.control.scheduler._Scheduler._select_prefill`) and nothing here
carries a snapshot of its own. The other four are live: they move only when a fact is
folded or a decision commits, and neither happens inside a pin.
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

    Split from the read that feeds it: :meth:`PrefixView.prefix_lengths` reads the
    directory (or serves a pinned snapshot), while
    :class:`~kvcache_sim.control._selector.LongestPrefixKeySelector` may be attached
    to a plain :class:`~proposed.view.View` and reads it itself. One definition either
    way.
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

    Derived rather than held: it reads :meth:`~proposed.view.View.locate`, so it takes
    no keyword and is never absent.
    """

    def prefix_lengths(self, block_keys: Sequence[str]) -> Dict[str, int]:
        """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

        Computed from the real ``locate_volumes`` result
        (``{key -> {volume_id -> StorageInfo}}``); the run stops at the first
        missing block, and instances holding none of the first block are omitted.
        A pure function of :meth:`~proposed.view.View.locate`, so it is coherent for
        the whole of a decision that pinned these keys without knowing it was pinned.
        """
        keys = list(block_keys)
        return prefix_lengths_of(self.locate(keys), keys)


class ClusterView(SensorView):
    """The cluster this capability decides against: :attr:`cluster`.

    Its reads are the sensor's own members (``busy_until``, ``occupancy``,
    ``predict_occupancy``), stated once where the sensor is
    (:class:`~kvcache_sim.control._sensor.ClusterSensor`).
    """

    cluster = Sensed()


class ReservedView(SensorView):
    """The prefills this plane promised and has not seen land: :attr:`reserved`.

    Written when a decision commits and when a host reports the prefill landing, read in
    between when the next decision predicts the decode batch it will meet -- against its
    own clock, so what has come true is not offered
    (:meth:`~kvcache_sim.control._sensor.ReservationSensor.pending`).
    """

    reserved = Sensed("reservation")


class RoutedView(SensorView):
    """The pulls this plane has priced against a peer: :attr:`routed`.

    Written by the plane that priced them and by the answer that spends one, read in
    between by the one link that answers a fetch from them
    (:class:`~kvcache_sim.control._selector.RoutedPull`).
    """

    routed = Sensed("routed-pull")


class KVView(PrefixView, ClusterView, ReservedView, RoutedView, LoadView):
    """Every read a KV-cache decision makes, over the run's ports.

    Each is optional and independent, so a caller wanting the prefix runs alone
    composes :class:`PrefixView` and never names the rest. C3 puts
    :class:`~proposed.view.View` once at the tail of this MRO, so the run's ports are
    taken up once however many are named.
    """
