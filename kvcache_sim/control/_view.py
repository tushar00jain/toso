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
  head link alone (:class:`~kvcache_sim.control._selector.RoutedPull`).

Each names its sensor with a :class:`~proposed.view.Sensed` attribute, so composing one
in is a name in a class statement and nothing else moves. They are disjoint: different
selectors read different ones and none of them touches another's, which is what makes
sensing any of them ambiently safe.

All four are *observed state* -- this plane's own sensors as much as the directory --
so whatever ranks, prices or gates senses it here instead of being handed the sensor,
and the plane reports its own decision into them the same way. What does not come this
way is a host's fact: the run puts a service in front of the cluster sensor for those.

:meth:`PrefixView.pinned` is the second half of the prefix idea. A routing decision
reads the prefix runs several times -- once for the candidate loop's local matches,
once per candidate when it asks the source :class:`~proposed.selector.KeySelector` which
peer would serve the gap -- and all of them must see the *same* directory state or the
decision is incoherent. Pinning also means the directory is walked once per request,
not once per read. The other three are live: they move only when a fact is folded or a
decision commits, and neither happens inside a pin.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import AbstractSet, Dict, Iterator, List, Optional, Sequence, Tuple

from proposed import Sensed, SensorView, View

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

    #: The keys one decision pinned and the runs it read for them, while
    #: :meth:`pinned` holds; ``None`` outside such a decision.
    _snapshot: Optional[Tuple[List[str], Dict[str, int]]] = None

    def prefix_lengths(self, block_keys: Sequence[str]) -> Dict[str, int]:
        """``instance -> leading blocks of ``block_keys`` it holds contiguously``.

        Computed from the real ``locate_volumes`` result
        (``{key -> {volume_id -> StorageInfo}}``); the run stops at the first
        missing block, and instances holding none of the first block are omitted.
        Served from the pinned snapshot while one is held.
        """
        keys = list(block_keys)
        if self._snapshot is not None:
            pinned_keys, counts = self._snapshot
            assert keys == pinned_keys, (
                "a pinned view answers for the keys it was pinned to; one decision "
                "reads one snapshot"
            )
            return counts
        return prefix_lengths_of(self.locate(keys), keys)

    @contextmanager
    def pinned(self, block_keys: Sequence[str]) -> Iterator[None]:
        """Read the directory once, and serve that snapshot for the block.

        Scoped state on the view rather than a snapshot object passed around,
        because every selector a decision consults senses through this same view
        (:meth:`~proposed.selector.Selector.attach`) and would otherwise read past the
        snapshot into the live directory.

        Sound because one decision cannot be interleaved with another: the directory
        read underneath it is a plain synchronous method
        (:meth:`~proposed.deployment.Controller.locate_raw`), so there is no
        suspension point between the pin and its release. Should one ever appear,
        the assertions fire -- here on a second decision entering, in
        :meth:`prefix_lengths` on a read of other keys arriving inside one.
        """
        assert self._snapshot is None, "a decision already holds this view's snapshot"
        keys = list(block_keys)
        self._snapshot = (keys, prefix_lengths_of(self.locate(keys), keys))
        try:
            yield
        finally:
            self._snapshot = None


class ClusterView(SensorView):
    """The cluster this capability decides against: :attr:`cluster`.

    Its reads are the sensor's own members (``busy_until``, ``occupancy``,
    ``predict_occupancy``), stated once where the sensor is
    (:class:`~kvcache_sim.control._sensor.ClusterSensor`).
    """

    cluster = Sensed()


class ReservedView(SensorView):
    """The prefills this plane promised and has not seen land: :attr:`reserved`.

    Written when a decision commits, read when the next one predicts the decode batch
    it will meet, and self-expiring on that read
    (:meth:`~kvcache_sim.control._sensor.ReservationSensor.pending`).
    """

    reserved = Sensed("reservation")


class RoutedView(SensorView):
    """The pulls this plane has priced against a peer: :attr:`routed`.

    Written by the plane that priced them, read by the one link that answers a fetch
    from them (:class:`~kvcache_sim.control._selector.RoutedPull`) -- and reading one
    consumes it (:meth:`~kvcache_sim.control._sensor.RoutedPullSensor.claim`).
    """

    routed = Sensed("routed-pull")


class KVView(PrefixView, ClusterView, ReservedView, RoutedView):
    """Every read a KV-cache decision makes, over the run's ports.

    Each is optional and independent, so a caller wanting the prefix runs alone
    composes :class:`PrefixView` and never names the rest. C3 puts
    :class:`~proposed.view.View` once at the tail of this MRO, so the run's ports are
    taken up once however many are named.
    """
