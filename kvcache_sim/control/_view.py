"""The senses one KV-cache decision reads, composed: :class:`KVView`.

A sense is one read, as a class, and a :class:`~proposed.view.View` in its own right:
a view is assembled by naming the senses a decision makes, and a capability needing
one of them composes that one alone (``view.derived(ClusterSense, cluster=m)``). A
selector reads the one it needs:

* :class:`PrefixSense`: how many leading blocks of a prompt an instance holds
  contiguously. The base view stops at "who holds this key", and a cache is only
  useful as a contiguous prefix -- a KV-cache notion, not a store notion;
* :class:`ClusterSense`: the predicted prefill queues and observed decode batches
  (:class:`~kvcache_sim.control._cluster.KVClusterModel`), which is what ranks, prices
  and gates a candidate host;
* :class:`ReservedSense`: the prefills this plane promised and has not seen land
  (:class:`~kvcache_sim.control._pending.Reservations`), read by the decode-side
  prediction of a run that rolls occupancy forward, and composed in only by such a run;
* :class:`RoutedSense`: the pulls a decision priced against a peer
  (:class:`~kvcache_sim.control._pending.RoutedPulls`), read by the fetch chain's head
  link alone (:class:`~kvcache_sim.control.scheduler.RoutedPull`).

Each sense claims its own keyword through :meth:`~proposed.view.View.derived`, holds
its own state and raises its own "this run supplied none", so composing one in is a
name in a class statement and nothing else moves. The senses are disjoint: different
selectors read different ones and none of them touches another's, which is what makes
sensing any of them ambiently safe.

They are all *observed state* -- this plane's own records as much as the directory --
so whatever ranks, prices or gates senses it here instead of being handed the object,
and the plane reports its own decision into them the same way. What does not come this
way is a host's fact: the run puts a service in front of the model for those.

:meth:`PrefixSense.pinned` is the second half of the prefix idea. A routing decision
reads the prefix runs several times -- once for the candidate loop's local matches,
once per candidate when it asks the source :class:`~proposed.selector.KeySelector` which
peer would serve the gap -- and all of them must see the *same* directory state or the
decision is incoherent. Pinning also means the directory is walked once per request,
not once per read. The other senses are live: they move only when a fact is folded or a
decision commits, and neither happens inside a pin.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    AbstractSet, Any, Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING,
)

from proposed import View

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._cluster import KVClusterModel
    from ._pending import Reservations, RoutedPulls

__all__ = [
    "prefix_lengths_of",
    "PrefixSense",
    "ClusterSense",
    "ReservedSense",
    "RoutedSense",
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

    Split from the read that feeds it: :meth:`PrefixSense.prefix_lengths` reads the
    directory (or serves a pinned snapshot), while
    :class:`~kvcache_sim.control._source.LongestPrefixKeySelector` may be attached to a
    plain :class:`~proposed.view.View` and reads it itself. One definition either
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


class PrefixSense(View):
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


class ClusterSense(View):
    """The cluster this capability decides against: :attr:`cluster`.

    Its reads are the model's own members (``busy_until``, ``occupancy``,
    ``predict_occupancy``), stated once where the model is.
    """

    def __init__(
        self,
        *ports: Any,
        cluster: Optional["KVClusterModel"] = None,
        **senses: Any,
    ) -> None:
        super().__init__(*ports, **senses)
        self._cluster = cluster

    @property
    def cluster(self) -> "KVClusterModel":
        """The run's model of the cluster, as a sense beside the directory.

        Raises like :meth:`~proposed.view.View.transfer_cost` does and for its reason:
        a view composed without the model cannot answer for the cluster, and "idle" is
        not a number to invent.
        """
        if self._cluster is None:
            raise RuntimeError(
                "this view was composed without a cluster model, so nothing here can "
                "be ranked, priced or gated against the cluster"
            )
        return self._cluster


class ReservedSense(View):
    """The prefills this plane promised and has not seen land: :attr:`reserved`.

    Written when a decision commits, read when the next one predicts the decode batch
    it will meet, and self-expiring on that read
    (:meth:`~kvcache_sim.control._pending.Reservations.pending`).
    """

    def __init__(
        self,
        *ports: Any,
        reserved: Optional["Reservations"] = None,
        **senses: Any,
    ) -> None:
        super().__init__(*ports, **senses)
        self._reserved = reserved

    @property
    def reserved(self) -> "Reservations":
        """The prefills promised and not yet landed.

        Raises like :attr:`ClusterSense.cluster` does, and for its reason: a run that
        promises nothing composes no record, and reading one that was never written
        would under-count every predicted batch and report no error.
        """
        if self._reserved is None:
            raise RuntimeError(
                "this view was composed without a reservation record, so this run "
                "does not roll decode occupancy forward and nothing here can read "
                "the prefills it promised"
            )
        return self._reserved


class RoutedSense(View):
    """The pulls this plane has priced against a peer: :attr:`routed`.

    Written by the plane that priced them, read by the one link that answers a fetch
    from them (:class:`~kvcache_sim.control.scheduler.RoutedPull`) -- and reading one
    consumes it (:meth:`~kvcache_sim.control._pending.RoutedPulls.claim`).
    """

    def __init__(
        self,
        *ports: Any,
        routed: Optional["RoutedPulls"] = None,
        **senses: Any,
    ) -> None:
        super().__init__(*ports, **senses)
        self._routed = routed

    @property
    def routed(self) -> "RoutedPulls":
        """The pulls priced and not yet claimed.

        Raises like :attr:`ClusterSense.cluster` does, and for its reason.
        """
        if self._routed is None:
            raise RuntimeError(
                "this view was composed without a routed-pull record, so nothing "
                "here can claim the peer a pull was priced against"
            )
        return self._routed


class KVView(PrefixSense, ClusterSense, ReservedSense, RoutedSense):
    """The senses a KV-cache decision reads, over the run's ports.

    Every sense is optional and independent, so a caller wanting the prefix runs alone
    composes :class:`PrefixSense` and never names the rest. C3 puts
    :class:`~proposed.view.View` once at the tail of this MRO, so the run's ports are
    taken up once however many senses are named.
    """
