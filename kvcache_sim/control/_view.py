"""Everything the KV-cache scheduler senses, as one object: :class:`KVView`.

Two reads, one sensor. :class:`~proposed.view.View` stops at "who holds this key",
and a KV-cache scheduler asks one step further on: *how many leading blocks of this
prompt does each instance hold contiguously?* -- because a cache is only useful as a
contiguous prefix. That is a KV-cache notion, not a store notion, so it is a subclass
here rather than a field on the base view.

The other read is the cluster the scheduler decides against
(:class:`~kvcache_sim.control._cluster.KVClusterModel`, handed in at
:meth:`~proposed.view.View.derived` and exposed as :attr:`KVView.cluster`): predicted
prefill queues, observed decode batches, and the pull a decision priced. That model
is the capability's own, so the run cannot supply it and the base view does not carry
it -- but it is *observed state* like the directory is, so whatever ranks, prices or
gates against it senses it here instead of being handed the model. Writing it is
still the owner's: a fact is folded by the service the run puts in front of the model
and a decision is committed by the plane that took it, neither of them through here.

:meth:`KVView.pinned` is the second half of the prefix idea. A routing decision reads
the prefix runs several times -- once for the candidate loop's local matches, once
per candidate when it asks the source :class:`~proposed.selector.KeySelector` which peer
would serve the gap -- and all of them must see the *same* directory state or the
decision is incoherent. Pinning also means the directory is walked once per
request, not once per read. The model reads are live: it moves only when a fact is
folded or a decision commits, and neither happens inside a pin.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    AbstractSet, Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING,
)

from proposed import Controller, Endpoint, TransferCost, View

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._cluster import KVClusterModel

__all__ = ["prefix_lengths_of", "KVView"]


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

    Split from the read that feeds it: :meth:`KVView.prefix_lengths` reads the
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


class KVView(View):
    """A :class:`~proposed.view.View` plus prefix runs and the cluster model.

    Args:
        directory / topology / cost: the run's ports, as
            :meth:`~proposed.view.View.derived` passes them.
        cluster: the run's :class:`~kvcache_sim.control._cluster.KVClusterModel`.
            ``None`` -- for a caller wanting the prefix runs alone -- makes
            :attr:`cluster` raise rather than answer "idle".
    """

    #: The keys one decision pinned and the runs it read for them, while
    #: :meth:`pinned` holds; ``None`` outside such a decision.
    _snapshot: Optional[Tuple[List[str], Dict[str, int]]] = None

    def __init__(
        self,
        directory: Controller,
        topology: Dict[str, Endpoint],
        cost: Optional[TransferCost] = None,
        cluster: Optional["KVClusterModel"] = None,
    ) -> None:
        super().__init__(directory, topology, cost)
        self._cluster = cluster

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

    # -- the cluster this capability decides against ------------------------- #
    @property
    def cluster(self) -> "KVClusterModel":
        """The run's model of the cluster, as a sensor beside the directory.

        Its own members are the reads (``busy_until``, ``occupancy``,
        ``predict_occupancy``, ``pending``, ``claim``), stated once where the model
        is. Raises like :meth:`~proposed.view.View.transfer_cost` does and for its
        reason: a sensor derived without a model cannot answer for the cluster, and
        "idle" is not a number to invent.
        """
        if self._cluster is None:
            raise RuntimeError(
                "this view was derived without a cluster model, so nothing here can "
                "be ranked, priced or gated against the cluster"
            )
        return self._cluster
