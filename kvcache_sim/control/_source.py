"""Which peer serves a prefix gap: the KV-cache source :class:`~proposed.selector.KeySelector`.

The *only* part of KV-cache routing that is a store question, so the only part that
goes through the shared selector interface. Everything else the scheduler decides --
which instance prefills, whether to pull at all or recompute locally, the TTFT/TBT
gates, where decode lands -- is compute placement, which the store knows nothing
about, and stays in :mod:`kvcache_sim.control.scheduler`.

Read twice by the one plane that holds it, which is why it is a ranking and not a
member of that plane's surface: the scheduler prices a source against recomputing the
prefix rather than being handed one, so it calls :meth:`select` itself while deciding
-- and the same object sits behind the answer it later gives a fetch
(:meth:`~kvcache_sim.control.scheduler._Scheduler.sources`), so the peer priced is the
peer read from.

:class:`LongestPrefixKeySelector` ranks on reuse value alone and is the default.
Spreading reads over the replicas of a hot prefix is that ranking under
:class:`~proposed.selector.Discount`, which bounds how much load may cancel out of a
prefix run, so a host holding a hot prefix does not serve every read of it. It is
opt-in and off by default: ``python -m kvcache_sim hotspot --spread-reads`` hands a
fresh ``Discount(LongestPrefixKeySelector())`` to each of that scenario's cache-aware
runs as :func:`~kvcache_sim.workload._serving.scheduler`'s ``source_selector``.
"""

from __future__ import annotations

from typing import Dict, Sequence

from proposed import Key, KeySelector, Selection

from ._view import prefix_lengths_of

__all__ = ["LongestPrefixKeySelector"]


class LongestPrefixKeySelector(KeySelector[int]):
    """Rank instances by how much of the requested block prefix they hold.

    Longest contiguous run first, instance id as the tie-break, so the choice is
    deterministic. The requester is accepted and ignored: reuse value here is a
    property of the *prefix*, and only the scheduler holds the other half of the
    trade (a nearer peer is cheaper to fetch from, a shorter prefix means more
    recompute), so it weighs locality itself when it prices the pull.
    """

    name = "longest-prefix"

    async def select(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[int]:
        """Instances holding a leading run of ``keys``, longest run first.

        ``KeySelector[int]``: the price is the run itself, in blocks -- a measurement,
        not a valuation, and what a source is *worth* remains the scheduler's to
        weigh. Published because the number this ranking already turns on is the only
        honest price for it, and because a re-ranking over it has to weigh something
        (:class:`~proposed.selector.Discount`).
        """
        counts = self._prefix_runs(list(keys))
        if not counts:
            return Selection.of([])
        ranked = sorted(counts, key=lambda inst: (-counts[inst], inst))
        return Selection.priced([(inst, counts[inst]) for inst in ranked])

    def _prefix_runs(self, keys: Sequence[Key]) -> Dict[str, int]:
        """Per-instance prefix runs, off whichever view this selector was attached to.

        The scheduler attaches its :class:`~kvcache_sim.control._view.KVView`, whose
        snapshot a routing decision pins, so the whole decision reads one directory.
        A run that stands this selector up on its own can only attach the plain
        :class:`~proposed.view.View`, since a prefix run is a KV-cache notion the
        store has no reason to know. Use the derived read when offered, derive it
        otherwise, off one shared definition.
        """
        derived = getattr(self.view, "prefix_lengths", None)
        if derived is not None:
            return derived(keys)
        return prefix_lengths_of(self.view.locate(keys), keys)
