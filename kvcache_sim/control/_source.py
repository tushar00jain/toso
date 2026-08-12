"""Which peer serves a prefix gap: the KV-cache source :class:`~proposed.policy.Policy`.

This is the *only* part of KV-cache routing that is a store question, so it is
the only part that goes through the shared policy interface. Everything else the
scheduler decides -- which instance prefills, whether to pull at all or recompute
locally, the TTFT/TBT gates, where decode lands -- is compute placement, which
the store knows nothing about, and stays in
:mod:`kvcache_sim.control.scheduler`.

Unlike ``dedup_sim``'s policy, this one is **not** installed in the controller: the
scheduler wants to *price* a source against recomputing the prefix rather than be
handed one, so it calls :meth:`select` itself, through the view, and then decides.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from proposed import Policy, Selection

from ._view import prefix_lengths_of

__all__ = ["LongestPrefixPolicy"]


class LongestPrefixPolicy(Policy):
    """Rank instances by how much of the requested block prefix they hold.

    Longest contiguous run first, instance id as the tie-break, so the choice is
    deterministic. The requester is accepted and ignored: reuse value here is a
    property of the *prefix*, and the caller weighs locality itself when it
    prices the pull (a nearer peer is cheaper to fetch from, but a shorter prefix
    means more recompute, and only the scheduler holds both halves of that
    trade).
    """

    name = "longest-prefix"

    async def select(
        self, view: Any, keys: Sequence[str], requester: str
    ) -> Selection:
        """Instances holding a leading run of ``keys``, longest run first."""
        counts = await self._prefix_runs(view, list(keys))
        if not counts:
            return Selection.of([])
        ranked = sorted(counts, key=lambda inst: (-counts[inst], inst))
        return Selection.of(ranked)

    @staticmethod
    async def _prefix_runs(view: Any, keys: Sequence[str]) -> Dict[str, int]:
        """Per-instance prefix runs, from whichever view this caller has.

        The scheduler hands a :class:`~kvcache_sim.control._view.KVView` -- usually
        the *pinned* one, so a routing decision reads one directory snapshot. The
        controller can only hand the plain :class:`~proposed.view.View` it was built
        with, since a prefix run is a KV-cache notion the store has no reason to
        know. Use the derived read when offered, derive it otherwise, off one shared
        definition.
        """
        pinned = getattr(view, "prefix_lengths", None)
        if pinned is not None:
            return await pinned(keys)
        return prefix_lengths_of(await view.locate(keys), keys)
