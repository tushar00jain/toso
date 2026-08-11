"""Which peer serves a prefix gap: the KV-cache source :class:`~proposed.policy.Policy`.

This is the *only* part of KV-cache routing that is a store question, so it is
the only part that goes through the shared policy interface. Everything else the
scheduler decides -- which instance prefills, whether to pull at all or recompute
locally, the TTFT/TBT gates, where decode lands -- is compute placement, which
the store knows nothing about, and stays in
:mod:`kvcache_sim.control.scheduler`.

Unlike ``dedup_sim``'s policy, this one is **not** installed in the controller.
The scheduler does not want to be handed a source; it wants to *price* one
against the alternative of recomputing the prefix. So it calls :meth:`select`
itself, through the view, and then decides.
"""

from __future__ import annotations

from typing import Any, Sequence

from proposed import Policy, Selection

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
        counts = await view.prefix_lengths(list(keys))
        if not counts:
            return Selection.of([])
        ranked = sorted(counts, key=lambda inst: (-counts[inst], inst))
        return Selection.of(ranked)
