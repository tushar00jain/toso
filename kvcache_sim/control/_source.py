"""Which peer serves a prefix gap: the KV-cache source :class:`~proposed.selector.KeySelector`.

The *only* part of KV-cache routing that is a store question, so the only part that
goes through the shared selector interface. Everything else the scheduler decides --
which instance prefills, whether to pull at all or recompute locally, the TTFT/TBT
gates, where decode lands -- is compute placement, which the store knows nothing
about, and stays in :mod:`kvcache_sim.control.scheduler`.

Not installed in the controller, unlike ``dedup_sim``'s selector: the scheduler
wants to *price* a source against recomputing the prefix rather than be handed
one, so it calls :meth:`select` itself and then decides.

:class:`LongestPrefixKeySelector` ranks on reuse value alone and is the default.
:class:`SpreadReadsKeySelector` is that same ranking with a bounded discount for how
much this selector has *lately routed* at each source, so a host holding a hot
prefix does not serve every read of it. It is opt-in and off by default:
``python -m kvcache_sim hotspot --spread-reads`` hands a fresh one to each of that
scenario's cache-aware runs as
:func:`~kvcache_sim.workload._serving.scheduler`'s ``source_selector``.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from proposed import DecisionLog, KeySelector, Selection

from ._view import prefix_lengths_of

__all__ = ["LongestPrefixKeySelector", "SpreadReadsKeySelector"]


class LongestPrefixKeySelector(KeySelector):
    """Rank instances by how much of the requested block prefix they hold.

    Longest contiguous run first, instance id as the tie-break, so the choice is
    deterministic. The requester is accepted and ignored: reuse value here is a
    property of the *prefix*, and only the scheduler holds the other half of the
    trade (a nearer peer is cheaper to fetch from, a shorter prefix means more
    recompute), so it weighs locality itself when it prices the pull.
    """

    name = "longest-prefix"

    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        """Instances holding a leading run of ``keys``, longest run first."""
        counts = self._prefix_runs(list(keys))
        if not counts:
            return Selection.of([])
        ranked = sorted(counts, key=lambda inst: (-counts[inst], inst))
        return Selection.of(ranked)

    def _prefix_runs(self, keys: Sequence[str]) -> Dict[str, int]:
        """Per-instance prefix runs, off whichever view this selector was attached to.

        The scheduler attaches its :class:`~kvcache_sim.control._view.KVView`, whose
        snapshot a routing decision pins, so the whole decision reads one directory.
        A run that installs this selector on its own can only attach the plain
        :class:`~proposed.view.View`, since a prefix run is a KV-cache notion the
        store has no reason to know. Use the derived read when offered, derive it
        otherwise, off one shared definition.
        """
        derived = getattr(self.view, "prefix_lengths", None)
        if derived is not None:
            return derived(keys)
        return prefix_lengths_of(self.view.locate(keys), keys)


class _Grant(NamedTuple):
    """One source handed out, and the moment it stops counting as recent."""

    expires_at: float
    source: str


class _Grants:
    """Sources this selector has named lately, tallied per source.

    A window rather than a running total: a total that only grows is not a load
    signal -- after enough traffic the differences between sources wash out and the
    ranking decays back to prefix-and-id, the exact behaviour
    :class:`SpreadReadsKeySelector` exists to change.

    Expiry runs on the **read**, the idiom :mod:`kvcache_sim.control._pending` uses
    and for the same reason: a selection reads the tally before it adds to it, so
    sweeping on the write would leave stale grants in place for precisely the read
    about to be answered against them.
    """

    def __init__(self, window: float) -> None:
        self.window = window
        self._issued: List[_Grant] = []

    def issue(self, at: float, source: str) -> None:
        """Record that ``source`` was named at ``at``; it counts for one window."""
        self._issued.append(_Grant(at + self.window, source))

    def outstanding(self, now: float) -> Dict[str, int]:
        """``source -> grants still inside their window``, dropping the rest.

        A non-positive window expires every grant the instant it is issued, which
        makes :class:`SpreadReadsKeySelector` rank exactly as :class:`LongestPrefixKeySelector`
        does.
        """
        self._issued = [g for g in self._issued if g.expires_at > now]
        counts: Dict[str, int] = {}
        for grant in self._issued:
            counts[grant.source] = counts.get(grant.source, 0) + 1
        return counts


class SpreadReadsKeySelector(LongestPrefixKeySelector):
    """Longest prefix, minus a bounded discount for reads lately routed there.

    Every replica holding a hot prefix ranks *identically* under
    :class:`LongestPrefixKeySelector`, and the instance-id tie-break then sends every read
    of it to whichever replica sorts first: deterministic, but a hotspot. This selector
    breaks that tie on something that moves.

    The load signal is this selector's own bookkeeping, because :mod:`proposed.view`
    has no ``load()`` to read and :meth:`select` is chokepoint enough -- every read
    this selector influences passes through it. A named source counts for ``window``
    seconds of the loop's clock and then stops counting (:class:`_Grants`);
    ``window`` is the one number this selector cannot derive -- too short and it
    forgets a peer it has just piled four transfers onto, too long and it goes on
    penalising a peer that finished them.

    Prefix length is the *value* signal and outstanding grants the *cost* signal,
    and a grant is only a guess about congestion, so the rank key bounds the cost:

        ``(-(run - min(outstanding, max_discount)), outstanding, instance)``

    Load may shave at most ``max_discount`` blocks off a run, so a source ahead by
    more than that wins however busy it is; among sources the discount has levelled,
    the raw grant count decides, so two identical replicas alternate indefinitely
    instead of reverting to id order once the discount saturates. The discount ranks
    and does nothing else: the scheduler prices the winner against the run length it
    read from the directory itself (``_LongerThanLocal`` in the scheduler), so a
    discounted source can never report a shorter prefix than it holds into the
    pull-versus-recompute decision.

    Determinism: every component of the key is an integer or an id, the instance id
    is last, and no branch reads a wall clock or an unseeded RNG, so a rank is total
    and a run reproduces. ``window`` is measured on
    :meth:`~proposed.view.View.now`, the loop's virtual clock.

    **The count is a model, not a measurement.** A grant records that this selector
    *named* a source, not that any byte moved -- the scheduler asks once per prefill
    candidate while pricing and then keeps one plan, so most grants belong to
    candidates that were dropped, and since nothing counts a source that served a
    read this selector never granted, the tally drifts one-way *above* reality: read
    it as "recently pointed at", not "currently serving". The scheduler's
    ``busy_until`` is a prediction too, but ``PrefillFinished`` corrects it against
    what happened, and **this selector has no such correction path** -- a window only
    bounds how long a wrong count can persist, it never learns that it was wrong.
    The fix is a measurement: real per-volume serving load, counted in the data
    plane and surfaced on :class:`~proposed.view.View`, the observation
    :mod:`proposed.view` leaves ``load()`` out for "until a caller can observe one".
    Then the ranking stays and the private tally goes.

    Args:
        window: seconds a grant counts for. Non-positive means no memory, which
            reproduces :class:`LongestPrefixKeySelector`'s ranking exactly.
        max_discount: the most blocks of prefix run load may cancel out. ``0`` makes
            the load term a pure tie-break between sources whose runs are already
            equal, which is enough for the replicated-hot-prefix case.
        trace: optional :class:`~proposed.selector.DecisionLog`. Records only.
    """

    name = "spread-reads"

    def __init__(
        self,
        *,
        window: float = 1.0,
        max_discount: int = 1,
        trace: Optional[DecisionLog] = None,
    ) -> None:
        self.max_discount = max_discount
        self.trace = trace
        self._grants = _Grants(window)

    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        """Rank as :class:`LongestPrefixKeySelector` does, spread over equal-value peers.

        The whole ranking is returned, not just the head, so a caller that rejects
        the first source still has the rest in a useful order. Only the head counts
        as a grant: it is the one the callers here act on, and counting a source the
        caller was never going to use would inflate the drift described above.
        """
        runs = self._prefix_runs(list(keys))
        if not runs:
            # An abstention: nothing was routed, so there is no grant to record.
            return Selection.of([])
        now = self.view.now()
        outstanding = self._grants.outstanding(now)
        ranked = sorted(runs, key=lambda inst: self._rank(inst, runs, outstanding))
        chosen = ranked[0]
        self._grants.issue(now, chosen)
        if self.trace is not None:
            self.trace.record(
                now,
                "spread",
                f"{requester} <- {chosen} "
                f"(prefix {runs[chosen]}, {outstanding.get(chosen, 0)} outstanding)",
            )
        return Selection.of(ranked)

    def _rank(
        self, inst: str, runs: Dict[str, int], outstanding: Dict[str, int]
    ) -> Tuple[int, int, str]:
        """Sort key for one source: discounted run, raw load, id. Total, always."""
        held = outstanding.get(inst, 0)
        return (-(runs[inst] - min(held, self.max_discount)), held, inst)
