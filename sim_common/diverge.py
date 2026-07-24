"""Localize the first event at which two supposedly-identical DES runs diverge.

A discrete-event sim is deterministic: same inputs => byte-identical trace. The
usual guard (:mod:`sim_common.tests`) asserts *that* two runs match, but when a
determinism regression sneaks in -- a set iterated in hash order, an accidental
wall-clock read past the contract lint, a dict that lost its ordering -- a
whole-trace equality check only tells you the runs differ, not *where*.

This module folds each run's trace into the :mod:`sim_common.trace` hash chain
(``h_i = H(h_{i-1} || event_i)``) and reports the first index where the two
chains disagree. Because it is a chain, ``h_i`` is equal for both runs for every
``i`` before the first differing event and unequal from it on -- so the first
mismatching chain index *is* the first differing event. This is the same
localize-the-divergence technique deterministic model-checking simulators use
when they log a per-iteration/per-run state hash, adapted here to our trace as the
canonical run fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from sim_common.trace import chain_digests, Event, Trace

# Either a Trace or a bare event sequence may be compared.
Events = Union[Trace, Sequence[Event]]


def _events(x: Events) -> List[Event]:
    return list(x.events if isinstance(x, Trace) else x)


def _fmt(event: Optional[Event]) -> str:
    if event is None:
        return "<none: run ended here>"
    t, kind, msg = event
    return f"t={t:<8.3f} {kind:<6} {msg}"


@dataclass
class Divergence:
    """Where two runs first differ.

    ``index`` is the 0-based event position of the first disagreement. ``a_event``
    / ``b_event`` are the events each run recorded there (``None`` if that run had
    already ended -- i.e. one trace is a strict prefix of the other). ``context``
    is the shared run of events immediately before the divergence, for orientation.
    """

    index: int
    a_event: Optional[Event]
    b_event: Optional[Event]
    context: List[Event]

    def describe(self) -> str:
        """A short, human-readable divergence report."""
        out = [f"first divergence at event #{self.index}"]
        if self.context:
            out.append("  common prefix (last shared events):")
            out.extend(f"    {_fmt(e)}" for e in self.context)
        out.append(f"  run A #{self.index}: {_fmt(self.a_event)}")
        out.append(f"  run B #{self.index}: {_fmt(self.b_event)}")
        return "\n".join(out)


def first_divergence(a: Events, b: Events, *, context: int = 3) -> Optional[Divergence]:
    """Return the first point two runs diverge, or ``None`` if identical.

    ``a`` / ``b`` may be :class:`~sim_common.trace.Trace` objects or bare event
    sequences. ``context`` is how many shared events preceding the divergence to
    include in the result. Runs that agree on every event but differ in length
    (one ended early -- e.g. a deadlock) diverge at the first extra event.
    """
    ea, eb = _events(a), _events(b)
    ca, cb = chain_digests(ea), chain_digests(eb)

    n = min(len(ca), len(cb))
    idx: Optional[int] = None
    for i in range(n):
        if ca[i] != cb[i]:
            idx = i
            break

    if idx is None:
        if len(ea) == len(eb):
            return None  # byte-identical runs
        idx = n  # identical prefix; one run has trailing events the other lacks

    lo = max(0, idx - context)
    return Divergence(
        index=idx,
        a_event=ea[idx] if idx < len(ea) else None,
        b_event=eb[idx] if idx < len(eb) else None,
        context=ea[lo:idx],
    )
