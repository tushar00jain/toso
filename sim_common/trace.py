"""Generic chronological event recorder for discrete-event simulations.

Besides the human-readable render, an event sequence can be folded into a **hash
chain** (``h_i = H(h_{i-1} || event_i)``). The final digest --
:meth:`Trace.fingerprint` -- is a single, stable fingerprint of a run: two runs
share it iff their traces are byte-identical. And because it is a *chain*, the
first index at which two runs' per-event digests disagree is exactly the first
differing event -- which :func:`sim_common.diverge.first_divergence` uses to
localize a determinism regression (the trace-based analog of the per-iteration
state hash that deterministic model-checking simulators log).

Hashing is a **determinism-debugging aid, never part of the measurement path**:
the digest does not affect simulated time or any measured metric. So it is
**off by default** and gated by the process config
(:data:`sim_common.config.SimConfig.fingerprint`): a normal performance run does
no hashing at all. When the config flag is on (the demos' ``--fingerprint``), each
:class:`Trace` maintains the chain incrementally -- one small BLAKE2b update per
:meth:`record` -- so a fingerprint is an O(1) read that can be sampled cheaply
mid-run. The flag is read *ambiently* from the config at construction, so it never
has to be threaded through the scenarios down to each ``Trace``. (The free
functions below fold on demand, so :func:`sim_common.diverge.first_divergence`
works on any recorded trace regardless of the flag.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from sim_common import config

__all__ = ["Event", "chain_digests", "fingerprint", "Trace"]

# One trace event: (virtual time, kind, message).
Event = Tuple[float, str, str]

# Domain-separation seed for the hash chain; bump the suffix if the event
# encoding ever changes so old and new fingerprints never collide.
_CHAIN_SEED = b"toso-sim-trace-chain-v1"
_SEP = b"\x1f"  # unit separator between fields, kept out of message text


def _fold(prev: bytes, event: Event) -> bytes:
    """Extend the hash chain by one event: ``H(prev || repr(t) || kind || msg)``.

    ``repr(float(t))`` is a stable, lossless float encoding in Python 3, so the
    digest is reproducible across runs and processes (unlike the salted builtin
    ``hash``). A 16-byte BLAKE2b digest is ample for divergence bisection.
    """
    now, kind, msg = event
    h = hashlib.blake2b(digest_size=16)
    h.update(prev)
    h.update(repr(float(now)).encode("utf-8"))
    h.update(_SEP)
    h.update(kind.encode("utf-8"))
    h.update(_SEP)
    h.update(msg.encode("utf-8"))
    return h.digest()


def chain_digests(events: Sequence[Event]) -> List[bytes]:
    """Return the cumulative chain digest *after* each event (``h_1..h_n``)."""
    out: List[bytes] = []
    h = _CHAIN_SEED
    for e in events:
        h = _fold(h, e)
        out.append(h)
    return out


def _fold_all(events: Sequence[Event]) -> bytes:
    """Fold an event sequence into its final chain digest."""
    h = _CHAIN_SEED
    for e in events:
        h = _fold(h, e)
    return h


def fingerprint(events: Sequence[Event]) -> str:
    """Fold an event sequence into a single hex run-fingerprint digest."""
    return _fold_all(events).hex()


@dataclass
class Trace:
    """Chronological record of simulated events (one line per event).

    Each entry is ``(time, kind, message)``; rendering produces one formatted
    line per event. Because a discrete-event sim is deterministic, two runs
    produce byte-identical trace strings.

    ``time_width`` and ``kind_width`` control the column widths used when
    rendering, so callers can tune the layout without changing behavior.

    ``hash_chain`` decides whether the event hash chain is maintained incrementally
    (one BLAKE2b update per :meth:`record`). It defaults to the process config's
    :data:`~sim_common.config.SimConfig.fingerprint` flag, read at construction --
    so ``--fingerprint`` turns it on everywhere without threading a param down to
    each ``Trace``. Pass it explicitly to override (e.g. in tests).

    ``enabled`` decides whether :meth:`record` does anything at all. When it is
    ``False`` ("quiet mode") every ``record`` early-returns -- no append, no
    hash-chain update -- so a large run pays none of the per-event string-format /
    list-growth bookkeeping. It defaults to the process config's
    :data:`~sim_common.config.SimConfig.trace` flag, read at construction (same
    ambient pattern as ``hash_chain``). Disabling only removes trace side effects;
    :meth:`render` / :meth:`fingerprint` still work (on an empty event list).
    """

    events: List[Tuple[float, str, str]] = field(default_factory=list)
    time_width: int = 6
    kind_width: int = 6
    # Ambient default: the process config's fingerprint flag at construction time
    # (config.configure(...) / config.overrides(...)). Explicit value still wins.
    hash_chain: bool = field(default_factory=lambda: config.current().fingerprint)
    # Ambient default: the process config's trace flag at construction time. When
    # False, `record` is a no-op (see the class docstring). Explicit value wins.
    enabled: bool = field(default_factory=lambda: config.current().trace)
    # Running hash chain over `events`, maintained only when `hash_chain` is True;
    # derived state, excluded from equality.
    _chain: bytes = field(default=_CHAIN_SEED, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Fold any pre-populated events so `_chain` matches `fingerprint` (only
        # when the chain is enabled; otherwise leave it at the seed).
        if self.hash_chain:
            for e in self.events:
                self._chain = _fold(self._chain, e)

    def record(self, now: float, kind: str, msg: str) -> None:
        """Append an event at ``now`` (and extend the hash chain if enabled).

        A no-op when ``enabled`` is ``False`` (quiet mode): no append and no
        hash-chain update, so all call sites become free through this one object.
        """
        if not self.enabled:
            return
        event = (now, kind, msg)
        self.events.append(event)
        if self.hash_chain:
            self._chain = _fold(self._chain, event)

    def fingerprint(self) -> str:
        """Hash-chain digest of every event so far -- a single run fingerprint.

        Two deterministic runs share a fingerprint iff their traces are identical.
        With ``hash_chain`` on this is the O(1) running digest (cheap to sample
        mid-run); with it off, the digest is folded on demand from the recorded
        events (one pass -- you pay only when you ask). To locate *where* two runs
        diverge, feed the two traces to
        :func:`sim_common.diverge.first_divergence`.
        """
        if self.hash_chain:
            return self._chain.hex()
        return _fold_all(self.events).hex()

    def render_lines(self) -> List[str]:
        """Render the event trace as a list of formatted lines (one per event)."""
        return [
            f"t={t:{self.time_width}.3f}  {kind:<{self.kind_width}} {msg}"
            for (t, kind, msg) in self.events
        ]

    def render(self) -> str:
        """Render the event trace, one line per event."""
        return "\n".join(self.render_lines())
