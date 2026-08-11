"""Process-wide simulation config (stdlib-only; no config file).

A single typed settings object, read *ambiently* via :func:`current`, so
cross-cutting run knobs -- the kind that would otherwise have to be threaded
through every scenario down to the object that needs them -- can be set once at
startup and read anywhere.

Sourcing precedence (highest first): explicit :func:`configure` overrides (wired
from a CLI flag) > ``TOSO_*`` environment variables > dataclass defaults. There
is deliberately no config *file* yet; add a loader in :func:`configure` when a
second or third knob makes it worthwhile.

Scope discipline -- mostly cross-cutting *debug/output* flags belong here (whether
to maintain the trace hash chain, etc.). Anything that changes the *simulated
result* -- a seed, a :class:`~sim_common.cost_model.MachineProfile` -- normally
stays an explicit function argument, so a run is always reproducible from its call
args and never from hidden global state. Most flags here never affect event
content or any measured metric, which is what makes an ambient read safe under the
determinism contract: config is loaded once at startup (the environment is read
there), so :func:`current` on the sim path is a plain in-memory attribute read.

The one deliberate exception is :attr:`SimConfig.contention` (the network/storage
contention model): it is a run-wide *fidelity* knob that DOES change measured
timing. It is placed here so a run selects one model once, at startup, and every
transport reads it ambiently -- there is no longer a per-scenario ``contention``
override argument; the mode is read from this config everywhere. It remains
deterministic under the contract: the mode is fixed for the whole run, not read
per event, and re-rating is ordered by a monotonic transfer sequence.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, Optional

__all__ = ["SimConfig", "current", "configure", "overrides"]


@dataclass(frozen=True)
class SimConfig:
    """Cross-cutting, debug/output-only run settings (see the module docstring)."""

    # Maintain the trace hash chain incrementally so a run fingerprint / divergence
    # bisection is cheap to sample. Off by default -- a normal measurement run pays
    # no per-event hashing. See sim_common.trace.Trace / sim_common.diverge.
    fingerprint: bool = False

    # Record the chronological event trace at all. On by default (the trace is the
    # human-readable output and the fingerprint's input). Turning it off ("quiet
    # mode") makes every sim_common.trace.Trace.record call a no-op, so a large run
    # pays none of the per-event string-formatting / list-growth bookkeeping. It
    # only removes trace side effects -- virtual time, ordering and every measured
    # metric are byte-identical either way. See sim_common.trace.Trace.
    trace: bool = True

    # Back the controller directory with the real torchstore Trie (True, default)
    # or a lightweight dict shim (False). The shim runs every bit of the real
    # Controller decision logic over a plain dict instead of a pygtrie Trie, so a
    # scale run skips the per-key trie tax; it is opt-in because the faithful real
    # directory is the default fidelity story. Like the flags above it never
    # changes a measured metric -- only the container behind the same Mapping
    # surface, and directory iteration order is never consumed by a metric -- so
    # real and shim runs are byte-identical (asserted by the divergence-gate
    # tests). See realsim.adapters.real_controller.make_controller_adapter.
    real_directory: bool = True

    # Network/storage contention model for the transport seam: one of
    # ``"none"`` (default), ``"serialize"``, or ``"progressive"`` (see
    # sim_common.resources). Unlike the flags above this DOES change measured
    # timing -- it is a deliberate fidelity model, not a debug/output toggle. It
    # is read ambiently everywhere (there is no per-scenario override argument):
    # a run selects one mode once at startup, so a non-default mode is
    # intentionally not byte-identical to ``"none"``. ``"none"`` reproduces the
    # historical independent-sleep behavior exactly.
    contention: str = "none"

    # Collapse each transport op's per-component resource charges into a single
    # virtual-clock sleep (see sim_common.resources / realsim.seams.transport).
    # Off by default -- today's exact per-component behavior. When ON *and*
    # ``contention == "none"``, a get charges one combined sleep of
    # ``storage+mem+network`` instead of three, and a put one of
    # ``network+storage`` instead of two, cutting the event-loop bounces per op
    # (the cheap large-run path). The combined sleep advances the clock by the
    # exact same total, so a request's completion time in isolation is unchanged;
    # it is NOT byte-identical because the intermediate sub-charge instants vanish
    # (a documented coarsening -- a concurrent coroutine may interleave
    # differently). It is inert when ``contention != "none"`` (there the
    # per-component awaits are intrinsic to the multi-resource contention model),
    # so this only reduces bounces on the non-contended default path.
    collapse_charges: bool = False

    # One-way latency of a service boundary, charged by realsim.seams.link's
    # ServiceHop: ``coordinator_rtt`` for a serving host reaching its coordinator
    # (CoordinatorHandle), ``controller_rtt`` for anyone reaching the directory
    # (LocalControllerHandle) -- which is every capability, the baseline included,
    # and every consultation of a policy installed there. Like ``contention``
    # these are fidelity knobs that DO change measured timing, and deliberately
    # so: a request pays a hop out and back before it can proceed, so a
    # coordinator hop lands in TTFT, which is the number kvcache exists to move.
    # ``0.0`` (the default) makes a hop inline -- awaiting a coroutine that never
    # suspends does not yield -- so the default path is byte-identical to calling
    # the object directly. One value per boundary for the whole run; the delivery
    # lag of one-way sends is not modelled (see the handles' docstrings).
    coordinator_rtt: float = 0.0
    controller_rtt: float = 0.0


_current = SimConfig()


def current() -> SimConfig:
    """Return the process-wide config (a cheap in-memory read; safe anywhere)."""
    return _current


def _bool_env(name: str) -> Optional[bool]:
    """Parse a boolean environment variable, or ``None`` if it is unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _from_env() -> dict:
    """Collect the ``TOSO_*`` environment overrides that are actually set."""
    out: dict = {}
    fingerprint = _bool_env("TOSO_FINGERPRINT")
    if fingerprint is not None:
        out["fingerprint"] = fingerprint
    trace = _bool_env("TOSO_TRACE")
    if trace is not None:
        out["trace"] = trace
    real_directory = _bool_env("TOSO_REAL_DIRECTORY")
    if real_directory is not None:
        out["real_directory"] = real_directory
    contention = os.environ.get("TOSO_CONTENTION")
    if contention is not None:
        out["contention"] = contention.strip().lower()
    collapse_charges = _bool_env("TOSO_COLLAPSE_CHARGES")
    if collapse_charges is not None:
        out["collapse_charges"] = collapse_charges
    coordinator_rtt = os.environ.get("TOSO_COORDINATOR_RTT")
    if coordinator_rtt is not None:
        out["coordinator_rtt"] = float(coordinator_rtt)
    controller_rtt = os.environ.get("TOSO_CONTROLLER_RTT")
    if controller_rtt is not None:
        out["controller_rtt"] = float(controller_rtt)
    return out


def configure(**overrides) -> SimConfig:
    """Set the process config: defaults <- environment <- explicit overrides.

    Call once at startup (e.g. wiring a ``--fingerprint`` CLI flag). ``None``
    values are ignored, so an unset CLI flag defers to the environment / default
    rather than forcing the value off. Fields not supplied by any source fall back
    to the :class:`SimConfig` defaults.
    """
    global _current
    values = _from_env()
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    _current = SimConfig(**values)
    return _current


@contextmanager
def overrides(**kw) -> Iterator[SimConfig]:
    """Temporarily override the config within a ``with`` block (auto-restored).

    Scoped and restore-on-exit, so tests can flip a flag without leaking it into
    other tests -- the isolation a bare module global would lack.
    """
    global _current
    prev = _current
    _current = replace(_current, **kw)
    try:
        yield _current
    finally:
        _current = prev
