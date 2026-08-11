"""Executing kvcache scenarios, for tests that assert on their outcomes.

A scenario declares :class:`~realsim.run.Run` values; the demo executes them.
These wrappers do the same thing for a test, so an assertion is one line and the
tests exercise exactly the configurations the demo shows -- not a re-wired copy
of them. Test scaffolding, not API.
"""

from __future__ import annotations

from typing import List, Sequence

from realsim.run import execute, Result, Run

from ..workload import scenarios

__all__ = [
    "results",
    "run",
    "run_disaggregation",
    "run_early_rejection",
    "run_eviction_sweep",
    "run_hotspot",
    "run_overload",
    "run_shared_prefix",
]


def results(runs: Sequence[Run]) -> List[Result]:
    """Execute a scenario's runs, in the order it declared them."""
    return [execute(r) for r in runs]


def run(topology, requests, kind: str, **knobs) -> Result:
    """One ad-hoc configuration, built the same way every scenario builds one."""
    return execute(scenarios.configure(kind, topology, requests, kind, **knobs))


def run_shared_prefix(seed: int = 0) -> List[Result]:
    return results(scenarios.shared_prefix(seed))


def run_hotspot(seed: int = 0) -> List[Result]:
    return results(scenarios.hotspot(seed))


def run_overload(seed: int = 0) -> List[Result]:
    return results(scenarios.overload(seed))


def run_disaggregation(seed: int = 0) -> List[Result]:
    return results(scenarios.disaggregation(seed))


def run_early_rejection(seed: int = 0) -> List[Result]:
    return results(scenarios.early_rejection(seed))


def run_eviction_sweep(seed: int = 0):
    """``(capacity, hit_rate, fabric_bytes)`` rows, as the report reads them."""
    return [
        (int(r.label), r.ledger.hit_rate, r.ledger.fabric_bytes)
        for r in results(scenarios.eviction_sweep(seed))
    ]
