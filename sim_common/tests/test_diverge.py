"""Tests for the trace hash-chain fingerprint and the divergence bisector.

Run from the worktree with the venv interpreter::

    PYTHONPATH=. /path/to/.venv/bin/python -m pytest sim_common/tests/test_diverge.py -q

The file is also runnable as a plain script if pytest is unavailable.
"""

from __future__ import annotations

import asyncio

from sim_common import config
from sim_common.async_engine import AsyncEngine
from sim_common.diverge import first_divergence
from sim_common.trace import _fingerprint, Trace


# --------------------------------------------------------------------------
# A tiny deterministic workload whose trace shape depends on one parameter.
# --------------------------------------------------------------------------


def _run(nap_c: float) -> Trace:
    async def workload(loop: AsyncEngine):
        async def worker(name: str, nap: float):
            loop.log("work", f"{name} start")
            await asyncio.sleep(nap)
            loop.log("work", f"{name} end")

        await asyncio.gather(
            worker("a", 3.0),
            worker("b", 1.0),
            worker("c", nap_c),
        )

    loop = AsyncEngine()
    try:
        loop.run_until_complete(workload(loop))
    finally:
        loop.close()
    return loop.trace


# --------------------------------------------------------------------------
# 1. Fingerprint: stable across identical runs, and matches the free function.
# --------------------------------------------------------------------------


def test_identical_runs_share_fingerprint_and_have_no_divergence():
    t1, t2 = _run(2.0), _run(2.0)
    assert t1.fingerprint() == t2.fingerprint()
    assert first_divergence(t1, t2) is None


def test_trace_fingerprint_matches_module_function():
    # Trace.fingerprint() folds on demand -> equal to the free fold over events.
    t = _run(2.0)
    assert t.fingerprint() == _fingerprint(t.events)


def test_trace_hash_chain_defaults_off_and_follows_config():
    # Off by default; the process config's fingerprint flag flips the default,
    # read at construction (scoped by config.overrides -> no leak).
    assert Trace().hash_chain is False
    with config.overrides(fingerprint=True):
        assert Trace().hash_chain is True
    assert Trace().hash_chain is False


def test_config_enables_incremental_chain_matching_on_demand():
    # With the config flag on, traces built inside the block maintain the chain
    # incrementally; the O(1) running digest must equal the on-demand fold.
    with config.overrides(fingerprint=True):
        t = _run(2.0)
    assert t.hash_chain is True
    assert t.fingerprint() == _fingerprint(t.events)


def test_fingerprint_differs_when_traces_differ():
    assert _run(2.0).fingerprint() != _run(2.5).fingerprint()


# --------------------------------------------------------------------------
# 2. Bisection: the first mismatching chain index is the first differing event.
# --------------------------------------------------------------------------


def test_divergence_localizes_first_differing_event():
    t1, t2 = _run(2.0), _run(2.5)  # c naps for different durations
    assert t1.fingerprint() != t2.fingerprint()

    d = first_divergence(t1, t2)
    assert d is not None
    # Everything strictly before the divergence index is byte-identical...
    assert t1.events[: d.index] == t2.events[: d.index]
    # ...and the events *at* the index actually differ (neither run ended early
    # here, both traces are the same length).
    assert t1.events[d.index] != t2.events[d.index]
    # describe() renders without error and names the divergence.
    assert "first divergence" in d.describe()


def test_divergence_reports_context_window():
    t1, t2 = _run(2.0), _run(2.5)
    d = first_divergence(t1, t2, context=2)
    assert d is not None
    assert len(d.context) <= 2
    # The context is the shared prefix immediately before the divergence.
    assert d.context == t1.events[max(0, d.index - 2) : d.index]


def test_prefix_run_reports_the_trailing_event():
    # Runs that agree everywhere but where one has extra trailing events
    # diverge at the first extra event, with the shorter run reporting None.
    a, b = Trace(), Trace()
    shared = [(0.0, "x", "1"), (1.0, "x", "2")]
    for e in shared:
        a.record(*e)
        b.record(*e)
    a.record(2.0, "x", "3")  # only run A has this event

    d = first_divergence(a, b)
    assert d is not None
    assert d.index == 2
    assert d.a_event == (2.0, "x", "3")
    assert d.b_event is None


# --------------------------------------------------------------------------
# Script fallback (no pytest required).
# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
