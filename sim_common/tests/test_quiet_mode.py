"""Tests for quiet mode: opting out of all per-event trace bookkeeping.

Quiet mode makes every :meth:`sim_common.trace.Trace.record` a no-op and drops
the engine's per-task finish-callback, so a large run pays none of the per-event
string-format / list-growth cost. It may ONLY remove trace side effects -- the
virtual clock, event ordering, task naming, scheduling and every returned
metric must stay byte-identical to a traced run.

Run from the worktree with the venv interpreter::

    PYTHONPATH=. /path/to/.venv/bin/python -m pytest sim_common/tests/test_quiet_mode.py -q

Also runnable as a plain script if pytest is unavailable. Tests that mutate the
global config restore it in a ``finally`` so the suite stays order-independent.
"""

from __future__ import annotations

import asyncio

from sim_common import config
from sim_common.async_engine import AsyncEngine, run_sim
from sim_common.trace import Trace


# --------------------------------------------------------------------------
# A small representative scenario: a gather fan-out with staggered sleeps plus
# scenario-level `loop.log` rows, so both the engine trace and scenario trace
# are exercised.
# --------------------------------------------------------------------------


async def _worker(loop: AsyncEngine, name: str, nap: float) -> float:
    loop.log("work", f"{name} start")
    await asyncio.sleep(nap)
    loop.log("work", f"{name} end")
    return loop.time()


async def _scenario(loop: AsyncEngine) -> list[float]:
    return await asyncio.gather(
        _worker(loop, "a", 3.0),
        _worker(loop, "b", 1.0),
        _worker(loop, "c", 2.0),
    )


def _run(quiet: bool):
    """Run the scenario once, returning (results, final_time, task_seq, trace)."""
    loop = AsyncEngine(quiet=quiet)
    try:
        results = loop.run_until_complete(_scenario(loop))
        return results, loop.time(), loop._task_seq, loop.trace
    finally:
        loop.close()


def _reset() -> None:
    config.configure()


# --------------------------------------------------------------------------
# (a) A quiet run records nothing.
# --------------------------------------------------------------------------


def test_quiet_run_has_empty_trace():
    _results, _t, _seq, trace = _run(quiet=True)
    assert trace.events == []
    assert trace.render() == ""


def test_disabled_trace_record_is_a_noop():
    # The one-object mechanism: a disabled Trace ignores record everywhere.
    t = Trace(enabled=False)
    t.record(1.0, "kind", "msg")
    assert t.events == []
    assert t.render() == ""
    # render()/fingerprint() still work (on the empty list).
    assert t.fingerprint() == Trace(enabled=True).fingerprint()


# --------------------------------------------------------------------------
# (b) Determinism: quiet and traced runs agree on every measured value.
# --------------------------------------------------------------------------


def test_quiet_matches_traced_metrics_and_final_time():
    traced_results, traced_time, traced_seq, traced_trace = _run(quiet=False)
    quiet_results, quiet_time, quiet_seq, _quiet_trace = _run(quiet=True)

    # Identical returned metrics and identical final simulated time.
    assert quiet_results == traced_results
    assert quiet_time == traced_time
    # Task naming still advances identically (names never depend on tracing).
    assert quiet_seq == traced_seq
    # And the traced run really did record something (guard against both empty).
    assert traced_trace.events != []


# --------------------------------------------------------------------------
# (c) The config flag flips the default; an explicit constructor arg overrides.
# --------------------------------------------------------------------------


def test_config_flag_flips_default_and_arg_overrides():
    _reset()
    try:
        # Default: tracing on.
        loop = AsyncEngine()
        try:
            assert loop.trace.enabled is True
        finally:
            loop.close()

        # config trace=False makes quiet the ambient default...
        with config.overrides(trace=False):
            loop = AsyncEngine()
            try:
                assert loop.trace.enabled is False
            finally:
                loop.close()
            # ...but an explicit quiet=False constructor arg overrides it.
            loop = AsyncEngine(quiet=False)
            try:
                assert loop.trace.enabled is True
            finally:
                loop.close()

        # Symmetrically, an explicit quiet=True overrides an on config.
        loop = AsyncEngine(quiet=True)
        try:
            assert loop.trace.enabled is False
        finally:
            loop.close()
    finally:
        _reset()


def test_run_sim_quiet_arg():
    result, trace = run_sim(_scenario_via_loop(), quiet=True)
    assert trace.events == []
    assert result == [3.0, 1.0, 2.0]


async def _scenario_via_loop() -> list[float]:
    # run_sim owns the loop, so reach it through the running-loop accessor.
    loop = asyncio.get_running_loop()
    return await _scenario(loop)


# --------------------------------------------------------------------------
# Script fallback (no pytest required).
# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
