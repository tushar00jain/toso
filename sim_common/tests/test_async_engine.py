"""Deterministic tests for the virtual-clock asyncio engine (design doc 4.1).

Run from the worktree with the venv interpreter::

    PYTHONPATH=. /path/to/.venv/bin/python -m pytest sim_common/tests/test_async_engine.py -q

The file is also runnable as a plain script (``python sim_common/tests/...``)
if pytest is unavailable. Assertions are on DES *outcomes* -- virtual time,
completion order, byte-identical traces -- never on wall-clock timing (except
the one bound proving ``sleep`` does not actually block).
"""

from __future__ import annotations

import asyncio
import time as _wallclock

from sim_common.async_engine import AsyncEngine, run_sim


# --------------------------------------------------------------------------
# Minimal workloads (in-memory awaitables only; no torchstore, no real I/O).
# --------------------------------------------------------------------------


async def _worker(loop: AsyncEngine, name: str, nap: float) -> float:
    """Sleep on the virtual clock, log around it, return the finish time."""
    loop.log("work", f"{name} start")
    await asyncio.sleep(nap)
    loop.log("work", f"{name} end")
    return loop.time()


async def _toy_workload(loop: AsyncEngine) -> list[float]:
    """A small fan-out with staggered virtual sleeps -> a stable trace."""
    results = await asyncio.gather(
        _worker(loop, "a", 3.0),
        _worker(loop, "b", 1.0),
        _worker(loop, "c", 2.0),
    )
    return results


def _run_toy(random_seed=None):
    loop = AsyncEngine(random_seed=random_seed)
    try:
        result = loop.run_until_complete(_toy_workload(loop))
    finally:
        loop.close()
    return result, loop.trace.render()


# --------------------------------------------------------------------------
# 1. Determinism: same workload -> byte-identical trace across two runs.
# --------------------------------------------------------------------------


def test_trace_deterministic_across_runs():
    _r1, trace1 = _run_toy()
    _r2, trace2 = _run_toy()
    assert trace1 == trace2
    # And a non-trivial trace (not accidentally empty).
    assert "task" in trace1 and "clock" in trace1


def test_seeded_random_mode_is_deterministic_per_seed():
    # Random ready-queue selection is still reproducible for a fixed seed.
    _r1, a = _run_toy(random_seed=1234)
    _r2, b = _run_toy(random_seed=1234)
    assert a == b


# --------------------------------------------------------------------------
# 2. Virtual time: sleep(10) advances the clock by 10 and returns immediately
#    in wall-clock (no real blocking).
# --------------------------------------------------------------------------


def test_sleep_advances_virtual_clock_not_wall_clock():
    async def nap(loop):
        assert loop.time() == 0.0
        await asyncio.sleep(10.0)
        return loop.time()

    loop = AsyncEngine()
    wall_start = _wallclock.perf_counter()
    try:
        virtual_end = loop.run_until_complete(nap(loop))
    finally:
        loop.close()
    wall_elapsed = _wallclock.perf_counter() - wall_start

    assert virtual_end == 10.0            # simulated time advanced by exactly 10
    assert wall_elapsed < 1.0             # but ~0 real seconds elapsed


async def _chain():
    loop = asyncio.get_running_loop()
    await asyncio.sleep(2.0)
    await asyncio.sleep(3.0)
    await asyncio.sleep(5.0)
    return loop.time()


def test_nested_sleeps_accumulate_on_the_virtual_clock():
    result, _trace = run_sim(_chain())
    assert result == 10.0


# --------------------------------------------------------------------------
# 3. Gather ordering: a fan-out of N in-memory awaitables completes in a
#    defined, reproducible order.
# --------------------------------------------------------------------------


def test_gather_returns_results_in_argument_order():
    # asyncio.gather preserves *argument* order in its result regardless of
    # completion order; results are the workers' finish times.
    results, _trace = _run_toy()
    assert results == [3.0, 1.0, 2.0]  # a napped 3, b napped 1, c napped 2


def test_gather_completion_order_is_by_virtual_time():
    # Completion order (as recorded in the trace) is defined by the virtual
    # clock: shortest sleep finishes first. Ties break FIFO by insertion.
    _results, trace = _run_toy()
    lines = trace.splitlines()
    ends = [ln for ln in lines if "end" in ln]
    # b (nap=1) < c (nap=2) < a (nap=3)
    assert [ln.split()[-2] for ln in ends] == ["b", "c", "a"]


def test_simultaneous_timers_fire_in_scheduling_order():
    # Timers due at the same virtual instant must fire strictly FIFO by
    # scheduling order (the `_SeqTimerHandle` (time, seq) tiebreak), not in
    # heapq-structural order. Mixing two fire times with interleaved insertion
    # exercises the heap reordering the tiebreak fixes: c5/b5/d5 are all due at
    # t=5 but inserted with a1/e1 (t=1) between them.
    fired: list[str] = []

    async def _drain(loop):
        await asyncio.sleep(6.0)  # keep the loop alive past every timer

    loop = AsyncEngine()
    try:
        loop.call_at(5.0, lambda: fired.append("c5"))  # seq 0
        loop.call_at(1.0, lambda: fired.append("a1"))  # seq 1
        loop.call_at(5.0, lambda: fired.append("b5"))  # seq 2
        loop.call_at(5.0, lambda: fired.append("d5"))  # seq 3
        loop.call_at(1.0, lambda: fired.append("e1"))  # seq 4
        loop.run_until_complete(_drain(loop))
    finally:
        loop.close()

    # t=1 group first (FIFO: a1 before e1), then the t=5 group (FIFO: c5, b5, d5).
    assert fired == ["a1", "e1", "c5", "b5", "d5"]


def test_simultaneous_sleepers_resume_in_creation_order():
    # The same guarantee, one level up: coroutines that sleep to the same wake
    # time resume in the order their sleeps were scheduled.
    async def scenario(loop):
        order: list[str] = []

        async def sleeper(name: str):
            await asyncio.sleep(4.0)
            order.append(name)

        # gather starts the coroutines in argument order; each schedules its
        # timer for t=4 in that order, so they must resume a, b, c, d, e.
        await asyncio.gather(*(sleeper(n) for n in ["a", "b", "c", "d", "e"]))
        return order

    loop = AsyncEngine()
    try:
        order = loop.run_until_complete(scenario(loop))
    finally:
        loop.close()
    assert order == ["a", "b", "c", "d", "e"]


def test_gather_over_in_memory_futures_completes_deterministically():
    # Fan-out over plain loop futures resolved in a fixed order -> defined
    # gather result order.
    async def scenario(loop):
        futures = [loop.create_future() for _ in range(5)]

        async def resolver():
            # Resolve in reverse index order at staggered virtual times.
            for i in reversed(range(5)):
                await asyncio.sleep(1.0)
                futures[i].set_result(i)

        loop.create_task(resolver())
        return await asyncio.gather(*futures)

    loop = AsyncEngine()
    try:
        out = loop.run_until_complete(scenario(loop))
    finally:
        loop.close()
    # gather preserves argument order in the returned list...
    assert out == [0, 1, 2, 3, 4]
    # ...and the whole thing took 5 virtual seconds.
    assert loop.time() == 5.0


# --------------------------------------------------------------------------
# Script fallback (no pytest required).
# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
