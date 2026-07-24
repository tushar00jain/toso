"""Tests for the shared resource layer (network/storage contention).

These drive :class:`~sim_common.resources.ResourceRegistry` directly on a fresh
:class:`~sim_common.async_engine.AsyncEngine`, asserting the contention-model
contract (the maintainer's hard gates) on virtual-time *outcomes* -- completion
times and their ordering -- never on wall-clock timing:

* gate B -- a lone transfer under ``serialize`` AND ``progressive`` charges
  exactly the analytic ``dt`` (``latency + nbytes/capacity``);
* gate C -- ``progressive`` shares a resource so N equal, fully-overlapping
  transfers each take ~N x their solo time, and a transfer re-rates when a
  co-tenant starts/stops; ``serialize`` runs contenders back-to-back (sum);
* determinism -- same input => same completion times + same trace, per mode.

Run from the worktree with the venv interpreter::

    PYTHONPATH=. /path/to/.venv/bin/python -m pytest sim_common/tests/test_resources.py -q

Also runnable as a plain script if pytest is unavailable (no ``unittest.main``).
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

import pytest

from sim_common.async_engine import AsyncEngine
from sim_common.resources import CONTENTION_MODES, ResourceRegistry

# A convenient analytic transfer: solo dt = latency + nbytes/capacity.
CAP = 10000.0
LAT = 0.001
NB = 1000
SOLO = LAT + NB / CAP  # 0.101


def _run(mode: str, specs: List[Tuple]) -> Tuple[Dict[int, Tuple[float, float]], str]:
    """Charge each ``(key, capacity, latency, nbytes, start_delay)`` on one registry.

    Returns ``{i -> (start, end)}`` (virtual times) and the loop's trace render.
    Every spec runs as its own gathered task, optionally after a virtual
    ``start_delay`` so overlaps can be staggered deterministically.
    """
    loop = AsyncEngine()
    reg = ResourceRegistry(mode)
    out: Dict[int, Tuple[float, float]] = {}

    async def one(i: int, key, cap, lat, nb, delay) -> None:
        if delay:
            await asyncio.sleep(delay)
        start = loop.time()
        end = await reg.transfer(key, capacity=cap, latency=lat, nbytes=nb)
        out[i] = (start, end)

    async def drive() -> None:
        await asyncio.gather(*(one(i, *s) for i, s in enumerate(specs)))

    try:
        loop.run_until_complete(drive())
    finally:
        loop.close()
    return out, loop.trace.render()


def _dur(res: Dict[int, Tuple[float, float]], i: int) -> float:
    start, end = res[i]
    return end - start


# --------------------------------------------------------------------------
# Construction / validation.
# --------------------------------------------------------------------------


def test_modes_are_exactly_the_three():
    assert CONTENTION_MODES == ("none", "serialize", "progressive")


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ResourceRegistry("bogus")


# --------------------------------------------------------------------------
# Gate B: a lone transfer charges exactly the analytic dt (both real modes).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("serialize", "progressive"))
def test_lone_transfer_charges_exact_analytic_dt(mode):
    res, _ = _run(mode, [(("net", "s"), CAP, LAT, NB, 0.0)])
    # Loop starts at t=0, so the completion time IS the charged dt, exactly.
    assert _dur(res, 0) == SOLO


@pytest.mark.parametrize("mode", ("serialize", "progressive"))
def test_lone_transfer_on_distinct_resources_is_free_of_each_other(mode):
    # Two transfers on *different* resources never contend: each is solo.
    res, _ = _run(
        mode,
        [(("net", "a"), CAP, LAT, NB, 0.0), (("net", "b"), CAP, LAT, NB, 0.0)],
    )
    assert _dur(res, 0) == SOLO
    assert _dur(res, 1) == SOLO


# --------------------------------------------------------------------------
# Gate C (progressive): equal share; re-rate on co-tenant start/stop.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", (2, 3, 5))
def test_progressive_full_overlap_is_n_times_solo(n):
    # N equal transfers starting together on one resource each get capacity/N, so
    # the (dominant) byte term stretches by N; the fixed latency does not, so each
    # finishes in ~N x solo (a touch under, since latency is charged once).
    res, _ = _run("progressive", [(("net", "s"), CAP, LAT, NB, 0.0)] * n)
    expected = LAT + n * NB / CAP
    for i in range(n):
        assert _dur(res, i) == pytest.approx(expected)
    # Strictly slower than solo, and within (n-1, n] x solo (latency not scaled).
    assert n - 1 < _dur(res, 0) / SOLO <= n


def test_progressive_transfer_slows_when_a_cotenant_appears():
    # A starts alone; B joins mid-flight on the same resource -> A must re-rate
    # (slow down), so A finishes later than its solo time.
    res, _ = _run(
        "progressive",
        [(("net", "s"), CAP, LAT, NB, 0.0), (("net", "s"), CAP, LAT, NB, 0.05)],
    )
    assert _dur(res, 0) > SOLO


def test_progressive_transfer_speeds_up_when_cotenant_leaves():
    # A big transfer overlapped by a short one: once the short co-tenant finishes,
    # A re-rates back up. So A's total is less than if it had shared for its whole
    # life (2 x solo) -- it ran solo for the tail.
    big = 4000
    res, _ = _run(
        "progressive",
        [
            (("net", "s"), CAP, LAT, big, 0.0),   # long-lived
            (("net", "s"), CAP, LAT, NB, 0.0),    # short co-tenant, leaves early
        ],
    )
    solo_big = LAT + big / CAP
    shared_whole_life = LAT + 2 * big / CAP  # if it had shared the entire time
    assert solo_big < _dur(res, 0) < shared_whole_life


# --------------------------------------------------------------------------
# Gate C (serialize): contenders run back-to-back (total ~ sum of solos).
# --------------------------------------------------------------------------


def test_serialize_runs_contenders_sequentially():
    res, _ = _run("serialize", [(("net", "s"), CAP, LAT, NB, 0.0)] * 2)
    # Each runs alone at full bandwidth: first ends at SOLO, second at 2 x SOLO.
    ends = sorted(end for _s, end in res.values())
    assert ends[0] == pytest.approx(SOLO)
    assert ends[1] == pytest.approx(2 * SOLO)


def test_serialize_distinct_resources_do_not_queue():
    # Different resources are independent even under serialize.
    res, _ = _run(
        "serialize",
        [(("net", "a"), CAP, LAT, NB, 0.0), (("net", "b"), CAP, LAT, NB, 0.0)],
    )
    assert _dur(res, 0) == SOLO
    assert _dur(res, 1) == SOLO


# --------------------------------------------------------------------------
# Determinism: same input => same completion times and same trace, per mode.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("serialize", "progressive"))
def test_deterministic_across_runs(mode):
    specs = [
        (("net", "s"), CAP, LAT, NB, 0.0),
        (("net", "s"), CAP, LAT, 3 * NB, 0.02),
        (("net", "s"), CAP, LAT, 2 * NB, 0.05),
        (("store", "v", "read"), CAP, LAT, NB, 0.0),
    ]
    a_res, a_trace = _run(mode, specs)
    b_res, b_trace = _run(mode, specs)
    assert a_res == b_res
    assert a_trace == b_trace


# --------------------------------------------------------------------------
# Script fallback (no pytest required).
# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import itertools

    # Expand the parametrized cases manually for the script runner.
    cases = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = None
        for m in marks:
            if m.name == "parametrize":
                params = m.args[1]
        if params is None:
            cases.append((name, fn, ()))
        else:
            for val in params:
                cases.append((name, fn, (val,)))
    for name, fn, args in cases:
        fn(*args)
        print(f"ok  {name}{args if args else ''}")
    print(f"\n{len(cases)} tests passed")
