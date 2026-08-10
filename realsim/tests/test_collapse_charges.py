"""Scenario-level tests for the ``collapse_charges`` transport flag.

These exercise the collapse feature end-to-end through the real transport seam,
asserting the maintainer's hard gates on virtual-time *outcomes* -- never on
wall-clock timing:

* **off (default)** -- ``collapse_charges=False`` is the default and is
  byte-identical to today's per-component behavior (the full suite already gates
  byte-identity; here we pin the default + explicit-False equivalence);
* **bounce reduction** -- with ``collapse_charges=True`` on the non-contended path
  a get suspends the coroutine ONCE (one transport ``_sleep``) instead of three,
  and a put once instead of twice; we instrument the transport ``_sleep`` count
  and assert the drop;
* **time-total preserved** -- the combined sleep advances the clock by exactly the
  sum of the component costs, so an isolated request's completion time is
  unchanged versus collapse-off;
* **payoff metrics intact** -- the dedup 1x fabric result is untouched by collapse
  (a metric that does not depend on sub-charge interleaving);
* **composition** -- collapse is inert under a contention model
  (``serialize`` / ``progressive``): the run is byte-identical to contention-only
  and stays deterministic.

The ``collapse_charges`` / ``contention`` modes are run-wide config, read
ambiently (see :mod:`sim_common.config`), so tests select them with the scoped
:func:`sim_common.config.overrides` context manager -- not a scenario argument --
via the :func:`_run` helper below.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest realsim/tests/test_collapse_charges.py -q
"""

from __future__ import annotations

import pytest

import realsim.seams.transport as transport_mod
from putget_sim.harness import run_burst
from putget_sim.workload.put_get import MODE_META, MODE_METADATA
from sim_common import config

MODES = (MODE_META, MODE_METADATA)
CONTENTION = ("none", "serialize", "progressive")
PAYLOAD_BYTES = 16 * 4  # DEFAULT_N float32 elements


def _run(*, collapse_charges: bool = False, contention: str = "none", **run_kwargs):
    """Run a burst with the run-wide ``collapse_charges`` / ``contention`` config
    set ambiently for the duration of the call (scoped + auto-restored), then
    pass the remaining scenario arguments straight to :func:`run_burst`.
    """
    with config.overrides(collapse_charges=collapse_charges, contention=contention):
        return run_burst(**run_kwargs)


def _count_transport_sleeps(**run_kwargs) -> int:
    """Run a burst and count how many times the transport suspends via ``_sleep``.

    Wraps :meth:`InMemoryTransport._sleep` (the single point every non-contended
    charge advances the virtual clock through) with a counter, restored in a
    ``finally`` so the wrap never leaks into other tests.
    """
    original = transport_mod.InMemoryTransport._sleep
    count = 0

    async def counting_sleep(self, dt):  # noqa: ANN001 - test shim
        nonlocal count
        count += 1
        return await original(self, dt)

    transport_mod.InMemoryTransport._sleep = counting_sleep
    try:
        _run(**run_kwargs)
    finally:
        transport_mod.InMemoryTransport._sleep = original
    return count


# --------------------------------------------------------------------------
# Off (default): collapse_charges=False is the default and byte-identical.
# --------------------------------------------------------------------------


def test_off_is_the_default():
    default = run_burst(num_readers=4)  # pure ambient default (no override)
    explicit = _run(num_readers=4, collapse_charges=False)
    assert default.trace.render() == explicit.trace.render()
    assert default.ledger.wallclock == explicit.ledger.wallclock


@pytest.mark.parametrize("carrier", MODES)
def test_collapse_changes_the_trace_but_not_the_metrics(carrier):
    # Collapse is a deliberate coarsening: the intermediate sub-charge instants
    # vanish, so the trace is NOT byte-identical -- but no fabric-byte metric
    # moves (accounting is emitted per component regardless).
    off = _run(num_readers=3, mode=carrier, collapse_charges=False)
    on = _run(num_readers=3, mode=carrier, collapse_charges=True)
    assert off.trace.render() != on.trace.render()
    assert on.ledger.origin_bytes == off.ledger.origin_bytes
    assert on.ledger.transfer_bytes == off.ledger.transfer_bytes
    assert on.ledger.items_done == off.ledger.items_done


# --------------------------------------------------------------------------
# Bounce reduction: a get suspends once (not three), a put once (not two).
# --------------------------------------------------------------------------


def test_bounce_count_falls_with_collapse():
    # A burst does 1 producer put + m reader gets. Transport suspends decompose as
    # ``put_sleeps + m*get_sleeps``, so measuring two reader counts isolates the
    # per-op bounce count without hooking op boundaries.
    off_1 = _count_transport_sleeps(num_readers=1, collapse_charges=False)
    off_3 = _count_transport_sleeps(num_readers=3, collapse_charges=False)
    on_1 = _count_transport_sleeps(num_readers=1, collapse_charges=True)
    on_3 = _count_transport_sleeps(num_readers=3, collapse_charges=True)

    off_get = (off_3 - off_1) // 2
    off_put = off_1 - off_get
    on_get = (on_3 - on_1) // 2
    on_put = on_1 - on_get

    # Off: a get is storage-read + mem + network (3 suspends); a put is
    # network + storage-write (2 suspends).
    assert (off_get, off_put) == (3, 2)
    # On: each op collapses to a single combined suspend.
    assert (on_get, on_put) == (1, 1)


# --------------------------------------------------------------------------
# Time-total preserved: an isolated request's completion time is unchanged.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", MODES)
def test_isolated_wallclock_is_preserved(carrier):
    # With a single reader there is no concurrency, so no coroutine could ever
    # interleave between the sub-charges: the combined sleep must land the request
    # at exactly the same virtual instant as the three separate sleeps did.
    off = _run(num_readers=1, mode=carrier, collapse_charges=False)
    on = _run(num_readers=1, mode=carrier, collapse_charges=True)
    assert on.ledger.wallclock == off.ledger.wallclock


# --------------------------------------------------------------------------
# Payoff metrics intact + determinism.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", MODES)
def test_fabric_bytes_invariant_to_collapse(carrier):
    m = 3
    on = _run(num_readers=m, mode=carrier, collapse_charges=True)
    assert on.ledger.origin_bytes == m * PAYLOAD_BYTES
    assert on.ledger.transfer_bytes == m * PAYLOAD_BYTES


@pytest.mark.parametrize("carrier", MODES)
def test_trace_is_byte_identical_across_runs(carrier):
    a = _run(num_readers=4, mode=carrier, collapse_charges=True)
    b = _run(num_readers=4, mode=carrier, collapse_charges=True)
    assert a.trace.render() == b.trace.render()
    assert a.trace.events == b.trace.events


# --------------------------------------------------------------------------
# Composition: collapse is inert under a contention model.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("serialize", "progressive"))
def test_collapse_is_inert_under_contention(mode):
    # Under a contention model each component occupies a different shared resource
    # and must be tracked separately, so collapse cannot merge them: the run is
    # byte-identical to contention-only, and every metric matches.
    contention_only = _run(num_readers=4, contention=mode, collapse_charges=False)
    with_collapse = _run(num_readers=4, contention=mode, collapse_charges=True)
    assert with_collapse.trace.render() == contention_only.trace.render()
    assert with_collapse.ledger.wallclock == contention_only.ledger.wallclock
    assert with_collapse.ledger.origin_bytes == contention_only.ledger.origin_bytes


@pytest.mark.parametrize("mode", ("serialize", "progressive"))
def test_collapse_under_contention_is_deterministic(mode):
    a = _run(num_readers=4, contention=mode, collapse_charges=True)
    b = _run(num_readers=4, contention=mode, collapse_charges=True)
    assert a.trace.render() == b.trace.render()
