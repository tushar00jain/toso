"""Scenario-level tests for the network/storage contention model.

These exercise the contention feature end-to-end through the real transport seam
on the read-burst scenario, asserting the maintainer's hard gates on virtual-time
*outcomes* -- never on wall-clock timing:

* gate A -- ``contention="none"`` reproduces the historical burst exactly (the
  full suite already gates byte-identity; here we just pin that ``none`` is the
  default and that the payoff metrics are untouched);
* gate D -- the hot-source case: ``m`` readers all pull the same payload from one
  origin. Under ``serialize`` / ``progressive`` the origin's egress becomes the
  bottleneck, so the burst wallclock rises versus ``none``, while the fabric-byte
  accounting (the payoff metric) is unchanged by the timing model;
* determinism -- same input => byte-identical trace, under each mode.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest realsim/tests/test_contention.py -q
"""

from __future__ import annotations

import pytest

from realsim.harness import run_burst
from realsim.scenarios.put_get import MODE_META, MODE_METADATA
from sim_common import config

MODES = (MODE_META, MODE_METADATA)
CONTENTION = ("none", "serialize", "progressive")
PAYLOAD_BYTES = 16 * 4  # DEFAULT_N float32 elements


# --------------------------------------------------------------------------
# Gate A: "none" is the default and matches an explicit "none".
# --------------------------------------------------------------------------


def test_none_is_the_default():
    default = run_burst(num_readers=4)
    with config.overrides(contention="none"):
        explicit = run_burst(num_readers=4)
    assert default.trace.render() == explicit.trace.render()
    assert default.ledger.wallclock == explicit.ledger.wallclock


# --------------------------------------------------------------------------
# Gate D: hot source -> egress bottleneck raises wallclock; fabric unchanged.
# --------------------------------------------------------------------------


def test_hot_source_wallclock_rises_with_contention():
    m = 4
    with config.overrides(contention="none"):
        none = run_burst(num_readers=m)
    with config.overrides(contention="serialize"):
        serialize = run_burst(num_readers=m)
    with config.overrides(contention="progressive"):
        progressive = run_burst(num_readers=m)

    # Under "none" the m concurrent pulls from the single origin overlap for free
    # (each assumes the full egress bandwidth). Modeling contention on the origin's
    # egress makes them share it, so the burst takes longer.
    assert serialize.ledger.wallclock > none.ledger.wallclock
    assert progressive.ledger.wallclock > none.ledger.wallclock


@pytest.mark.parametrize("mode", CONTENTION)
def test_fabric_bytes_are_invariant_to_contention(mode):
    # The contention model changes *timing*, never *what crosses the fabric*: the
    # naive burst is m x the payload regardless of mode (the payoff metric holds).
    m = 3
    with config.overrides(contention=mode):
        res = run_burst(num_readers=m)
    assert res.ledger.origin_bytes == m * PAYLOAD_BYTES
    assert res.ledger.transfer_bytes == m * PAYLOAD_BYTES
    assert res.ledger.items_done == res.ledger.items_total == m


# --------------------------------------------------------------------------
# Determinism: same input => byte-identical trace, per mode and carrier.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", CONTENTION)
@pytest.mark.parametrize("carrier", MODES)
def test_trace_is_byte_identical_across_runs(mode, carrier):
    with config.overrides(contention=mode):
        a = run_burst(num_readers=4, mode=carrier)
        b = run_burst(num_readers=4, mode=carrier)
    assert a.trace.render() == b.trace.render()
    assert a.trace.events == b.trace.events


@pytest.mark.parametrize("mode", CONTENTION)
def test_trace_is_byte_identical_for_fixed_seed(mode):
    with config.overrides(contention=mode):
        a = run_burst(num_readers=4, random_seed=7)
        b = run_burst(num_readers=4, random_seed=7)
    assert a.trace.render() == b.trace.render()
