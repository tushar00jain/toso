"""Tests for :class:`sim_common.report.Ledger` -- the shared measurement half.

The ledger replaces two hand-rolled accounting objects (a burst's fabric counters
and the KV-cache request table), so what is asserted here is exactly the
behaviour both of them had:

* ``get`` transfers are counted, ``put`` transfers are not, and only a transfer
  sourced at an *origin* counts as a fabric byte;
* a zero-byte transfer records no edge (it would draw a phantom arrow);
* the aggregation helpers (sum / mean / nearest-rank percentile / fraction)
  reduce arbitrary row types by attribute name, and answer the documented
  neutral value on an empty selection.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest sim_common/tests/test_report.py -q
"""

from __future__ import annotations

from dataclasses import dataclass

from sim_common.report import Ledger, Outcome, percentile, render_tree


@dataclass
class _Row:
    """A stand-in for a capability's own outcome row."""

    id: str
    ok: bool
    value: float


def _ledger_with_rows() -> Ledger:
    ledger = Ledger()
    ledger.add(_Row("a", True, 1.0))
    ledger.add(_Row("b", True, 3.0))
    ledger.add(_Row("c", False, 100.0))
    return ledger


# --------------------------------------------------------------------------
# Transfer accounting.
# --------------------------------------------------------------------------


def test_only_gets_are_counted():
    ledger = Ledger(origins={"volp"})
    ledger.record_transfer("put", "volr0", "volp", 64, 0.1)
    assert ledger.transfer_bytes == 0
    assert ledger.origin_bytes == 0
    assert ledger.edges == []


def test_origin_sourced_gets_are_fabric_bytes():
    ledger = Ledger(origins={"volp"})
    ledger.record_transfer("get", "volp", "volr0", 64, 0.1)
    ledger.record_transfer("get", "volr0", "volr1", 64, 0.1)  # peer -> not fabric
    assert ledger.transfer_bytes == 128
    assert ledger.origin_bytes == 64
    assert ledger.edges == [("volp", "volr0", "volr0"), ("volr0", "volr1", "volr1")]
    # The edges are exactly what the shared tree renderer consumes.
    assert render_tree(ledger.edges) == ["volp ──▶ volr0 ──▶ volr1"]


def test_zero_byte_transfer_records_no_edge():
    ledger = Ledger(origins={"volp"})
    ledger.record_transfer("get", "volp", "volp", 0, 0.0)
    assert ledger.transfer_bytes == 0
    assert ledger.edges == []


# --------------------------------------------------------------------------
# Rows + aggregation.
# --------------------------------------------------------------------------


def test_rows_and_items_done():
    ledger = Ledger()
    ledger.items_total = 2
    ledger.add(Outcome(id="r0", released=0.0, done=1.5))
    assert ledger.items_done == 1
    assert ledger.items_total == 2
    assert ledger.rows[0].done == 1.5


def test_total_mean_and_count_respect_the_selector():
    ledger = _ledger_with_rows()
    ok = lambda r: r.ok  # noqa: E731 - a one-expression selector reads better inline
    assert ledger.total("value") == 104.0
    assert ledger.total("value", ok) == 4.0
    assert ledger.mean("value", ok) == 2.0
    assert ledger.count(lambda r: not r.ok) == 1


def test_empty_selection_gives_the_documented_neutral_value():
    ledger = Ledger()
    assert ledger.total("value") == 0
    assert ledger.mean("value") == 0.0
    assert ledger.percentile("value", 90) == 0.0
    # A fraction over nothing is 1.0 ("no row missed the target"), matching the
    # SLO-attainment reading the KV-cache report has always used.
    assert ledger.fraction(lambda r: True) == 1.0


def test_percentile_is_nearest_rank_and_clamped():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0    # index int(0.5 * 4) == 2
    assert percentile(values, 90) == 4.0    # index int(3.6) == 3
    assert percentile(values, 100) == 4.0   # clamped to the last element
    assert percentile([], 90) == 0.0
    ledger = _ledger_with_rows()
    assert ledger.percentile("value", 90, lambda r: r.ok) == 3.0


def test_fraction_over_a_subset():
    ledger = _ledger_with_rows()
    # Two of the three rows are ok; of those, one has value <= 1.0.
    assert ledger.fraction(lambda r: r.value <= 1.0, lambda r: r.ok) == 0.5
