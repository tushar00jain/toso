"""The kvcache reports: one :class:`~realsim.run.Report` per comparison.

Thin by design. The measurements and the table formatting live in
:mod:`kvcache_sim.report.metrics`, which owns this capability's whole per-request
outcome model (``Metrics`` is the run's ``Ledger``). These classes only say
*which* runs a comparison is between, so a demo can render any of them -- and any
other capability's -- through the same :meth:`~realsim.run.Report.render`.

Every one takes the scenario's results in the order its scenario returned them,
so the positional meaning is fixed in one place: the scenario.
"""

from __future__ import annotations

from typing import Sequence

from realsim.run import Report, Result

from .metrics import (
    render_disaggregation,
    render_early_rejection,
    render_eviction_sweep,
    render_hotspot,
    render_summary,
)

__all__ = [
    "CacheVsBaselineReport",
    "DisaggregationReport",
    "EarlyRejectionReport",
    "EvictionReport",
    "HotspotReport",
]


class CacheVsBaselineReport(Report):
    """Cache-aware against the load-balancing baseline (``[aware, baseline]``)."""

    def __init__(self, name: str, results: Sequence[Result]) -> None:
        self.name = name
        self.aware, self.baseline = results

    def render(self) -> str:
        return render_summary(self.name, self.aware.ledger, self.baseline.ledger)


class EvictionReport(Report):
    """The capacity sweep: one run per capacity, labelled with it."""

    def __init__(self, results: Sequence[Result]) -> None:
        self.results = list(results)

    def render(self) -> str:
        rows = [
            (int(r.label), r.ledger.hit_rate, r.ledger.fabric_bytes)
            for r in self.results
        ]
        return render_eviction_sweep(rows)


class HotspotReport(Report):
    """``[baseline, no_replication, replication]``."""

    def __init__(self, results: Sequence[Result]) -> None:
        self.baseline, self.no_repl, self.repl = results

    def render(self) -> str:
        return render_hotspot(
            self.baseline.ledger, self.no_repl.ledger, self.repl.ledger
        )


class DisaggregationReport(Report):
    """``[disaggregated, coupled]``, against a TBT target."""

    def __init__(self, results: Sequence[Result], target_tbt: float) -> None:
        self.disagg, self.coupled = results
        self.target_tbt = target_tbt

    def render(self) -> str:
        return render_disaggregation(
            self.disagg.ledger, self.coupled.ledger, self.target_tbt
        )


class EarlyRejectionReport(Report):
    """``[off, early, predict]``, against a TBT SLO."""

    def __init__(self, results: Sequence[Result], slo_tbt: float) -> None:
        self.off, self.early, self.predict = results
        self.slo_tbt = slo_tbt

    def render(self) -> str:
        return render_early_rejection(
            self.off.ledger, self.early.ledger, self.predict.ledger, self.slo_tbt
        )
