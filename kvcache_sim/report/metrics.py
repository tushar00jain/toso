"""Outcome metrics + rendering.

Metrics are on the DES *outcome* (TTFT, cache hit rate, compute saved, fabric
bytes, rejections) -- never on wall-clock. Because the sim is deterministic (seeded
workload), two runs produce byte-identical traces and identical metrics.

The generic event recorder lives in ``sim_common.trace.Trace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sim_common.trace import Trace  # noqa: F401  (re-exported for scenarios)


@dataclass
class RequestResult:
    """Per-request outcome (one row of the metrics table)."""

    id: str
    accepted: bool
    ttft: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    uncached_tokens: int = 0
    transfer_bytes: int = 0
    prefill: str = ""
    reuse_source: Optional[str] = None
    tbt: float = 0.0                  # worst inter-token gap observed in decode
    decode_rejected: bool = False    # shed at decode admission (TBT SLO)
    wasted_prefill: bool = False     # prefill was spent, then decode rejected


@dataclass
class Metrics:
    """Aggregate outcome metrics for one scheduler run."""

    results: List[RequestResult] = field(default_factory=list)

    def add(self, r: RequestResult) -> None:
        self.results.append(r)

    @property
    def accepted(self) -> List[RequestResult]:
        return [r for r in self.results if r.accepted]

    @property
    def rejections(self) -> int:
        return sum(1 for r in self.results if not r.accepted)

    @property
    def hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache (reuse), over accepted."""
        tot = sum(r.prompt_tokens for r in self.accepted)
        cached = sum(r.cached_tokens for r in self.accepted)
        return cached / tot if tot else 0.0

    @property
    def compute_tokens(self) -> int:
        """Prompt tokens actually (re)computed by prefill."""
        return sum(r.uncached_tokens for r in self.accepted)

    @property
    def saved_tokens(self) -> int:
        """Prompt tokens avoided via prefix reuse."""
        return sum(r.cached_tokens for r in self.accepted)

    @property
    def fabric_bytes(self) -> int:
        """Cross-instance KV bytes moved to reuse remote prefixes."""
        return sum(r.transfer_bytes for r in self.accepted)

    @property
    def mean_ttft(self) -> float:
        acc = self.accepted
        return sum(r.ttft for r in acc) / len(acc) if acc else 0.0

    def pct_ttft(self, pct: float) -> float:
        """Return the ``pct`` percentile TTFT over accepted requests."""
        acc = sorted(r.ttft for r in self.accepted)
        if not acc:
            return 0.0
        idx = min(len(acc) - 1, int(pct / 100.0 * len(acc)))
        return acc[idx]

    @property
    def mean_tbt(self) -> float:
        """Mean worst-case time-between-tokens over accepted (decoded) requests."""
        acc = self.accepted
        return sum(r.tbt for r in acc) / len(acc) if acc else 0.0

    def pct_tbt(self, pct: float) -> float:
        """Return the ``pct`` percentile TBT over accepted requests."""
        acc = sorted(r.tbt for r in self.accepted)
        if not acc:
            return 0.0
        idx = min(len(acc) - 1, int(pct / 100.0 * len(acc)))
        return acc[idx]

    def tbt_slo_met(self, slo: float) -> float:
        """Fraction of accepted requests whose TBT met ``slo`` (1.0 if none)."""
        acc = self.accepted
        if not acc:
            return 1.0
        return sum(1 for r in acc if r.tbt <= slo) / len(acc)

    @property
    def wasted_prefills(self) -> int:
        """Requests whose prefill was computed, then shed at decode admission."""
        return sum(1 for r in self.results if r.wasted_prefill)

    @property
    def decode_rejections(self) -> int:
        """Requests rejected at decode admission (TBT SLO)."""
        return sum(1 for r in self.results if r.decode_rejected)


def render_summary(name: str, cache_aware: Metrics, baseline: Metrics) -> str:
    """Render a cache-aware vs baseline comparison block."""
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    lines = [
        f"scenario: {name}",
        f"  {'':22}{'cache-aware':>14}{'load-balance':>14}",
        f"  {'prefix hit rate':22}{pct(cache_aware.hit_rate):>14}"
        f"{pct(baseline.hit_rate):>14}",
        f"  {'prefill tokens':22}{cache_aware.compute_tokens:>14}"
        f"{baseline.compute_tokens:>14}",
        f"  {'tokens saved (reuse)':22}{cache_aware.saved_tokens:>14}"
        f"{baseline.saved_tokens:>14}",
        f"  {'mean TTFT':22}{cache_aware.mean_ttft:>14.3f}"
        f"{baseline.mean_ttft:>14.3f}",
        f"  {'p90 TTFT':22}{cache_aware.pct_ttft(90):>14.3f}"
        f"{baseline.pct_ttft(90):>14.3f}",
        f"  {'KV fabric bytes':22}{cache_aware.fabric_bytes:>14}"
        f"{baseline.fabric_bytes:>14}",
        f"  {'rejections':22}{cache_aware.rejections:>14}{baseline.rejections:>14}",
    ]
    return "\n".join(lines)


def render_disaggregation(disagg: Metrics, coupled: Metrics,
                          target: float) -> str:
    """Render the coupled-vs-disaggregated TBT comparison block.

    Both configs serve the same load (the ``served/total`` row makes clear neither
    rejects), so the story is entirely in the TBT columns: disaggregation keeps
    decode on its own timeline (batch-sized TBT, target held ~100%), coupling lets
    prefills collide with decode steps (a fraction of served requests miss). The
    ``target`` is met per-request on the *worst* inter-token gap observed.
    """
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    def served(m: Metrics) -> str:
        return f"{len(m.accepted)}/{len(m.results)}"

    return "\n".join([
        f"  TBT target = {target:.3f} (5x the batch=1 baseline; worst per-request gap)",
        f"  {'':22}{'disaggregated':>15}{'coupled':>15}",
        f"  {'served / total':22}{served(disagg):>15}{served(coupled):>15}",
        f"  {'TBT target attainment':22}{pct(disagg.tbt_slo_met(target)):>15}"
        f"{pct(coupled.tbt_slo_met(target)):>15}",
        f"  {'mean TBT':22}{disagg.mean_tbt:>15.3f}{coupled.mean_tbt:>15.3f}",
        f"  {'p90 TBT':22}{disagg.pct_tbt(90):>15.3f}{coupled.pct_tbt(90):>15.3f}",
    ])


def render_early_rejection(off: Metrics, early: Metrics, predict: Metrics,
                           slo: float) -> str:
    """Render the off/early/predict early-rejection table.

    Columns: wasted prefills (compute spent then decode-rejected), decode
    rejections, accepted (decoded) count, and TBT-SLO attainment. ``off`` wastes
    prefill; ``early``/``predict`` never do (they reject before prefill) -- but only
    ``predict`` also holds the SLO, since it routes decode by *predicted* load.
    """
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    def row(label: str, fn) -> str:
        return (f"  {label:22}{fn(off):>12}{fn(early):>12}{fn(predict):>12}")

    return "\n".join([
        f"  TBT SLO = {slo:.3f} (3x the batch=1 baseline)",
        f"  {'':22}{'off':>12}{'early':>12}{'predict':>12}",
        row("wasted prefills", lambda m: m.wasted_prefills),
        row("decode rejections", lambda m: m.decode_rejections),
        row("accepted (decoded)", lambda m: len(m.accepted)),
        row("TBT SLO attainment", lambda m: pct(m.tbt_slo_met(slo))),
    ])
