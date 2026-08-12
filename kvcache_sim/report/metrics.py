"""Outcome metrics + rendering.

Metrics are on the DES *outcome* (TTFT, cache hit rate, compute saved, fabric
bytes, rejections) -- never on wall-clock. Because the sim is deterministic (seeded
workload), two runs produce byte-identical traces and identical metrics.

:class:`Metrics` is a :class:`sim_common.report.Ledger` whose rows are
:class:`RequestResult`. Everything below is a named reading of that one row table
through the ledger's shared aggregations (sum / mean / nearest-rank percentile /
fraction) -- the arithmetic is not re-implemented per capability.

Why this is a collector, not a row a host carries
-------------------------------------------------
One request is served by two machines: one prefills it and publishes its KV, and
(under disaggregation) a different one fetches that KV back and decodes it. Each
knows a disjoint half of the outcome -- the first knows the routing decision, the
reuse and whether the publish fit; the second knows what the handoff cost and what
inter-token gaps the batch produced -- and neither can be asked about the other's.

The serving plane used to resolve that by *shipping the row*: the prefill host
built a :class:`RequestResult`, held it, and handed it to the decode host along
with the request, so whoever finished the request reported all of it. That is a
measurement object crossing a process boundary for no reason but bookkeeping, and
it is the shape that made a host need a reference to another host.

So the join lives here instead, which is where every real system puts it: hosts
emit what they observed, keyed by request id, and the collector assembles the row.
:meth:`Metrics.add` is the first writer (the prefill host, or the routing host on a
rejection) and :meth:`Metrics.handed_off` / :meth:`Metrics.decoded` are the second
(the decode host), each amending the row the first created. They fail loudly on an
id nobody opened, because a decode with no prefill is a wiring bug, not a metric.

There is a third writer, and it is not a host: :meth:`Metrics.completed` is the
**client's**, and carries the one number no host can produce. A request's end-to-end
latency spans both machines and the redirects between them, so the only participant
present at both ends of it is the thing outside the cluster that walked the chain --
which is also who a latency SLO is written for. That it can be written at all is
recent: the decode leg used to answer at admission, so by the time the last token
landed the client had long returned.

The generic event recorder lives in ``sim_common.trace.Trace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sim_common.report import Ledger

__all__ = [
    "RequestResult",
    "Metrics",
    "render_summary",
    "render_disaggregation",
    "render_early_rejection",
    "render_eviction_sweep",
    "render_hotspot",
]


@dataclass
class RequestResult:
    """Per-request outcome (one row of the metrics table).

    Written by two hosts and a client, in three passes -- see the module docstring.
    The fields down to ``published`` are the prefill host's; the three after it are
    the decode host's, and stay at their defaults for a run that does not model
    decode; ``latency`` is the client's, for the same reason it can only be the
    client's -- it spans both hosts.
    """

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
    published: bool = True           # the prefix this request computed was cached
    # -- the decode host's half -------------------------------------------- #
    decode: str = ""                 # the instance that actually decoded it
    handoff_bytes: int = 0           # KV pulled out of the store to decode here
    handoff_missed: bool = False     # ...or the chain was gone and none was
    # -- and the client's ---------------------------------------------------- #
    #: Arrival to last token, measured by the client that walked the chain. Stays
    #: 0.0 where there is no last token to wait for: a run that does not model
    #: decode ends the client's walk at prefill, so it has no end-to-end latency
    #: rather than a shorter one, and its reports do not offer the column.
    latency: float = 0.0


def _accepted(r: RequestResult) -> bool:
    return r.accepted


class Metrics(Ledger):
    """Aggregate outcome metrics for one scheduler run.

    Also the run's **collector**: the one place a request's two halves meet, since
    the host that prefilled it and the host that decoded it never speak. See the
    module docstring for why the join is here rather than in a row that travels.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # id -> the row, so a second writer finds it without rescanning the table.
        # A plain index over ``rows``, not a second copy: the values *are* the rows.
        self._by_id: Dict[str, RequestResult] = {}

    # -- recording, by whichever host observed it -------------------------- #
    def add(self, row: Any) -> None:
        """Open a request's row (the routing host, or the host that prefilled it)."""
        super().add(row)
        self._by_id[row.id] = row

    def _open(self, request_id: str) -> RequestResult:
        """The row ``request_id`` was opened with, or a loud failure.

        Loud because the alternative is a decode-side measurement quietly landing
        nowhere: every path that reaches a decode host went through a prefill host
        that opened the row first, so a miss is a wiring bug in the redirect chain
        and not a request that happens to have no prefill half.
        """
        row = self._by_id.get(request_id)
        if row is None:
            raise KeyError(
                f"no outcome row open for {request_id!r}: a decode host is "
                f"reporting on a request no prefill host recorded"
            )
        return row

    def handed_off(
        self, request_id: str, decode: str, nbytes: int, *, missed: bool = False
    ) -> None:
        """The decode host pulled this request's KV out of the store to run it.

        ``missed`` says the chain was not there to pull -- the publish did not fit,
        or a volume dropped it in between -- so nothing moved and the request is
        decoding on KV this model does not account for. Counted rather than hidden:
        it is exactly the signal that a run's cache is too small for the store to
        be a credible handoff.
        """
        row = self._open(request_id)
        row.decode = decode
        row.handoff_bytes = nbytes
        row.handoff_missed = missed

    def decoded(self, request_id: str, tbt: float) -> None:
        """The decode host emitted this request's last token at ``tbt`` worst gap."""
        self._open(request_id).tbt = tbt

    def completed(self, request_id: str, latency: float) -> None:
        """The client saw this request's last token ``latency`` after it arrived.

        The client's own stopwatch, and deliberately not a subtraction this class
        performs from timestamps the hosts reported: the interval that matters is
        the one the caller experienced, redirects and hops included, and a
        collector reassembling it from two hosts' clocks would be reporting a
        number nobody observed.

        Called only for a request that actually finished decoding, which is why
        there is no ``rejected``/``no last token`` variant: a request refused at
        either gate has no end-to-end latency, and its row is already marked
        ``accepted=False`` and excluded from every aggregate below.
        """
        self._open(request_id).latency = latency

    # -- aggregation -------------------------------------------------------- #
    @property
    def results(self) -> List[RequestResult]:
        """The per-request rows, in the order they were finalized."""
        return self.rows

    @property
    def accepted(self) -> List[RequestResult]:
        return self.select(_accepted)

    @property
    def rejections(self) -> int:
        return self.count(lambda r: not r.accepted)

    @property
    def hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache (reuse), over accepted."""
        tot = self.total("prompt_tokens", _accepted)
        cached = self.total("cached_tokens", _accepted)
        return cached / tot if tot else 0.0

    @property
    def compute_tokens(self) -> int:
        """Prompt tokens actually (re)computed by prefill."""
        return self.total("uncached_tokens", _accepted)

    @property
    def saved_tokens(self) -> int:
        """Prompt tokens avoided via prefix reuse."""
        return self.total("cached_tokens", _accepted)

    @property
    def fabric_bytes(self) -> int:
        """Cross-instance KV bytes moved to reuse remote prefixes."""
        return self.total("transfer_bytes", _accepted)

    @property
    def handoff_bytes(self) -> int:
        """KV bytes pulled out of the store by decode hosts.

        Kept apart from :attr:`fabric_bytes` because the two answer different
        questions and only one of them is a choice. A prefix pull is reuse the
        coordinator elected to buy instead of recomputing; a handoff is the price of
        decoding somewhere other than where the prompt was prefilled, which
        disaggregation pays on every single request and cannot decline.
        """
        return self.total("handoff_bytes", _accepted)

    @property
    def handoff_misses(self) -> int:
        """Handoffs that found no KV in the store (see :meth:`handed_off`)."""
        return self.count(lambda r: r.handoff_missed)

    @property
    def mean_ttft(self) -> float:
        return self.mean("ttft", _accepted)

    def pct_ttft(self, pct: float) -> float:
        """Return the ``pct`` percentile TTFT over accepted requests."""
        return self.percentile("ttft", pct, _accepted)

    @property
    def mean_tbt(self) -> float:
        """Mean worst-case time-between-tokens over accepted (decoded) requests."""
        return self.mean("tbt", _accepted)

    def pct_tbt(self, pct: float) -> float:
        """Return the ``pct`` percentile TBT over accepted requests."""
        return self.percentile("tbt", pct, _accepted)

    @property
    def mean_latency(self) -> float:
        """Mean arrival-to-last-token over accepted requests, as the client saw it.

        The measured counterpart to :attr:`mean_ttft`, which is control's
        *prediction* -- and the only column here that a cost charged between the
        two halves of a request can land in. The KV handoff is the case that
        motivated it: a real ``get_batch`` of the whole block chain, charged on
        the clock, invisible to TTFT (predicted before it happens) and to TBT
        (measured after it finishes).

        Zero for a run that does not model decode, where nothing stamps it -- see
        :attr:`RequestResult.latency`. Reading it off such a run is reading an
        absence, so the prefill-side reports do not show it.
        """
        return self.mean("latency", _accepted)

    def pct_latency(self, pct: float) -> float:
        """Return the ``pct`` percentile end-to-end latency over accepted requests."""
        return self.percentile("latency", pct, _accepted)

    def tbt_slo_met(self, slo: float) -> float:
        """Fraction of accepted requests whose TBT met ``slo`` (1.0 if none)."""
        return self.fraction(lambda r: r.tbt <= slo, _accepted)

    @property
    def wasted_prefills(self) -> int:
        """Requests whose prefill was computed, then shed at decode admission."""
        return self.count(lambda r: r.wasted_prefill)

    @property
    def decode_rejections(self) -> int:
        """Requests rejected at decode admission (TBT SLO)."""
        return self.count(lambda r: r.decode_rejected)

    @property
    def unpublished(self) -> int:
        """Accepted requests whose computed prefix did not fit in the cache.

        A cache fill is allowed to fail, so this is not an error -- but it is the
        difference between "cached" and "tried to cache and had no room", which a
        hit rate alone cannot show and a capacity sweep is measuring.
        """
        return self.count(lambda r: r.accepted and not r.published)


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
    rejects), so the story is in the TBT columns and in what each pays for them.
    Disaggregation keeps decode on its own timeline, so no prefill collides with a
    decode step; coupling lets them collide. But a decode host that did not prefill
    the prompt has to fetch its KV back out of the store, and the ``KV handoff
    bytes`` row is that bill -- the one a design that hands the request to its
    decode host as a method call does not show, and the reason the attainment
    columns are closer together than the coupling story alone would predict. The
    ``target`` is met per-request on the *worst* inter-token gap observed.

    The end-to-end rows are where that bill is paid in *time* rather than bytes,
    and they are the reason this table is the one place the column had to appear.
    Every other number here is measured inside one host: TTFT is control's
    prediction from before the handoff, TBT is the decode cadence from after it.
    Arrival to last token is the only interval that contains it -- so a run that
    moves three and a half times the KV can no longer look free.
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
        f"  {'mean end-to-end':22}{disagg.mean_latency:>15.3f}"
        f"{coupled.mean_latency:>15.3f}",
        f"  {'p90 end-to-end':22}{disagg.pct_latency(90):>15.3f}"
        f"{coupled.pct_latency(90):>15.3f}",
        f"  {'KV handoff bytes':22}{disagg.handoff_bytes:>15}"
        f"{coupled.handoff_bytes:>15}",
        f"  {'handoff misses':22}{disagg.handoff_misses:>15}"
        f"{coupled.handoff_misses:>15}",
    ])


def render_early_rejection(off: Metrics, early: Metrics, predict: Metrics,
                           slo: float) -> str:
    """Render the off/early/predict early-rejection table.

    Columns: wasted prefills (compute spent then decode-rejected), decode
    rejections, accepted (decoded) count, TBT-SLO attainment, and what the client
    actually waited. ``off`` wastes prefill; ``early``/``predict`` never do (they
    reject before prefill) -- but only ``predict`` also holds the SLO, since it
    routes decode by *predicted* load.

    End-to-end is here as well as in the disaggregation table because this
    comparison is the one where the per-token metric and the wait can disagree:
    an admission policy that spreads decode holds the inter-token gap by keeping
    batches small, which is a longer queue somewhere for somebody. The row says
    whether that trade shows up in what a caller experienced. Note what it does
    *not* average over: a rejected request has no end-to-end latency at all, so
    ``off``'s column describes the requests it kept, not the ones it shed.
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
        row("mean end-to-end", lambda m: f"{m.mean_latency:.3f}"),
    ])


def render_eviction_sweep(rows) -> str:
    """Render the capacity -> (hit rate, fabric bytes) sweep table."""
    lines = ["  %10s %12s %14s" % ("capacity", "hit_rate", "fabric_bytes")]
    for cap, hit_rate, fabric_bytes in rows:
        lines.append("  %10d %11.1f%% %14d" % (cap, 100.0 * hit_rate, fabric_bytes))
    return "\n".join(lines)


def render_hotspot(baseline: Metrics, no_repl: Metrics, repl: Metrics) -> str:
    """Render the load-balance / cache-no-repl / cache-repl comparison table."""
    def row(label: str, fmt: str, fn) -> str:
        return ("  %-26s" + fmt * 3) % (label, fn(baseline), fn(no_repl), fn(repl))

    return "\n".join([
        "  %-26s%12s%12s%12s" % ("", "load-bal", "cache/no-repl", "cache/repl"),
        row("mean TTFT", "%12.3f", lambda m: m.mean_ttft),
        row("p90 TTFT", "%12.3f", lambda m: m.pct_ttft(90)),
        row("prefix hit rate %", "%12.1f", lambda m: 100 * m.hit_rate),
        row("prefill tokens", "%12d", lambda m: m.compute_tokens),
        row("KV fabric bytes", "%12d", lambda m: m.fabric_bytes),
    ])
