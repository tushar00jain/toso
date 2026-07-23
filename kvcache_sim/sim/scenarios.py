"""Scenario builders and a deterministic run harness.

Each scenario fixes a set of serving instances and a synthetic workload, runs it
under a scheduler, and returns ``(Metrics, Trace)``. Scenarios mirror the shape of
the dedup sim's toy/reshard/versioning trio, but exercise the KV-cache
value: prefix reuse, eviction, hot-block replication and overload rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import cost
from .client import Client, submit_all
from sim_common.engine import Sim
from .index import BlockIndex
from .model import Instance
from .scheduler import CacheAwareScheduler, LoadBalanceScheduler
from .trace import Metrics, Trace
from .workload import make_workload

BLOCK_TOKENS = 512
BYTES_PER_TOKEN = 1  # illustrative; keeps fabric numbers readable


def make_instances(num: int, per_node: int = 2) -> List[Instance]:
    """Build ``num`` instances laid out ``per_node`` per node (distinct hosts).

    Same-node instances exchange KV over NVLink; cross-node over RDMA -- so *where*
    a reusable prefix lives changes the cost of pulling it.
    """
    insts: List[Instance] = []
    for i in range(num):
        node = f"N{i // per_node}"
        insts.append(Instance(id=f"s{i}", host=f"h{i}", node=node))
    return insts


@dataclass
class Run:
    """Output of one scheduler run."""

    metrics: Metrics
    trace: Trace


def run(instances, requests, kind: str, *, capacity: Optional[int] = None,
        balance_threshold: float = 1.5, replicate: bool = True,
        slo_ttft: float = float("inf"), slo_tbt: float = float("inf"),
        simulate_decode: bool = False, max_batch: int = 8, coupled: bool = False,
        prefill_pool: Optional[List[str]] = None,
        decode_pool: Optional[List[str]] = None,
        early_rejection: str = "off") -> Run:
    """Run ``requests`` on ``instances`` under scheduler ``kind``.

    ``kind`` is ``"cache_aware"`` (cache-aware coordinator) or ``"load_balance"``
    (baseline). Returns metrics + trace.

    The ``simulate_decode`` group of kwargs drives the batched-decode / TBT model
    (K6): ``simulate_decode`` turns it on, ``max_batch`` is the VRAM cap on a decode
    batch, ``coupled`` shares prefill+decode compute on one timeline (vs a private
    decode timeline when disaggregated), ``prefill_pool``/``decode_pool`` split the
    instances into disjoint (or overlapping) prefill and decode roles, and
    ``early_rejection`` (``"off"``/``"early"``/``"predict"``) picks the TBT admission
    policy. All default to the pre-decode behaviour, so callers that pass none of
    them run exactly as before.
    """
    sim = Sim()
    index = BlockIndex()
    common = dict(block_tokens=BLOCK_TOKENS, bytes_per_token=BYTES_PER_TOKEN,
                  capacity=capacity, slo_ttft=slo_ttft, slo_tbt=slo_tbt,
                  simulate_decode=simulate_decode, max_batch=max_batch,
                  coupled=coupled, prefill_pool=prefill_pool,
                  decode_pool=decode_pool, early_rejection=early_rejection)
    if kind == "cache_aware":
        sched = CacheAwareScheduler(sim, index, instances,
                                    balance_threshold=balance_threshold,
                                    replicate=replicate, **common)
    elif kind == "load_balance":
        sched = LoadBalanceScheduler(sim, index, instances, **common)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown scheduler kind {kind!r}")

    trace = Trace(time_width=8, kind_width=7)
    metrics = Metrics()
    client = Client(sim, sched, BLOCK_TOKENS, trace, metrics)
    submit_all(client, requests)
    sim.run()
    return Run(metrics=metrics, trace=trace)


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #

def shared_prefix_workload(seed: int = 0):
    """Many conversations sharing a hot system prompt + per-conv context."""
    return make_workload(
        num_requests=200, num_conversations=8, system_blocks=4,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.1, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def run_shared_prefix(seed: int = 0) -> Tuple[Run, Run]:
    """Cache-aware vs load-balance on the shared-prefix workload (ample capacity)."""
    insts = make_instances(4)
    reqs = shared_prefix_workload(seed)
    cache_aware = run(insts, reqs, "cache_aware")
    baseline = run(insts, reqs, "load_balance")
    return cache_aware, baseline


def run_eviction_sweep(seed: int = 0) -> List[Tuple[int, float, int]]:
    """Sweep cache capacity; return ``(capacity, hit_rate, fabric_bytes)`` rows.

    Reproduces the expected qualitative shape: hit rate rises with capacity, then
    plateaus once the hot working set fits.
    """
    insts = make_instances(4)
    reqs = make_workload(
        num_requests=400, num_conversations=12, system_blocks=2,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.05, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )
    rows: List[Tuple[int, float, int]] = []
    for cap in (2, 4, 8, 16, 32, 64, 256):
        r = run(insts, reqs, "cache_aware", capacity=cap)
        rows.append((cap, r.metrics.hit_rate, r.metrics.fabric_bytes))
    return rows


def hotspot_workload(seed: int = 0):
    """One dominant conversation (extreme skew) -> a single hot instance."""
    return make_workload(
        num_requests=160, num_conversations=4, system_blocks=6,
        conv_base_blocks=6, query_blocks=2, zipf_s=2.2, arrival_rate=3.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )


def run_hotspot(seed: int = 0) -> Tuple[Run, Run, Run]:
    """Compare (a) baseline, (b) cache-aware no-replication, (c) cache-aware.

    With extreme skew a naive cache-aware policy that always routes to the single
    instance holding the hot prefix (balance_threshold very high => never pull to a
    peer) piles load on it. A moderate threshold replicates the hot prefix to peers
    (read-through), spreading load and cutting p90 TTFT, at the cost of a few KV
    transfers.
    """
    insts = make_instances(4)
    reqs = hotspot_workload(seed)
    baseline = run(insts, reqs, "load_balance")
    no_repl = run(insts, reqs, "cache_aware", replicate=False)
    repl = run(insts, reqs, "cache_aware", balance_threshold=1.2, replicate=True)
    return baseline, no_repl, repl


def run_overload(seed: int = 0) -> Tuple[Run, Run]:
    """High arrival rate + a TTFT SLO -> some requests must be rejected.

    Cache-aware reuse shortens prefill, freeing capacity, so it should reject fewer
    requests than the load-balancing baseline under the same SLO.
    """
    insts = make_instances(4)
    reqs = make_workload(
        num_requests=300, num_conversations=6, system_blocks=6,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=9.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )
    slo = 6.0
    cache_aware = run(insts, reqs, "cache_aware", slo_ttft=slo)
    baseline = run(insts, reqs, "load_balance", slo_ttft=slo)
    return cache_aware, baseline


# --------------------------------------------------------------------------- #
# Decode-side scenarios (K6): TBT SLO under batched decode.
# --------------------------------------------------------------------------- #

# Target TBT for the disaggregation scenario: 5 x the batch=1 baseline (TBT_BASE).
# This is a *reporting* target (fraction of served requests whose worst inter-token
# gap stayed under it) -- deliberately NOT the admission SLO. Admission is disabled
# in this scenario (slo_tbt=inf) so BOTH configs serve every request; the whole
# contrast is then attainment of this target among the served, never a rejection
# count. A healthy decode batch fits comfortably under it; a prefill colliding with
# a decode step blows straight through it.
DISAGG_TARGET_TBT = 5 * cost.TBT_BASE  # 0.100
DISAGG_MAX_BATCH = 8                   # VRAM cap on the decode batch


def run_disaggregation(seed: int = 0) -> Tuple[Run, Run]:
    """Mooncake's headline: disaggregating prefill from decode protects TBT.

    Both configs run with admission disabled (``slo_tbt=inf``, ``early_rejection=
    "off"``) so **every** request is served and gets a real measured TBT -- there is
    no overload rejection to muddy the picture. Decode capacity is held fixed (two
    instances, ``DISAGG_MAX_BATCH`` each), so the *only* difference is whether those
    two instances also do prefill. The contrast is purely the TBT-target attainment
    among served requests (:data:`DISAGG_TARGET_TBT`), which is decoupled from
    admission:

    * DISAGGREGATED -- prefill runs on ``s0``/``s1``, decode on a dedicated
      ``s2``/``s3`` pool with its *own* compute timeline. A prefill can never delay a
      decode step, so every request sees a clean, batch-sized TBT and attainment is
      ~100%.
    * COUPLED -- prefill and decode share the *same* two instances (``s2``/``s3``),
      so their compute timelines are aliased (``coupled=True``). Most requests still
      decode cleanly, but whenever a decode step happens to collide with a prefill on
      that instance, that request's *worst* inter-token gap jumps to ~a prefill time
      and it misses the target -- so a large, believable fraction of served requests
      blow the TBT target even though both configs admit the identical load.

    The arrival rate is kept low enough that the coupled pool is not saturated (no
    runaway queue): misses come from occasional prefill/decode collisions, not a
    queue explosion. Returns ``(disaggregated, coupled)``.
    """
    insts = make_instances(4)  # s0..s3
    reqs = make_workload(
        # Small prompts + short outputs keep per-request prefill/decode cheap and
        # decode-step event counts sane; the modest arrival rate keeps the coupled
        # pool busy but unsaturated, so misses are collisions, not a queue collapse.
        num_requests=120, num_conversations=8, system_blocks=2,
        conv_base_blocks=2, query_blocks=1, zipf_s=1.1, arrival_rate=1.2,
        block_tokens=BLOCK_TOKENS, output_tokens=12, seed=seed,
    )
    # slo_tbt=inf => admit_decode never rejects => all requests served & measured.
    common = dict(simulate_decode=True, slo_tbt=float("inf"),
                  early_rejection="off", max_batch=DISAGG_MAX_BATCH)
    disaggregated = run(
        insts, reqs, "cache_aware",
        prefill_pool=["s0", "s1"], decode_pool=["s2", "s3"], coupled=False,
        **common,
    )
    # Same two decode instances, now doing both jobs -> shared compute timeline.
    coupled = run([insts[2], insts[3]], reqs, "cache_aware", coupled=True,
                  **common)
    return disaggregated, coupled


# TBT SLO for the early-rejection scenario: 3 x baseline. Tight enough that a
# handful of co-located decodes overruns it, forcing the admission decision.
EARLY_SLO_TBT = 3 * cost.TBT_BASE    # 0.060
EARLY_MAX_BATCH = 8


def run_early_rejection(seed: int = 0) -> Tuple[Run, Run, Run]:
    """Mooncake's early-rejection benefit: don't spend prefill you can't decode.

    One workload + instance set, decode simulated with a tight TBT SLO under heavy
    decode load. Three ``cache_aware`` runs differ *only* in ``early_rejection``:

    * ``off``     -- no TBT gate at admission; the scheduler double-checks decode
      load only *after* prefill completes and late-rejects on a violation. Every
      such rejection is a **wasted prefill** -- compute already spent on a request
      that never decodes.
    * ``early``   -- gate at routing on the decode pool's *current* occupancy. It
      never wastes prefill (it rejects before computing), but because slow prefills
      hide the decode load that is *about* to land, the snapshot reads ~empty: it
      admits blindly and piles decode onto one instance, blowing the TBT SLO.
    * ``predict`` -- gate at routing on the decode load *predicted at prefill
      completion*, including in-flight prefills that will have landed by then. This
      is the Mooncake decode-load prediction: it wastes no prefill *and* routes
      decode by foreseen load, so it holds the TBT SLO where ``early`` cannot.

    Returns ``(off, early, predict)``.
    """
    insts = make_instances(4)
    reqs = make_workload(
        num_requests=160, num_conversations=6, system_blocks=6,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=20.0,
        block_tokens=BLOCK_TOKENS, output_tokens=32, seed=seed,
    )
    common = dict(simulate_decode=True, slo_tbt=EARLY_SLO_TBT,
                  max_batch=EARLY_MAX_BATCH)
    off = run(insts, reqs, "cache_aware", early_rejection="off", **common)
    early = run(insts, reqs, "cache_aware", early_rejection="early", **common)
    predict = run(insts, reqs, "cache_aware", early_rejection="predict", **common)
    return off, early, predict
