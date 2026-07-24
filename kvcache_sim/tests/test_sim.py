"""Deterministic tests for the KV-cache simulation on the *real* TorchStore.

Run from the parent directory: ``python -m pytest kvcache_sim/tests -q``.
Assertions are on the simulation *outcome* (block presence in the real directory,
prefix hit rate, prefill compute, eviction, rejections, TBT, determinism) -- never
on wall-clock timing. Every scenario drives the real ``Controller`` directory + real
per-instance clients on the shared deterministic async engine.
"""

from __future__ import annotations

from sim_common import config
from sim_common.async_engine import AsyncEngine, run_sim

from kvcache_sim.sim.cache import LRUCache
from kvcache_sim.sim.cluster import Cluster
from kvcache_sim.sim.cost import decode_step_time, PROFILE
from kvcache_sim.sim.decode import DecodeEngine
from kvcache_sim.sim.model import (
    block_keys_for,
    longest_prefix_run,
    Request,
)
from kvcache_sim.sim.scenarios import (
    DISAGG_TARGET_TBT,
    EARLY_SLO_TBT,
    make_topology,
    run,
    run_disaggregation,
    run_early_rejection,
    run_eviction_sweep,
    run_hotspot,
    run_overload,
    run_shared_prefix,
    shared_prefix_workload,
)


# 1. Prefix-hash addressing: shared prefixes yield shared keys.
def test_prefix_hash_chain_shares_prefix():
    a = block_keys_for("m0", [0, 1, 7, 9])
    b = block_keys_for("m0", [0, 1, 8])
    assert a[:2] == b[:2]         # shared leading segments -> identical keys
    assert a[2] != b[2]           # divergence -> distinct keys
    # different model never aliases:
    assert block_keys_for("m1", [0, 1])[0] != block_keys_for("m0", [0, 1])[0]


def test_longest_prefix_run():
    keys = block_keys_for("m0", [0, 1, 2, 3])
    present = {keys[0], keys[1], keys[3]}   # a gap at index 2
    assert longest_prefix_run(keys, present) == 2   # stops at first miss


# 2. REAL directory: per-instance prefix-match length, incl. after eviction.
#    Publishing records block->volume presence in the real Controller directory;
#    locate_volumes reads it back; eviction removes it. This is the cache-aware
#    scheduler's core query, answered directly by the real directory.
def test_real_directory_prefix_presence_and_eviction():
    topo = make_topology(2)
    keys = block_keys_for("m0", [0, 1, 2, 3])

    async def scenario():
        cl = Cluster(topo, block_tokens=512)
        with cl.installed():
            await cl.publish("s0", list(keys[:3]))   # s0 holds 3 leading blocks
            await cl.publish("s1", list(keys[:1]))   # s1 holds 1
            counts = await cl.prefix_lengths(list(keys))
            assert counts == {"s0": 3, "s1": 1}
            await cl.evict("s0", [keys[1]])          # break s0's run at index 1
            counts2 = await cl.prefix_lengths(list(keys))
            assert counts2 == {"s0": 1, "s1": 1}
        return True

    ok, _ = run_sim(scenario())
    assert ok


# 3. LRU cache: bounded size + deterministic eviction of the coldest.
def test_lru_capacity_bound_and_eviction():
    c = LRUCache(capacity=2)
    assert c.admit(["a"]) == []
    assert c.admit(["b"]) == []
    c.touch(["a"])                              # a now most-recent
    evicted = c.admit(["d"])                     # over capacity -> evict coldest (b)
    assert evicted == ["b"]
    assert len(c) == 2 and "a" in c and "d" in c and "b" not in c


def test_lru_unbounded_never_evicts():
    c = LRUCache(capacity=None)
    assert c.admit([str(i) for i in range(100)]) == []
    assert len(c) == 100


# 4. Determinism: same seed -> byte-identical trace + identical metrics.
def test_deterministic_trace_and_metrics():
    a = run_shared_prefix(seed=1)[0]
    b = run_shared_prefix(seed=1)[0]
    assert a.trace.render() == b.trace.render()
    assert a.metrics.hit_rate == b.metrics.hit_rate
    assert a.metrics.compute_tokens == b.metrics.compute_tokens


# 5. Cache-aware beats load-balance on reuse + TTFT (shared-prefix workload).
def test_cache_aware_improves_reuse_and_ttft():
    cache_aware, baseline = run_shared_prefix()
    assert cache_aware.metrics.hit_rate >= baseline.metrics.hit_rate
    assert cache_aware.metrics.compute_tokens <= baseline.metrics.compute_tokens
    assert cache_aware.metrics.mean_ttft <= baseline.metrics.mean_ttft


# 6. Reuse actually happens (some prefix tokens served from cache).
def test_reuse_is_nonzero():
    cache_aware, _ = run_shared_prefix()
    assert cache_aware.metrics.saved_tokens > 0
    assert 0.0 < cache_aware.metrics.hit_rate < 1.0


# 7. Eviction: hit rate rises with capacity, then plateaus at the unbounded value.
def test_eviction_hit_rate_monotone_then_plateau():
    rows = run_eviction_sweep()
    caps = [c for c, _hr, _fb in rows]
    hrs = [hr for _c, hr, _fb in rows]
    assert caps == sorted(caps)
    # non-decreasing hit rate as capacity grows
    assert all(hrs[i] <= hrs[i + 1] + 1e-9 for i in range(len(hrs) - 1))
    # a large cache is strictly better than the smallest useful one
    assert hrs[-1] > hrs[0]
    # large finite cap reaches (near) the unbounded hit rate
    topo = make_topology(4)
    reqs = shared_prefix_workload()
    unbounded = run(topo, reqs, "cache_aware", capacity=None).metrics.hit_rate
    big = run(topo, reqs, "cache_aware", capacity=100000).metrics.hit_rate
    assert abs(unbounded - big) < 1e-9


# 8. Hotspot: replication lowers TTFT and prefill compute vs recompute-only,
#    at the cost of KV fabric bytes.
def test_hotspot_replication_helps():
    baseline, no_repl, repl = run_hotspot()
    assert repl.metrics.mean_ttft <= no_repl.metrics.mean_ttft <= baseline.metrics.mean_ttft
    assert repl.metrics.compute_tokens <= no_repl.metrics.compute_tokens
    assert no_repl.metrics.fabric_bytes == 0     # replicate=False never pulls
    assert repl.metrics.fabric_bytes > 0         # replication moves KV once per spread


# 9. Overload: cache-aware rejects no more than the baseline, and admits some.
def test_overload_fewer_rejections():
    cache_aware, baseline = run_overload()
    assert cache_aware.metrics.rejections <= baseline.metrics.rejections
    total = len(cache_aware.metrics.results)
    assert 0 < len(cache_aware.metrics.accepted) < total   # some admitted, some shed


# 10. Fan-out sanity: the LRU primitive never exceeds capacity, and a comfortably
#     sized run still reuses.
def test_cache_never_exceeds_capacity():
    topo = make_topology(4)
    reqs = shared_prefix_workload()
    r = run(topo, reqs, "cache_aware", capacity=64)
    assert r.metrics.hit_rate > 0
    cap = 8
    c = LRUCache(capacity=cap)
    for i in range(200):
        c.admit([f"k{i}"])
        assert len(c) <= cap


# 11. Decode-step cost (from the cost model): baseline at batch 1, clamped below,
#     and strictly increasing in batch size.
def test_decode_step_time_shape():
    base = decode_step_time(1, PROFILE)
    assert decode_step_time(0, PROFILE) == base     # clamps to batch >= 1
    steps = [decode_step_time(b, PROFILE) for b in range(1, 9)]
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))
    assert decode_step_time(2, PROFILE) > base


# 12. Batching raises TBT: a solo request decodes at the batch=1 baseline; several
#     requests co-batched at the same instant each observe a strictly larger gap.
def _run_decode_batch(n: int):
    """Admit ``n`` requests at t=0 on one instance; return {id -> worst TBT}."""
    loop = AsyncEngine()
    res = {}
    eng = DecodeEngine(
        loop, ["s0"], max_batch=8,
        on_finish=lambda r, tbt: res.__setitem__(r.id, tbt),
    )

    async def drive():
        for i in range(n):
            eng.admit(
                Request(id=f"r{i}", arrival=0.0, block_keys=("m0|0",),
                        prompt_tokens=512, output_tokens=6),
                "s0",
            )
        await eng.drain()

    try:
        loop.run_until_complete(drive())
    finally:
        loop.close()
    return res


def test_batching_raises_tbt():
    base = decode_step_time(1, PROFILE)
    solo = _run_decode_batch(1)
    assert abs(solo["r0"] - base) < 1e-9
    batched = _run_decode_batch(4)
    assert min(batched.values()) >= decode_step_time(2, PROFILE)
    assert min(batched.values()) > solo["r0"]


# 13. Disaggregation protects served-request TBT from prefill interference.
def test_disaggregation_protects_tbt():
    disagg, coupled = run_disaggregation()
    # Both serve the identical load with no decode rejection (admission disabled).
    for r in (disagg, coupled):
        assert len(r.metrics.accepted) == len(r.metrics.results)
        assert r.metrics.decode_rejections == 0
    # A dedicated decode pool holds the TBT target for (nearly) every served
    # request; coupling prefill into decode makes a real fraction miss it.
    assert disagg.metrics.tbt_slo_met(DISAGG_TARGET_TBT) >= 0.95
    assert coupled.metrics.tbt_slo_met(DISAGG_TARGET_TBT) < 0.9
    assert (disagg.metrics.tbt_slo_met(DISAGG_TARGET_TBT)
            > coupled.metrics.tbt_slo_met(DISAGG_TARGET_TBT))


# 14. Early rejection avoids wasted prefill; prediction routes decode better.
def test_early_rejection_avoids_wasted_prefill():
    off, early, predict = run_early_rejection()
    # 'off' late-checks decode load after prefill -> some prefills are wasted.
    assert off.metrics.wasted_prefills > 0
    # 'early'/'predict' gate before prefill -> never waste it.
    assert early.metrics.wasted_prefills == 0
    assert predict.metrics.wasted_prefills == 0
    # Only 'predict' routes decode by foreseen load, so it holds the TBT SLO where
    # 'early' (stale current-occupancy snapshot) cannot.
    assert (predict.metrics.tbt_slo_met(EARLY_SLO_TBT)
            > early.metrics.tbt_slo_met(EARLY_SLO_TBT))


# 14b. Divergence gate: the opt-in dict-shim directory yields byte-identical
#      trace + payoff metrics vs the real Trie directory (Task B). Same real
#      Controller decision logic over a plain dict -> hit rate, TTFT, fabric
#      bytes, rejections, and compute must all match exactly.
def test_shim_directory_matches_real():
    real = run_shared_prefix(seed=1)[0]
    with config.overrides(real_directory=False):
        shim = run_shared_prefix(seed=1)[0]
    assert shim.trace.render() == real.trace.render()
    assert shim.metrics.hit_rate == real.metrics.hit_rate
    assert shim.metrics.mean_ttft == real.metrics.mean_ttft
    assert shim.metrics.pct_ttft(90) == real.metrics.pct_ttft(90)
    assert shim.metrics.compute_tokens == real.metrics.compute_tokens
    assert shim.metrics.fabric_bytes == real.metrics.fabric_bytes
    assert shim.metrics.rejections == real.metrics.rejections


def test_shim_overload_rejections_match_real():
    real = run_overload()[0]
    with config.overrides(real_directory=False):
        shim = run_overload()[0]
    assert shim.metrics.rejections == real.metrics.rejections
    assert shim.trace.render() == real.trace.render()


# 15. Determinism of the decode scenarios: same seed -> identical trace + metrics.
def test_new_scenarios_deterministic():
    d1, c1 = run_disaggregation(seed=2)
    d2, c2 = run_disaggregation(seed=2)
    assert d1.trace.render() == d2.trace.render()
    assert c1.trace.render() == c2.trace.render()
    assert d1.metrics.mean_tbt == d2.metrics.mean_tbt
    assert c1.metrics.mean_tbt == c2.metrics.mean_tbt

    off1, early1, predict1 = run_early_rejection(seed=3)
    off2, early2, predict2 = run_early_rejection(seed=3)
    for a, b in ((off1, off2), (early1, early2), (predict1, predict2)):
        assert a.trace.render() == b.trace.render()
        assert a.metrics.wasted_prefills == b.metrics.wasted_prefills
        assert a.metrics.mean_tbt == b.metrics.mean_tbt
