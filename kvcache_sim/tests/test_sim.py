"""Deterministic tests for the KV-cache DES prototype (SPEC.md ~8).

Run from the parent directory: ``python -m pytest kvcache_sim/tests -q``.
Assertions are on the DES *outcome* (hit rate, compute, eviction, rejections,
determinism) -- never on wall-clock timing.
"""

from __future__ import annotations

from kvcache_sim.sim.cache import LRUCache
from kvcache_sim.sim.cost import decode_step_time, TBT_BASE
from kvcache_sim.sim.decode import DecodeEngine
from kvcache_sim.sim.index import BlockIndex
from kvcache_sim.sim.model import (
    block_keys_for,
    longest_prefix_run,
    Request,
)
from kvcache_sim.sim.scenarios import (
    DISAGG_TARGET_TBT,
    EARLY_SLO_TBT,
    make_instances,
    run,
    run_disaggregation,
    run_early_rejection,
    run_eviction_sweep,
    run_hotspot,
    run_overload,
    run_shared_prefix,
    shared_prefix_workload,
)
from sim_common.engine import Sim


# 1. Prefix-hash addressing: shared prefixes yield shared keys (K1).
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


# 2. Index: per-instance prefix-match length (the cache-aware scheduler query).
def test_index_instances_with_prefix():
    idx = BlockIndex()
    keys = block_keys_for("m0", [0, 1, 2, 3])
    for k in keys[:3]:
        idx.notify_put(k, "s0")
    for k in keys[:1]:
        idx.notify_put(k, "s1")
    counts = idx.instances_with_prefix(list(keys))
    assert counts == {"s0": 3, "s1": 1}
    idx.notify_delete(keys[1], "s0")            # break s0's run at index 1
    assert idx.instances_with_prefix(list(keys)) == {"s0": 1, "s1": 1}


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


# 4. Determinism: same seed -> identical trace + metrics.
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
    insts = make_instances(4)
    reqs = shared_prefix_workload()
    unbounded = run(insts, reqs, "cache_aware", capacity=None).metrics.hit_rate
    big = run(insts, reqs, "cache_aware", capacity=100000).metrics.hit_rate
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


# 10. Fan-out sanity: no instance's cache ever exceeds its capacity.
def test_cache_never_exceeds_capacity():
    insts = make_instances(4)
    reqs = shared_prefix_workload()
    # A capacity comfortably larger than one prompt (10 blocks) still yields reuse.
    r = run(insts, reqs, "cache_aware", capacity=64)
    assert r.metrics.hit_rate > 0    # the run itself is well-formed
    # Assert the capacity invariant on the LRU primitive under an adversarial seq.
    cap = 8
    c = LRUCache(capacity=cap)
    for i in range(200):
        c.admit([f"k{i}"])
        assert len(c) <= cap


# 11. Batched-decode cost model (K6): step time is TBT_BASE at batch 1 and
#     strictly increasing in batch size.
def test_decode_step_time_shape():
    assert decode_step_time(1) == TBT_BASE
    assert decode_step_time(0) == TBT_BASE          # clamps to batch >= 1
    steps = [decode_step_time(b) for b in range(1, 9)]
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))
    assert decode_step_time(2) > TBT_BASE


# 12. Batching raises TBT: a solo request decodes at the batch=1 baseline; several
#     requests co-batched at the same instant each observe a strictly larger gap.
def test_batching_raises_tbt():
    # Solo: one request, output_tokens > 1 -> its TBT is the batch=1 step time.
    sim = Sim()
    solo_tbt = {}
    eng = DecodeEngine(sim, ["s0"], max_batch=8,
                       on_finish=lambda r, tbt: solo_tbt.__setitem__(r.id, tbt))
    eng.admit(Request(id="r0", arrival=0.0, block_keys=("m0|0",),
                      prompt_tokens=512, output_tokens=6), "s0")
    sim.run()
    # A solo request never shares a step, so every gap is the batch=1 step time
    # (float accumulation over steps -> compare within tolerance).
    assert abs(solo_tbt["r0"] - decode_step_time(1)) < 1e-9
    assert decode_step_time(1) == TBT_BASE

    # Batched: several requests admitted at the same instant share every step, so
    # each sees at least the batch>=2 step time -- strictly above the solo gap.
    sim2 = Sim()
    batch_tbt = {}
    eng2 = DecodeEngine(sim2, ["s0"], max_batch=8,
                        on_finish=lambda r, tbt: batch_tbt.__setitem__(r.id, tbt))
    for i in range(4):
        eng2.admit(Request(id=f"r{i}", arrival=0.0, block_keys=("m0|0",),
                           prompt_tokens=512, output_tokens=6), "s0")
    sim2.run()
    assert min(batch_tbt.values()) >= decode_step_time(2)
    assert min(batch_tbt.values()) > solo_tbt["r0"]


# 13. Disaggregation protects served-request TBT from prefill interference.
def test_disaggregation_protects_tbt():
    disagg, coupled = run_disaggregation()
    # Both serve the identical load with no decode rejection (admission disabled).
    for r in (disagg, coupled):
        assert len(r.metrics.accepted) == len(r.metrics.results)
        assert r.metrics.decode_rejections == 0
    # A dedicated decode pool holds the TBT target for every served request;
    # coupling prefill into decode makes a real fraction miss it.
    assert disagg.metrics.tbt_slo_met(DISAGG_TARGET_TBT) == 1.0
    assert coupled.metrics.tbt_slo_met(DISAGG_TARGET_TBT) < 0.95
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


# 15. Determinism of the new decode scenarios: same seed -> identical trace + key
#     metrics.
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
