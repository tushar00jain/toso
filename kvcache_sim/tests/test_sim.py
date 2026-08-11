"""Deterministic tests for the KV-cache simulation on the *real* TorchStore.

Run from the parent directory: ``python -m pytest kvcache_sim/tests -q``.
Assertions are on the simulation *outcome* (block presence in the real directory,
prefix hit rate, prefill compute, eviction, rejections, TBT, determinism) -- never
on wall-clock timing. Every scenario drives the real ``Controller`` directory + real
per-instance clients on the shared deterministic async engine.
"""

from __future__ import annotations

import pytest

from sim_common import config
from sim_common.async_engine import AsyncEngine, run_sim

from kvcache_sim.control._cache import LRUCache
from kvcache_sim.control.view import KVView, _longest_prefix_run
from kvcache_sim.workload._serving import _sim_block_carrier
from realsim.simulation import Simulation
from sim_common.cost_model import DEFAULT_PROFILE
from domain import decode_step_time
from kvcache_sim.data._decode import DecodeEngine
from kvcache_sim.data.store import KVStore
from kvcache_sim.control.request import Request
from kvcache_sim.workload._generator import _block_keys_for
from kvcache_sim.tests._run import (
    run,
    run_disaggregation,
    run_early_rejection,
    run_eviction_sweep,
    run_hotspot,
    run_overload,
    run_shared_prefix,
)
from kvcache_sim.workload.scenarios import (
    DISAGG_TARGET_TBT,
    EARLY_SLO_TBT,
    _make_topology,
    _shared_prefix_workload,
)


# 1. Prefix-hash addressing: shared prefixes yield shared keys.
def test_prefix_hash_chain_shares_prefix():
    a = _block_keys_for("m0", [0, 1, 7, 9])
    b = _block_keys_for("m0", [0, 1, 8])
    assert a[:2] == b[:2]         # shared leading segments -> identical keys
    assert a[2] != b[2]           # divergence -> distinct keys
    # different model never aliases:
    assert _block_keys_for("m1", [0, 1])[0] != _block_keys_for("m0", [0, 1])[0]


def test_longest_prefix_run():
    keys = _block_keys_for("m0", [0, 1, 2, 3])
    present = {keys[0], keys[1], keys[3]}   # a gap at index 2
    assert _longest_prefix_run(keys, present) == 2   # stops at first miss


# 2. REAL directory: per-instance prefix-match length, incl. after eviction.
#    Publishing records block->volume presence in the real Controller directory;
#    locate_volumes reads it back; eviction removes it. This is the cache-aware
#    scheduler's core query, answered directly by the real directory.
def test_real_directory_prefix_presence_and_eviction():
    topo = _make_topology(2)
    keys = _block_keys_for("m0", [0, 1, 2, 3])

    sim = Simulation(topo)
    store = KVStore.for_deployment(
        sim.mesh, block_tokens=512, carrier=_sim_block_carrier(512)
    )
    view = KVView(sim.view.directory, sim.topology)

    async def scenario():
        # The data plane publishes/evicts; the control-plane view reads back.
        with sim.mesh.installed():
            await store.publish("s0", list(keys[:3]))  # s0 holds 3 leading blocks
            await store.publish("s1", list(keys[:1]))  # s1 holds 1
            counts = await view.prefix_lengths(list(keys))
            assert counts == {"s0": 3, "s1": 1}
            await store.evict("s0", [keys[1]])         # break s0's run at index 1
            counts2 = await view.prefix_lengths(list(keys))
            assert counts2 == {"s0": 1, "s1": 1}
        return True

    try:
        ok = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
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
    assert a.ledger.hit_rate == b.ledger.hit_rate
    assert a.ledger.compute_tokens == b.ledger.compute_tokens


# 5. Cache-aware beats load-balance on reuse + TTFT (shared-prefix workload).
def test_cache_aware_improves_reuse_and_ttft():
    cache_aware, baseline = run_shared_prefix()
    assert cache_aware.ledger.hit_rate >= baseline.ledger.hit_rate
    assert cache_aware.ledger.compute_tokens <= baseline.ledger.compute_tokens
    assert cache_aware.ledger.mean_ttft <= baseline.ledger.mean_ttft


# 6. Reuse actually happens (some prefix tokens served from cache).
def test_reuse_is_nonzero():
    cache_aware, _ = run_shared_prefix()
    assert cache_aware.ledger.saved_tokens > 0
    assert 0.0 < cache_aware.ledger.hit_rate < 1.0


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
    topo = _make_topology(4)
    reqs = _shared_prefix_workload()
    unbounded = run(topo, reqs, "cache_aware", capacity=None).ledger.hit_rate
    big = run(topo, reqs, "cache_aware", capacity=100000).ledger.hit_rate
    assert abs(unbounded - big) < 1e-9


# 8. Hotspot: replication lowers TTFT and prefill compute vs recompute-only,
#    at the cost of KV fabric bytes.
def test_hotspot_replication_helps():
    baseline, no_repl, repl = run_hotspot()
    assert repl.ledger.mean_ttft <= no_repl.ledger.mean_ttft <= baseline.ledger.mean_ttft
    assert repl.ledger.compute_tokens <= no_repl.ledger.compute_tokens
    assert no_repl.ledger.fabric_bytes == 0     # replicate=False never pulls
    assert repl.ledger.fabric_bytes > 0         # replication moves KV once per spread


# 9. Overload: cache-aware rejects no more than the baseline, and admits some.
def test_overload_fewer_rejections():
    cache_aware, baseline = run_overload()
    assert cache_aware.ledger.rejections <= baseline.ledger.rejections
    total = len(cache_aware.ledger.results)
    assert 0 < len(cache_aware.ledger.accepted) < total   # some admitted, some shed


# 10. Fan-out sanity: the LRU primitive never exceeds capacity, and a comfortably
#     sized run still reuses.
def test_cache_never_exceeds_capacity():
    topo = _make_topology(4)
    reqs = _shared_prefix_workload()
    r = run(topo, reqs, "cache_aware", capacity=64)
    assert r.ledger.hit_rate > 0
    cap = 8
    c = LRUCache(capacity=cap)
    for i in range(200):
        c.admit([f"k{i}"])
        assert len(c) <= cap


# 11. Decode-step cost (from the cost model): baseline at batch 1, clamped below,
#     and strictly increasing in batch size.
def test_decode_step_time_shape():
    base = decode_step_time(1, DEFAULT_PROFILE)
    assert decode_step_time(0, DEFAULT_PROFILE) == base     # clamps to batch >= 1
    steps = [decode_step_time(b, DEFAULT_PROFILE) for b in range(1, 9)]
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))
    assert decode_step_time(2, DEFAULT_PROFILE) > base


# 12. Batching raises TBT: a solo request decodes at the batch=1 baseline; several
#     requests co-batched at the same instant each observe a strictly larger gap.
def _run_decode_batch(n: int):
    """Admit ``n`` requests at t=0 on one instance; return {id -> worst TBT}.

    Drives the decode engine against a bare clock: it needs no store, no
    directory and no topology, so assembling a Simulation would build a mesh for
    nothing.
    """
    loop = AsyncEngine()
    res = {}
    eng = DecodeEngine(
        ["s0"], max_batch=8,
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
    base = decode_step_time(1, DEFAULT_PROFILE)
    solo = _run_decode_batch(1)
    assert abs(solo["r0"] - base) < 1e-9
    batched = _run_decode_batch(4)
    assert min(batched.values()) >= decode_step_time(2, DEFAULT_PROFILE)
    assert min(batched.values()) > solo["r0"]


# 13. Disaggregation protects served-request TBT from prefill interference.
def test_disaggregation_protects_tbt():
    disagg, coupled = run_disaggregation()
    # Both serve the identical load with no decode rejection (admission disabled).
    for r in (disagg, coupled):
        assert len(r.ledger.accepted) == len(r.ledger.results)
        assert r.ledger.decode_rejections == 0
    # A dedicated decode pool holds the TBT target for (nearly) every served
    # request; coupling prefill into decode makes a real fraction miss it.
    assert disagg.ledger.tbt_slo_met(DISAGG_TARGET_TBT) >= 0.95
    assert coupled.ledger.tbt_slo_met(DISAGG_TARGET_TBT) < 0.9
    assert (disagg.ledger.tbt_slo_met(DISAGG_TARGET_TBT)
            > coupled.ledger.tbt_slo_met(DISAGG_TARGET_TBT))


# 14. Early rejection avoids wasted prefill; prediction routes decode better.
def test_early_rejection_avoids_wasted_prefill():
    off, early, predict = run_early_rejection()
    # 'off' late-checks decode load after prefill -> some prefills are wasted.
    assert off.ledger.wasted_prefills > 0
    # 'early'/'predict' gate before prefill -> never waste it.
    assert early.ledger.wasted_prefills == 0
    assert predict.ledger.wasted_prefills == 0
    # Only 'predict' routes decode by foreseen load, so it holds the TBT SLO where
    # 'early' (stale current-occupancy snapshot) cannot.
    assert (predict.ledger.tbt_slo_met(EARLY_SLO_TBT)
            > early.ledger.tbt_slo_met(EARLY_SLO_TBT))


# 14b. Divergence gate: the opt-in dict-shim directory yields byte-identical
#      trace + payoff metrics vs the real Trie directory (Task B). Same real
#      Controller decision logic over a plain dict -> hit rate, TTFT, fabric
#      bytes, rejections, and compute must all match exactly.
def test_shim_directory_matches_real():
    real = run_shared_prefix(seed=1)[0]
    with config.overrides(real_directory=False):
        shim = run_shared_prefix(seed=1)[0]
    assert shim.trace.render() == real.trace.render()
    assert shim.ledger.hit_rate == real.ledger.hit_rate
    assert shim.ledger.mean_ttft == real.ledger.mean_ttft
    assert shim.ledger.pct_ttft(90) == real.ledger.pct_ttft(90)
    assert shim.ledger.compute_tokens == real.ledger.compute_tokens
    assert shim.ledger.fabric_bytes == real.ledger.fabric_bytes
    assert shim.ledger.rejections == real.ledger.rejections


def test_shim_overload_rejections_match_real():
    real = run_overload()[0]
    with config.overrides(real_directory=False):
        shim = run_overload()[0]
    assert shim.ledger.rejections == real.ledger.rejections
    assert shim.trace.render() == real.trace.render()


# 14c. Contention smoke (Task D): the shared-prefix scenario runs under each
#      network/storage contention model and stays deterministic per mode. The
#      registry is read ambiently by the KVStore (config.contention), mirroring
#      how the dict-shim directory flag is wired.
@pytest.mark.parametrize("contention", ("none", "serialize", "progressive"))
def test_shared_prefix_runs_under_each_contention_mode(contention):
    with config.overrides(contention=contention):
        a = run_shared_prefix(seed=1)[0]
        b = run_shared_prefix(seed=1)[0]
    # Runs to completion, produces a sane reuse metric, and is deterministic.
    assert 0.0 < a.ledger.hit_rate < 1.0
    assert a.trace.render() == b.trace.render()
    assert a.ledger.hit_rate == b.ledger.hit_rate
    assert a.ledger.compute_tokens == b.ledger.compute_tokens


# 14d. Collapse-charges gate: coalescing each transport op's per-component sleeps
#      into one (a get's storage+mem+network) is a timing coarsening on the
#      non-contended path -- the cache-aware payoff metrics (hit rate, compute
#      saved) do not depend on the vanished sub-charge instants, so they are
#      unchanged vs collapse-off, and the run stays deterministic. The flag is
#      read ambiently by the KVStore's transports, like the contention flag.
def test_shared_prefix_metrics_invariant_to_collapse():
    off = run_shared_prefix(seed=1)[0]
    with config.overrides(collapse_charges=True):
        on = run_shared_prefix(seed=1)[0]
    assert on.ledger.hit_rate == off.ledger.hit_rate
    assert on.ledger.saved_tokens == off.ledger.saved_tokens
    assert on.ledger.compute_tokens == off.ledger.compute_tokens
    assert on.ledger.fabric_bytes == off.ledger.fabric_bytes


def test_shared_prefix_collapse_is_deterministic():
    with config.overrides(collapse_charges=True):
        a = run_shared_prefix(seed=1)[0]
        b = run_shared_prefix(seed=1)[0]
    assert a.trace.render() == b.trace.render()
    assert a.ledger.hit_rate == b.ledger.hit_rate


# 15. Determinism of the decode scenarios: same seed -> identical trace + metrics.
def test_new_scenarios_deterministic():
    d1, c1 = run_disaggregation(seed=2)
    d2, c2 = run_disaggregation(seed=2)
    assert d1.trace.render() == d2.trace.render()
    assert c1.trace.render() == c2.trace.render()
    assert d1.ledger.mean_tbt == d2.ledger.mean_tbt
    assert c1.ledger.mean_tbt == c2.ledger.mean_tbt

    off1, early1, predict1 = run_early_rejection(seed=3)
    off2, early2, predict2 = run_early_rejection(seed=3)
    for a, b in ((off1, off2), (early1, early2), (predict1, predict2)):
        assert a.trace.render() == b.trace.render()
        assert a.ledger.wasted_prefills == b.ledger.wasted_prefills
        assert a.ledger.mean_tbt == b.ledger.mean_tbt


# 16. The coordinator seam: control is a service, so the hop is somewhere to
#     charge. At the default (0) it is inline and changes nothing; given a
#     duration it is paid out and back before prefill can start, so it lands in
#     TTFT -- the number this capability exists to move.
def test_coordinator_rtt_defaults_to_free_and_byte_identical():
    baseline = run_shared_prefix(seed=1)[0]
    with config.overrides(coordinator_rtt=0.0):
        explicit = run_shared_prefix(seed=1)[0]
    assert explicit.trace.render() == baseline.trace.render()
    assert explicit.ledger.mean_ttft == baseline.ledger.mean_ttft


def test_coordinator_rtt_lands_in_ttft_and_costs_reuse():
    """A distant coordinator costs latency *and* hit rate.

    Latency because every request pays the round trip before prefill can start,
    and the delay compounds through the prefill queue. Hit rate because routing
    then reads a directory snapshot one hop old, so a prefix another request has
    just published is not there to reuse yet. The RTT has to be large enough to
    matter against a ~4s prefill: at 0.01 nothing moves, which is its own honest
    result about what resolution this workload can see.
    """
    free = run_shared_prefix(seed=1)[0]
    with config.overrides(coordinator_rtt=0.5):
        distant = run_shared_prefix(seed=1)[0]
    assert distant.ledger.mean_ttft > free.ledger.mean_ttft
    assert distant.ledger.hit_rate < free.ledger.hit_rate
    # The comparison still holds -- both schedulers pay the same hop.
    with config.overrides(coordinator_rtt=0.5):
        cache_aware, load_balance = run_shared_prefix(seed=1)
    assert cache_aware.ledger.hit_rate > load_balance.ledger.hit_rate
    assert cache_aware.ledger.mean_ttft < load_balance.ledger.mean_ttft


# 17. The peer that serves a pull is the peer the coordinator priced.
#     Control ranks candidates by prefix and prices the pull at that peer's
#     locality tier (NVLink within a node, RDMA across). The store cannot know
#     that: `locate_volumes` returns every holder and the client takes the first,
#     so for a block several instances hold -- a shared system prompt, or
#     anything replicated -- the bytes could come from a different tier than the
#     one predicted. The run installs LongestPrefixPolicy in the directory and
#     the fetch names its source, so the answer is narrowed to the priced peer.
def _unplanned_edges(result) -> int:
    """Transfer edges whose (source, destination) no accepted plan asked for."""
    from collections import Counter

    planned = Counter(
        (r.reuse_source, r.prefill)
        for r in result.ledger.rows
        if getattr(r, "reuse_source", None)
    )
    actual = Counter((src, dst) for src, dst, _label in result.ledger.edges)
    return sum((actual - planned).values())


def test_a_pull_is_served_by_the_peer_that_was_priced():
    cache_aware = run_shared_prefix(seed=1)[0]
    assert cache_aware.ledger.edges, "no transfers -- the assertion would be vacuous"
    assert _unplanned_edges(cache_aware) == 0
    # Replication is the case that makes a block multi-holder on purpose.
    _baseline, _no_repl, replicated = run_hotspot(seed=2)
    assert replicated.ledger.edges
    assert _unplanned_edges(replicated) == 0


# 18. The source policy serves both its callers' views.
#     The scheduler hands it a KVView (pinned, so one decision reads one
#     snapshot); the controller can only hand it the plain View the mesh built,
#     because a prefix run is a KV-cache notion the store has no reason to know.
#     It used to require the first, so the ranking branch raised AttributeError on
#     the second -- unreachable only because the controller-side call always
#     carries a chosen source and short-circuits before touching the view.
def test_the_source_policy_accepts_a_plain_view():
    from kvcache_sim.control._source import LongestPrefixPolicy
    from realsim.simulation import Simulation

    topo = _make_topology(2)
    sim = Simulation(topo)
    keys = _block_keys_for("m0", [0, 1])

    async def scenario():
        with sim.mesh.installed():
            empty = await LongestPrefixPolicy().select(sim.view, keys, "s0")
            store = KVStore.for_deployment(
                sim.mesh, block_tokens=512, carrier=_sim_block_carrier(512)
            )
            await store.publish("s1", list(keys))
            ranked = await LongestPrefixPolicy().select(sim.view, keys, "s0")
        return empty, ranked

    try:
        empty, ranked = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert empty.sources == ()                  # nobody holds it yet
    assert ranked.sources == ("s1",)            # ...and now the holder is ranked
