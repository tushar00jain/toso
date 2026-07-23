"""Deterministic tests for the dedup DES prototype (SPEC.md §8).

Run from the parent directory: ``python -m pytest dedup_sim/tests -q``.
Assertions are on the DES *outcome* (bytes moved, assembled regions,
fan-out, determinism) -- never on wall-clock timing.
"""

from __future__ import annotations

from dedup_sim.sim.model import decompose, region_bytes
from dedup_sim.sim.scenarios import (
    reshard_scenario,
    run,
    toy_scenario,
    versioning_result,
)


def _need_bytes(need, dtype_bytes):
    return sum(region_bytes(r, dtype_bytes) for r in need)


# 1. Correctness: every generator ends holding exactly its need (as atomics).
def test_dedup_assembles_exact_need_toy():
    scn = toy_scenario()
    res = run(scn, "dedup", fanout_cap=1)
    for gid, need in scn.needs.items():
        expected = set(decompose(need, scn.atomics))
        assert res.metrics.assembled[gid] == expected


def test_naive_assembles_exact_need_toy():
    scn = toy_scenario()
    res = run(scn, "naive")
    for gid, need in scn.needs.items():
        expected = set(decompose(need, scn.atomics))
        assert res.metrics.assembled[gid] == expected


# 2. 1x fabric: dedup trainer->gen bytes == union of needs; naive is m x; and
#    dedup strictly less than naive.
def test_fabric_dedup_is_1x_naive_is_mx():
    scn = toy_scenario(num_gens=3)
    dedup = run(scn, "dedup", fanout_cap=1)
    naive = run(scn, "naive")

    union = scn.union_bytes
    assert dedup.metrics.fabric_bytes == union  # each unique region leaves once
    assert naive.metrics.fabric_bytes == sum(
        _need_bytes(need, scn.dtype_bytes) for need in scn.needs.values()
    )
    assert naive.metrics.fabric_bytes == 3 * union  # m = 3, full replication
    assert dedup.metrics.fabric_bytes < naive.metrics.fabric_bytes


def test_fabric_dedup_1x_independent_of_fanout_cap():
    scn = toy_scenario(num_gens=4)
    for cap in (1, 2, 3):
        dedup = run(scn, "dedup", fanout_cap=cap)
        assert dedup.metrics.fabric_bytes == scn.union_bytes


# 3. Fan-out cap respected: no source exceeds FANOUT_CAP concurrent serves.
def test_fanout_cap_never_exceeded():
    for cap in (1, 2, 3):
        scn = toy_scenario(num_gens=5)
        dedup = run(scn, "dedup", fanout_cap=cap)
        assert dedup.peak_serving <= cap


def test_fanout_cap1_is_a_chain():
    scn = toy_scenario(num_gens=4)
    dedup = run(scn, "dedup", fanout_cap=1)
    # each source serves at most one consumer -> a chain
    from collections import Counter

    fanout = Counter(src for (src, _dst, _r) in dedup.metrics.edges)
    assert max(fanout.values()) == 1


def test_fanout_cap2_builds_a_tree():
    scn = toy_scenario(num_gens=4)
    dedup = run(scn, "dedup", fanout_cap=2)
    from collections import Counter

    fanout = Counter(src for (src, _dst, _r) in dedup.metrics.edges)
    assert max(fanout.values()) == 2  # a source fans out to two peers


# 4. Determinism: two runs yield byte-identical trace strings.
def test_trace_deterministic_across_runs():
    scn = toy_scenario()
    a = run(scn, "dedup", fanout_cap=1).trace.render()
    b = run(scn, "dedup", fanout_cap=1).trace.render()
    assert a == b

    scn2 = toy_scenario()
    c = run(scn2, "dedup", fanout_cap=2).trace.render()
    d = run(scn2, "dedup", fanout_cap=2).trace.render()
    assert c == d


# 5. Versioning: a put bump invalidates the cache (subsequent burst re-pulls).
def test_version_bump_invalidates_cache():
    f1, f2_bump, union = versioning_result(bump=True)
    assert f1 == union  # first burst pulls the union once
    assert f2_bump == union  # bump -> cache invalid -> re-pull the union

    f1b, f2_nobump, union2 = versioning_result(bump=False)
    assert f1b == union2
    assert f2_nobump == 0  # no bump -> cache serves burst 2, zero trainer fabric


# 6. Reshard: differing trainer/generator partitions; correct + still 1x fabric.
def test_reshard_correct_and_1x_fabric():
    scn = reshard_scenario()
    dedup = run(scn, "dedup", fanout_cap=1)
    naive = run(scn, "naive")

    # correctness: every generator assembles exactly its need (as atomics)
    for gid, need in scn.needs.items():
        expected = set(decompose(need, scn.atomics))
        assert dedup.metrics.assembled[gid] == expected
        assert naive.metrics.assembled[gid] == expected

    # 1x fabric: each unique atomic region leaves the trainer exactly once
    assert dedup.metrics.fabric_bytes == scn.union_bytes
    # overlap in needs -> dedup strictly beats naive
    assert dedup.metrics.fabric_bytes < naive.metrics.fabric_bytes


def test_reshard_atomic_split_covers_needs_exactly():
    scn = reshard_scenario()
    # assembled union of atomics for each need must reconstruct the need range
    for _gid, need in scn.needs.items():
        atoms = decompose(need, scn.atomics)
        covered = sum(region_bytes(a, scn.dtype_bytes) for a in atoms)
        assert covered == _need_bytes(need, scn.dtype_bytes)


# Extra: all readers complete (no hangs / unresolved promises).
def test_all_readers_complete():
    for scn in (toy_scenario(), reshard_scenario()):
        for cap in (1, 2):
            res = run(scn, "dedup", fanout_cap=cap)
            assert res.metrics.readers_done == res.metrics.readers_total
            assert res.metrics.readers_done == len(scn.needs)
