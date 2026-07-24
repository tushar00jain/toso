"""Deterministic tests for the analytic resource cost model (design doc 3).

Run from the repo root with the venv interpreter::

    PYTHONPATH=. /path/to/.venv/bin/python -m pytest sim_common/tests/test_cost_model.py -q

Also runnable as a plain script (``python sim_common/tests/test_cost_model.py``)
if pytest is unavailable. Assertions are on cost-model *properties* --
monotonicity, zero-quantity => 0, roofline binding term, determinism -- never on
measured/wall-clock quantities (the model never measures anything).
"""

from __future__ import annotations

from dataclasses import replace

from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
    compute_time,
    mem_copy_time,
    network_time,
    storage_time,
)
from sim_common.topology import Endpoint, Tier

# --------------------------------------------------------------------------
# Fixtures: endpoints spanning every locality tier + a compact profile.
# --------------------------------------------------------------------------

_A = Endpoint(id="a", host="h1", node="n1")
_A2 = Endpoint(id="a2", host="h1", node="n1")   # same host as _A -> SHM
_B = Endpoint(id="b", host="h2", node="n1")     # same node, diff host -> NVLINK
_C = Endpoint(id="c", host="h3", node="n2")     # diff node -> RDMA

PROFILE = DEFAULT_PROFILE


# --------------------------------------------------------------------------
# Zero-quantity => 0.0 (mirrors transfer_time's free zero-byte transfer).
# --------------------------------------------------------------------------


def test_zero_quantity_is_free() -> None:
    assert network_time(_A, _C, 0, PROFILE) == 0.0
    assert mem_copy_time(0, PROFILE) == 0.0
    assert storage_time(0, "read", PROFILE) == 0.0
    assert storage_time(0, "write", PROFILE) == 0.0
    assert compute_time(0, "float32", "cuda", PROFILE) == 0.0
    assert compute_time(0, "float32", "cpu", PROFILE, nbytes=0) == 0.0


def test_same_endpoint_network_is_free() -> None:
    # Same identity is a no-op even for a nonzero byte count.
    same = Endpoint(id="a", host="h9", node="n9")
    assert network_time(_A, same, 1_000_000, PROFILE) == 0.0


# --------------------------------------------------------------------------
# Monotonicity: more bytes / more flops => strictly more time.
# --------------------------------------------------------------------------


def test_network_monotonic_in_bytes() -> None:
    for src, dst in ((_A, _A2), (_A, _B), (_A, _C)):
        t1 = network_time(src, dst, 1_000, PROFILE)
        t2 = network_time(src, dst, 2_000, PROFILE)
        assert 0.0 < t1 < t2


def test_network_tier_ordering() -> None:
    # Farther locality is never cheaper for the same payload.
    n = 100_000
    shm = network_time(_A, _A2, n, PROFILE)
    nvlink = network_time(_A, _B, n, PROFILE)
    rdma = network_time(_A, _C, n, PROFILE)
    assert shm < nvlink < rdma


def test_mem_copy_monotonic_in_bytes() -> None:
    assert mem_copy_time(1_000, PROFILE) < mem_copy_time(2_000, PROFILE)


def test_storage_monotonic_in_bytes() -> None:
    for kind in ("read", "write"):
        assert storage_time(1_000, kind, PROFILE) < storage_time(2_000, kind, PROFILE)


def test_compute_monotonic_in_flops() -> None:
    # With no memory term, more flops strictly costs more.
    t1 = compute_time(1.0e6, "float32", "cuda", PROFILE)
    t2 = compute_time(2.0e6, "float32", "cuda", PROFILE)
    assert 0.0 < t1 < t2


def test_compute_monotonic_in_bytes_when_memory_bound() -> None:
    # With no flops, more bytes strictly costs more (pure memory term).
    t1 = compute_time(0.0, "float32", "cuda", PROFILE, nbytes=1_000_000)
    t2 = compute_time(0.0, "float32", "cuda", PROFILE, nbytes=2_000_000)
    assert 0.0 < t1 < t2


# --------------------------------------------------------------------------
# Roofline: max() picks the binding term; the other term is slack.
# --------------------------------------------------------------------------


def test_roofline_picks_binding_term() -> None:
    flops = 1.0e6
    dtype, device = "float32", "cuda"
    compute_only = compute_time(flops, dtype, device, PROFILE)
    eff = PROFILE.gpu_flops[dtype]
    mem_bw = PROFILE.gpu_mem_bandwidth

    # Choose bytes so the memory term dominates (memory-bound kernel).
    mem_bound_bytes = int(compute_only * mem_bw * 4) + 1
    t_mem = compute_time(flops, dtype, device, PROFILE, nbytes=mem_bound_bytes)
    assert t_mem == mem_bound_bytes / mem_bw          # memory term binds
    assert t_mem > compute_only

    # Choose bytes so the compute term dominates (compute-bound kernel).
    compute_bound_bytes = int(compute_only * mem_bw / 4)
    t_compute = compute_time(flops, dtype, device, PROFILE, nbytes=compute_bound_bytes)
    assert t_compute == flops / eff                   # compute term binds
    assert t_compute == compute_only


def test_roofline_is_the_max_of_both_terms() -> None:
    flops, nbytes = 3.0e6, 500_000
    dtype, device = "bfloat16", "cuda"
    got = compute_time(flops, dtype, device, PROFILE, nbytes=nbytes)
    expected = max(
        flops / PROFILE.gpu_flops[dtype],
        nbytes / PROFILE.gpu_mem_bandwidth,
    )
    assert got == expected


# --------------------------------------------------------------------------
# Device / dtype resolution.
# --------------------------------------------------------------------------


def test_gpu_dtype_selection() -> None:
    # float16 has a higher flop rate than float32 in the demo profile, so the
    # same flop count is cheaper in the faster dtype.
    slow = compute_time(1.0e6, "float32", "cuda", PROFILE)
    fast = compute_time(1.0e6, "float16", "cuda", PROFILE)
    assert fast < slow


def test_unknown_gpu_dtype_falls_back_to_default() -> None:
    got = compute_time(1.0e6, "int4", "cuda", PROFILE)
    assert got == 1.0e6 / PROFILE.gpu_flops_default


def test_cpu_device_uses_cpu_flops_and_ram_bw() -> None:
    flops, nbytes = 1.0e5, 400_000
    got = compute_time(flops, "float32", "cpu", PROFILE, nbytes=nbytes)
    expected = max(flops / PROFILE.cpu_flops, nbytes / PROFILE.ram_bandwidth)
    assert got == expected


# --------------------------------------------------------------------------
# Storage kind selection + validation.
# --------------------------------------------------------------------------


def test_storage_read_vs_write_use_distinct_bandwidths() -> None:
    n = 1_000_000
    read = storage_time(n, "read", PROFILE)
    write = storage_time(n, "write", PROFILE)
    # Demo profile: writes are slower than reads.
    assert write > read
    assert read == PROFILE.storage_latency + n / PROFILE.storage_read_bw
    assert write == PROFILE.storage_latency + n / PROFILE.storage_write_bw


def test_storage_rejects_unknown_kind() -> None:
    try:
        storage_time(1_000, "append", PROFILE)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown storage kind")


# --------------------------------------------------------------------------
# Exact formulas (documents the shape callers depend on).
# --------------------------------------------------------------------------


def test_mem_copy_formula() -> None:
    n = 123_456
    assert mem_copy_time(n, PROFILE) == PROFILE.ram_latency + n / PROFILE.ram_bandwidth


def test_network_formula_matches_tier() -> None:
    n = 100_000
    lat, bw = PROFILE.tiers[Tier.RDMA]
    assert network_time(_A, _C, n, PROFILE) == lat + n / bw


# --------------------------------------------------------------------------
# Determinism: identical inputs => byte-identical outputs, always.
# --------------------------------------------------------------------------


def test_determinism_repeated_calls() -> None:
    calls = [
        lambda: network_time(_A, _C, 987_654, PROFILE),
        lambda: mem_copy_time(987_654, PROFILE),
        lambda: storage_time(987_654, "write", PROFILE),
        lambda: compute_time(4.2e6, "bfloat16", "cuda", PROFILE, nbytes=333_333),
    ]
    for call in calls:
        first = call()
        for _ in range(100):
            assert call() == first


def test_profile_is_swappable() -> None:
    # A different profile (frozen dataclass replace) gives a different, still
    # deterministic answer -- constants live in the profile, not the functions.
    faster = replace(PROFILE, ram_bandwidth=PROFILE.ram_bandwidth * 2)
    assert isinstance(faster, MachineProfile)
    base = mem_copy_time(1_000_000, PROFILE)
    quick = mem_copy_time(1_000_000, faster)
    assert quick < base


# --------------------------------------------------------------------------
# Script fallback (no pytest required).
# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
