"""Deterministic tests for the dedup capability on the real TorchStore directory.

These assert the dedup *outcome* on the real directory/client (not wall-clock
timing): every reader receives the right payload, the fabric is the 1x union (each
unique byte crosses the fabric once) versus ``m x`` for the naive baseline, the
fan-out cap shapes a chain/tree, and the trace is byte-identical across runs.

The data plane is allocation-free (meta tensor / descriptor carriers, see
``docs/realsim_design.md`` s7), so correctness is asserted on
shape/dtype/nbytes rather than exact bytes -- the exact-byte reassembly guarantee
of the real client lives in ``realsim/tests/test_correctness.py``.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests -q
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from dedup_sim.scenario import (
    DEFAULT_N,
    MODE_META,
    MODE_METADATA,
    run_dedup_burst,
    run_naive_burst,
)
from realsim.seams.transport import TensorDescriptor
from sim_common import config

MODES = (MODE_META, MODE_METADATA)
PAYLOAD_BYTES = DEFAULT_N * 4  # DEFAULT_N float32 elements


def _shape_dtype_nbytes(payload):
    """Uniform ``(shape, dtype, nbytes)`` for a meta tensor or a descriptor."""
    if isinstance(payload, torch.Tensor):
        return tuple(payload.shape), payload.dtype, payload.numel() * payload.element_size()
    assert isinstance(payload, TensorDescriptor)
    return payload.shape, payload.dtype, payload.nbytes


# 1. Correctness: every reader receives a payload of the right shape/dtype/nbytes.
@pytest.mark.parametrize("mode", MODES)
def test_every_reader_receives_the_payload(mode):
    res = run_dedup_burst(num_readers=4, fanout_cap=1, mode=mode)
    assert set(res.results) == {"r0", "r1", "r2", "r3"}
    for reader_id, payload in res.results.items():
        shape, dtype, nbytes = _shape_dtype_nbytes(payload)
        assert shape == (DEFAULT_N,), reader_id
        assert dtype == torch.float32, reader_id
        assert nbytes == PAYLOAD_BYTES, reader_id


# 2. 1x fabric: dedup crosses the fabric once; naive is m x; dedup < naive.
@pytest.mark.parametrize("mode", MODES)
def test_fabric_dedup_is_1x_naive_is_mx(mode):
    m = 3
    dedup = run_dedup_burst(num_readers=m, fanout_cap=1, mode=mode)
    naive = run_naive_burst(num_readers=m, mode=mode)

    union = PAYLOAD_BYTES  # the key crosses the fabric exactly once
    assert dedup.metrics.fabric_bytes == union
    assert naive.metrics.fabric_bytes == m * union  # full replication baseline
    assert dedup.metrics.fabric_bytes < naive.metrics.fabric_bytes
    # Both still deliver the full payload to every reader (dedup saves *fabric*,
    # not delivered bytes).
    assert dedup.metrics.total_get_bytes == naive.metrics.total_get_bytes == m * union


# 3. 1x fabric holds for any fan-out cap.
@pytest.mark.parametrize("mode", MODES)
def test_fabric_dedup_1x_independent_of_fanout_cap(mode):
    for cap in (1, 2, 3):
        dedup = run_dedup_burst(num_readers=4, fanout_cap=cap, mode=mode)
        assert dedup.metrics.fabric_bytes == PAYLOAD_BYTES, cap


# 4. Exactly one reader pulls from the origin; the rest pull from peers.
@pytest.mark.parametrize("mode", MODES)
def test_only_one_hop_crosses_from_the_origin(mode):
    dedup = run_dedup_burst(num_readers=5, fanout_cap=1, mode=mode)
    origin_edges = [e for e in dedup.metrics.edges if e[0] == dedup.origin_id]
    assert len(origin_edges) == 1  # the single fabric hop
    # Every other transfer's source is a reader-side peer, not the origin.
    peer_edges = [e for e in dedup.metrics.edges if e[0] != dedup.origin_id]
    assert len(peer_edges) == 4
    assert all(src != dedup.origin_id for (src, _dst, _k) in peer_edges)


# 5. Fan-out cap shapes the topology: cap 1 = chain, cap 2 = tree.
def test_fanout_cap1_is_a_chain():
    dedup = run_dedup_burst(num_readers=4, fanout_cap=1)
    fanout = Counter(src for (src, _dst, _k) in dedup.metrics.edges)
    assert max(fanout.values()) == 1  # every source serves at most one reader


def test_fanout_cap2_builds_a_tree():
    dedup = run_dedup_burst(num_readers=4, fanout_cap=2)
    fanout = Counter(src for (src, _dst, _k) in dedup.metrics.edges)
    assert max(fanout.values()) == 2  # a source fans out to two peers


# 6. Determinism: two runs yield byte-identical traces (default + seeded).
@pytest.mark.parametrize("mode", MODES)
def test_trace_is_byte_identical_across_runs(mode):
    a = run_dedup_burst(num_readers=3, fanout_cap=1, mode=mode)
    b = run_dedup_burst(num_readers=3, fanout_cap=1, mode=mode)
    assert a.trace.render() == b.trace.render()
    assert a.trace.events == b.trace.events


@pytest.mark.parametrize("mode", MODES)
def test_trace_is_byte_identical_for_fixed_seed(mode):
    a = run_dedup_burst(num_readers=4, fanout_cap=2, random_seed=7, mode=mode)
    b = run_dedup_burst(num_readers=4, fanout_cap=2, random_seed=7, mode=mode)
    assert a.trace.render() == b.trace.render()


# 7. All readers complete (no hangs / unresolved routing).
@pytest.mark.parametrize("mode", MODES)
def test_all_readers_complete(mode):
    for cap in (1, 2, 3):
        res = run_dedup_burst(num_readers=5, fanout_cap=cap, mode=mode)
        assert res.metrics.readers_done == res.metrics.readers_total == 5


# 7b. Divergence gate: the opt-in dict-shim directory yields byte-identical
#     trace + payoff metrics vs the real Trie directory (Task B). The shim runs
#     the same real Controller decision logic over a plain dict, so the dedup 1x
#     routing, fabric bytes, and delivered bytes must match exactly.
@pytest.mark.parametrize("mode", MODES)
def test_shim_directory_matches_real(mode):
    real = run_dedup_burst(num_readers=3, fanout_cap=1, mode=mode, real_directory=True)
    shim = run_dedup_burst(num_readers=3, fanout_cap=1, mode=mode, real_directory=False)
    assert shim.trace.render() == real.trace.render()
    assert shim.metrics.fabric_bytes == real.metrics.fabric_bytes
    assert shim.metrics.total_get_bytes == real.metrics.total_get_bytes
    assert shim.metrics.wallclock == real.metrics.wallclock
    assert sorted(shim.metrics.edges) == sorted(real.metrics.edges)


# 7c. Contention gate (Task D): the payoff metric -- 1x fabric -- is invariant to
#     the network/storage contention model, which only changes *timing*. The
#     dedup burst still crosses the fabric exactly once under none/serialize/
#     progressive, and each mode is deterministic run-to-run.
CONTENTION = ("none", "serialize", "progressive")


@pytest.mark.parametrize("contention", CONTENTION)
def test_dedup_stays_1x_under_every_contention_mode(contention):
    with config.overrides(contention=contention):
        dedup = run_dedup_burst(num_readers=4, fanout_cap=1)
    assert dedup.metrics.fabric_bytes == PAYLOAD_BYTES  # 1x union, any mode
    assert dedup.metrics.readers_done == dedup.metrics.readers_total == 4


@pytest.mark.parametrize("contention", CONTENTION)
def test_dedup_trace_is_byte_identical_per_contention_mode(contention):
    with config.overrides(contention=contention):
        a = run_dedup_burst(num_readers=3, fanout_cap=1)
        b = run_dedup_burst(num_readers=3, fanout_cap=1)
    assert a.trace.render() == b.trace.render()


# 8. Allocation-free carriers survive the dedup path.
def test_meta_mode_allocates_no_storage():
    res = run_dedup_burst(num_readers=3, fanout_cap=1, mode=MODE_META)
    assert isinstance(res.expected, torch.Tensor)
    assert res.expected.device.type == "meta"
    for payload in res.results.values():
        assert isinstance(payload, torch.Tensor)
        assert payload.device.type == "meta"
        assert payload.data_ptr() == 0  # zero real allocation


def test_metadata_mode_carries_only_a_descriptor():
    res = run_dedup_burst(num_readers=3, fanout_cap=1, mode=MODE_METADATA)
    assert isinstance(res.expected, TensorDescriptor)
    for payload in res.results.values():
        assert isinstance(payload, TensorDescriptor)
