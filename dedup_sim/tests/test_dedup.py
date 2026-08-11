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

from dedup_sim.tests._run import run
from putget_sim.workload.put_get import DEFAULT_N, MODE_META, MODE_METADATA
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
    res = run(num_readers=4, fanout_cap=1, mode=mode)
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
    dedup = run(num_readers=m, fanout_cap=1, mode=mode)
    naive = run(num_readers=m, mode=mode)

    union = PAYLOAD_BYTES  # the key crosses the fabric exactly once
    assert dedup.ledger.origin_bytes == union
    assert naive.ledger.origin_bytes == m * union  # full replication baseline
    assert dedup.ledger.origin_bytes < naive.ledger.origin_bytes
    # Both still deliver the full payload to every reader (dedup saves *fabric*,
    # not delivered bytes).
    assert dedup.ledger.transfer_bytes == naive.ledger.transfer_bytes == m * union


# 3. 1x fabric holds for any fan-out cap.
@pytest.mark.parametrize("mode", MODES)
def test_fabric_dedup_1x_independent_of_fanout_cap(mode):
    for cap in (1, 2, 3):
        dedup = run(num_readers=4, fanout_cap=cap, mode=mode)
        assert dedup.ledger.origin_bytes == PAYLOAD_BYTES, cap


# 4. Exactly one reader pulls from the origin; the rest pull from peers.
@pytest.mark.parametrize("mode", MODES)
def test_only_one_hop_crosses_from_the_origin(mode):
    dedup = run(num_readers=5, fanout_cap=1, mode=mode)
    origin_edges = [e for e in dedup.ledger.edges if e[0] == dedup.workload.origin_id]
    assert len(origin_edges) == 1  # the single fabric hop
    # Every other transfer's source is a reader-side peer, not the origin.
    peer_edges = [e for e in dedup.ledger.edges if e[0] != dedup.workload.origin_id]
    assert len(peer_edges) == 4
    assert all(src != dedup.workload.origin_id for (src, _dst, _k) in peer_edges)


# 5. Fan-out cap shapes the topology: cap 1 = chain, cap 2 = tree.
def test_fanout_cap1_is_a_chain():
    dedup = run(num_readers=4, fanout_cap=1)
    fanout = Counter(src for (src, _dst, _k) in dedup.ledger.edges)
    assert max(fanout.values()) == 1  # every source serves at most one reader


def test_fanout_cap2_builds_a_tree():
    dedup = run(num_readers=4, fanout_cap=2)
    fanout = Counter(src for (src, _dst, _k) in dedup.ledger.edges)
    assert max(fanout.values()) == 2  # a source fans out to two peers


# 6. Determinism: two runs yield byte-identical traces (default + seeded).
@pytest.mark.parametrize("mode", MODES)
def test_trace_is_byte_identical_across_runs(mode):
    a = run(num_readers=3, fanout_cap=1, mode=mode)
    b = run(num_readers=3, fanout_cap=1, mode=mode)
    assert a.trace.render() == b.trace.render()
    assert a.trace.events == b.trace.events


@pytest.mark.parametrize("mode", MODES)
def test_trace_is_byte_identical_for_fixed_seed(mode):
    a = run(num_readers=4, fanout_cap=2, random_seed=7, mode=mode)
    b = run(num_readers=4, fanout_cap=2, random_seed=7, mode=mode)
    assert a.trace.render() == b.trace.render()


# 7. All readers complete (no hangs / unresolved routing).
@pytest.mark.parametrize("mode", MODES)
def test_all_readers_complete(mode):
    for cap in (1, 2, 3):
        res = run(num_readers=5, fanout_cap=cap, mode=mode)
        assert res.ledger.items_done == res.ledger.items_total == 5


# 7b. Divergence gate: the opt-in dict-shim directory yields byte-identical
#     trace + payoff metrics vs the real Trie directory (Task B). The shim runs
#     the same real Controller decision logic over a plain dict, so the dedup 1x
#     routing, fabric bytes, and delivered bytes must match exactly.
@pytest.mark.parametrize("mode", MODES)
def test_shim_directory_matches_real(mode):
    real = run(num_readers=3, fanout_cap=1, mode=mode, real_directory=True)
    shim = run(num_readers=3, fanout_cap=1, mode=mode, real_directory=False)
    assert shim.trace.render() == real.trace.render()
    assert shim.ledger.origin_bytes == real.ledger.origin_bytes
    assert shim.ledger.transfer_bytes == real.ledger.transfer_bytes
    assert shim.ledger.wallclock == real.ledger.wallclock
    assert sorted(shim.ledger.edges) == sorted(real.ledger.edges)


# 7c. Contention gate (Task D): the payoff metric -- 1x fabric -- is invariant to
#     the network/storage contention model, which only changes *timing*. The
#     dedup burst still crosses the fabric exactly once under none/serialize/
#     progressive, and each mode is deterministic run-to-run.
CONTENTION = ("none", "serialize", "progressive")


@pytest.mark.parametrize("contention", CONTENTION)
def test_dedup_stays_1x_under_every_contention_mode(contention):
    with config.overrides(contention=contention):
        dedup = run(num_readers=4, fanout_cap=1)
    assert dedup.ledger.origin_bytes == PAYLOAD_BYTES  # 1x union, any mode
    assert dedup.ledger.items_done == dedup.ledger.items_total == 4


@pytest.mark.parametrize("contention", CONTENTION)
def test_dedup_trace_is_byte_identical_per_contention_mode(contention):
    with config.overrides(contention=contention):
        a = run(num_readers=3, fanout_cap=1)
        b = run(num_readers=3, fanout_cap=1)
    assert a.trace.render() == b.trace.render()


# 7d. Collapse-charges gate: coalescing an op's per-component sleeps (a get's
#     storage+mem+network into one) is a timing coarsening on the non-contended
#     path -- it does not depend on sub-charge interleaving, so the dedup payoff
#     metric (1x fabric, delivered bytes) is unchanged vs collapse-off, and it
#     stays deterministic run-to-run.
@pytest.mark.parametrize("mode", MODES)
def test_dedup_1x_fabric_invariant_to_collapse(mode):
    with config.overrides(collapse_charges=False):
        off = run(num_readers=4, fanout_cap=1, mode=mode)
    with config.overrides(collapse_charges=True):
        on = run(num_readers=4, fanout_cap=1, mode=mode)
    assert on.ledger.origin_bytes == off.ledger.origin_bytes == PAYLOAD_BYTES
    assert on.ledger.transfer_bytes == off.ledger.transfer_bytes
    assert sorted(on.ledger.edges) == sorted(off.ledger.edges)


def test_dedup_collapse_is_deterministic():
    with config.overrides(collapse_charges=True):
        a = run(num_readers=3, fanout_cap=1)
        b = run(num_readers=3, fanout_cap=1)
    assert a.trace.render() == b.trace.render()


# 8. Allocation-free carriers survive the dedup path.
def test_meta_mode_allocates_no_storage():
    res = run(num_readers=3, fanout_cap=1, mode=MODE_META)
    assert isinstance(res.workload.expected, torch.Tensor)
    assert res.workload.expected.device.type == "meta"
    for payload in res.results.values():
        assert isinstance(payload, torch.Tensor)
        assert payload.device.type == "meta"
        assert payload.data_ptr() == 0  # zero real allocation


def test_metadata_mode_carries_only_a_descriptor():
    res = run(num_readers=3, fanout_cap=1, mode=MODE_METADATA)
    assert isinstance(res.workload.expected, TensorDescriptor)
    for payload in res.results.values():
        assert isinstance(payload, TensorDescriptor)


# 9. The routing lives in the controller, not in a lie told to the client.
#
#    The 1x chain used to be produced by swapping each reader's private
#    ``client._controller`` for a handle that narrowed the real directory answer,
#    and by a burst loop inside the policy. Both are gone: the readers run an
#    untouched real client over the mesh's own controller handle, and the chain
#    is a consequence of the controller withholding each answer until the planned
#    peer registers. These two tests pin that, because it is the whole point of
#    the change -- the 1x number alone would still pass with the monkeypatch.
def test_readers_run_an_untouched_real_client():
    result = run(3, fanout_cap=1)
    mesh = result.sim.mesh

    for reader_id in result.workload.reader_ids:
        # The one controller handle the mesh built, not a per-reader view of it.
        assert mesh.client(reader_id)._controller is mesh.controller_handle


def test_the_scenario_holds_no_burst_loop():
    """Dedup and the baseline run the *same* scenario code, policy aside."""
    import ast
    import inspect

    from putget_sim.workload.put_get import PutGetBurst

    from dedup_sim.workload import scenarios

    tree = ast.parse(inspect.getsource(scenarios))
    # The capability contributes a policy and a data plane; the burst itself is
    # putget_sim's fixture. So the scenario stages nothing of its own: no
    # coroutine, hence no gather, no await, no execution order to get wrong.
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.Await, ast.AsyncFunctionDef, ast.AsyncFor))
    ]
    # ...and every run is literally the same workload object, one policy apart:
    # the baseline and each routed cap cannot differ in what they simulate.
    runs = scenarios.Dedup().runs()
    assert [r.label for r in runs] == ["baseline", "cap=1", "cap=2"]
    assert all(isinstance(r.workload, PutGetBurst) for r in runs)
    assert len({id(r.workload) for r in runs}) == 1
    assert runs[0].control is None and runs[0].data is None
    assert all(r.control is not None and r.data is not None for r in runs[1:])
