"""Determinism + invariants for the realsim read-burst on the deterministic engine.

The data plane is allocation-free (see ``docs/realsim_design.md`` s7),
so these DES tests assert only **shape/dtype/nbytes + trace determinism** -- the
exact-byte reassembly guarantee lives off the sim path in
``test_correctness.py`` (tiny real CPU tensors). Both carriers are exercised:

* ``mode="meta"`` (default) -- W is a ``device="meta"`` tensor (real
  ``torch.Tensor``, zero storage, exact shape/dtype);
* ``mode="metadata"`` -- W is a ``TensorDescriptor`` (no tensor at all).

Guarantees asserted:

(a) **Byte-identical traces** -- running a scenario twice (default FIFO and a
    fixed ``random_seed``) produces exactly the same trace, for both modes (the
    hard determinism requirement).

(b) **Invariants under random scheduling** -- across a couple of seeds every
    reader still receives a payload of the right shape/dtype/nbytes and the
    fabric-byte accounting stays consistent (naive => ``m x`` payload).
"""

from __future__ import annotations

import pytest
import torch

from putget_sim.harness import run_burst
from putget_sim.workload.put_get import DEFAULT_N, MODE_META, MODE_METADATA
from realsim.seams.transport import TensorDescriptor

MODES = (MODE_META, MODE_METADATA)


def _shape_dtype_nbytes(payload):
    """Uniform ``(shape, dtype, nbytes)`` for a meta tensor or a descriptor."""
    if isinstance(payload, torch.Tensor):
        return tuple(payload.shape), payload.dtype, payload.numel() * payload.element_size()
    assert isinstance(payload, TensorDescriptor)
    return payload.shape, payload.dtype, payload.nbytes


@pytest.mark.parametrize("mode", MODES)
def test_trace_is_byte_identical_across_runs(mode):
    a = run_burst(num_readers=3, mode=mode)
    b = run_burst(num_readers=3, mode=mode)
    # The whole trace string (engine scheduling rows + scenario + transport)
    # must match to the byte.
    assert a.trace.render() == b.trace.render()
    # Events, not just the rendered string.
    assert a.trace.events == b.trace.events


@pytest.mark.parametrize("mode", MODES)
def test_trace_is_byte_identical_for_fixed_seed(mode):
    a = run_burst(num_readers=4, random_seed=7, mode=mode)
    b = run_burst(num_readers=4, random_seed=7, mode=mode)
    assert a.trace.render() == b.trace.render()


def test_meta_mode_allocates_no_storage():
    # The default carrier is a real tensor with zero storage on the meta device.
    res = run_burst(num_readers=2, mode=MODE_META)
    assert isinstance(res.expected, torch.Tensor)
    assert res.expected.device.type == "meta"
    for payload in res.results.values():
        assert isinstance(payload, torch.Tensor)
        assert payload.device.type == "meta"
        # A meta tensor has exact shape/nbytes metadata but no real memory
        # backing it (null data pointer) -- i.e. zero real allocation.
        assert payload.data_ptr() == 0


def test_metadata_mode_carries_only_a_descriptor():
    # The metadata-only carrier is a descriptor -- no tensor object at all.
    res = run_burst(num_readers=2, mode=MODE_METADATA)
    assert isinstance(res.expected, TensorDescriptor)
    for payload in res.results.values():
        assert isinstance(payload, TensorDescriptor)


@pytest.mark.parametrize("mode", MODES)
def test_invariants_hold_across_seeds(mode):
    exp_shape = (DEFAULT_N,)
    exp_dtype = torch.float32
    payload_nbytes = DEFAULT_N * 4  # DEFAULT_N float32 elements

    for seed in (None, 0, 1, 7):
        res = run_burst(num_readers=4, random_seed=seed, mode=mode)

        # Every reader received a payload of the right shape/dtype/nbytes (the
        # allocation-free analogue of the old exact-byte equality check).
        assert set(res.results) == {"r0", "r1", "r2", "r3"}
        for reader_id, payload in res.results.items():
            shape, dtype, nbytes = _shape_dtype_nbytes(payload)
            assert shape == exp_shape, reader_id
            assert dtype == exp_dtype, reader_id
            assert nbytes == payload_nbytes, reader_id

        # All readers completed.
        assert res.ledger.items_done == res.ledger.items_total == 4

        # Naive policy: each of m readers pulls the full payload from the origin,
        # so fabric == total delivered == m * payload (the m x baseline).
        assert res.ledger.transfer_bytes == 4 * payload_nbytes
        assert res.ledger.origin_bytes == 4 * payload_nbytes

        # The fetch tree is the origin fanning out to every reader.
        srcs = {src for (src, _dst, _k) in res.ledger.edges}
        dsts = {dst for (_src, dst, _k) in res.ledger.edges}
        assert srcs == {res.origin_id}
        assert dsts == {"volr0", "volr1", "volr2", "volr3"}


@pytest.mark.parametrize("mode", MODES)
def test_wallclock_reflects_overlap_not_sum(mode):
    # Concurrent same-size transfers from the origin overlap under the engine, so
    # the burst wallclock is one transfer's cost, not the sum of m of them.
    one = run_burst(num_readers=1, mode=mode)
    many = run_burst(num_readers=5, mode=mode)
    assert many.ledger.wallclock == one.ledger.wallclock
    assert many.ledger.wallclock > 0.0
