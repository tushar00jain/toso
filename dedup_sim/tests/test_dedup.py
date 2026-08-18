"""Deterministic tests for the dedup capability on the real TorchStore directory.

These assert the dedup *outcome* on the real directory/client (not wall-clock
timing): every reader receives the right payload, the fabric is the 1x union (each
unique byte crosses the fabric once) versus ``m x`` for the naive baseline, the
fan-out cap shapes a chain/tree, and the trace is byte-identical across runs.

The data plane is allocation-free (meta tensor / descriptor carriers, see
``docs/des_design.md``), so correctness is asserted on
shape/dtype/nbytes rather than exact bytes -- the exact-byte reassembly guarantee
of the real client lives in ``realsim/tests/test_correctness.py``.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests -q
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest
import torch

from dedup_sim.tests._run import run
from proposed import Dispatcher, Stored
from putget_sim.workload.put_get import DEFAULT_N, MODE_META, MODE_METADATA
from realsim.seams.transport import TensorDescriptor
from sim_common import config
from sim_common.async_engine import run_sim

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


def test_a_chain_deeper_than_the_fabric_charge_starts_a_new_one():
    """Where 1x stops, which is a trade the price makes rather than a promise it keeps.

    A peer wins while the wait behind it costs less than the fabric a fresh hop burns
    (:mod:`dedup_sim.control._selector`), so a cap-1 chain folds readers in until it is
    seven deep and then reads a holder again -- a 64-hop chain being the worse answer,
    not the safer one. Any cap above 1 keeps the tree logarithmic, so it never reaches
    that depth and the fabric is 1x for a burst of any size.
    """
    assert run(num_readers=7, fanout_cap=1).ledger.origin_bytes == PAYLOAD_BYTES
    assert run(num_readers=8, fanout_cap=1).ledger.origin_bytes == 2 * PAYLOAD_BYTES
    assert run(num_readers=64, fanout_cap=2).ledger.origin_bytes == PAYLOAD_BYTES


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
#    and by a burst loop inside the selector. Both are gone: the readers run an
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
    """Dedup and the baseline run the *same* scenario code, selector aside."""
    import ast
    import inspect

    from putget_sim.workload.put_get import PutGetBurst

    from dedup_sim.workload import scenarios

    tree = ast.parse(inspect.getsource(scenarios))
    # The capability contributes a selector and a data plane; the burst itself is
    # putget_sim's fixture. So the scenario stages nothing of its own: no
    # coroutine, hence no gather, no await, no execution order to get wrong.
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.Await, ast.AsyncFunctionDef, ast.AsyncFor))
    ]
    # ...and every run is literally the same workload object, one selector apart:
    # the baseline and each routed cap cannot differ in what they simulate.
    runs = scenarios.Dedup().runs()
    assert [r.label for r in runs] == ["baseline", "cap=1", "cap=2"]
    assert all(isinstance(r.workload, PutGetBurst) for r in runs)
    assert len({id(r.workload) for r in runs}) == 1
    assert runs[0].control is None and runs[0].data is None
    assert all(r.control is not None and r.data is not None for r in runs[1:])


# --------------------------------------------------------------------------
# The waiting, which is the commit notification and nothing else. These are the
# properties a hand-rolled latch gets wrong, so they are asserted directly rather
# than only through a burst that happens to exercise them.
#
# A gate observes the directory, then waits only for publications still missing.
# ``landed`` stands in for the directory state.
# --------------------------------------------------------------------------

#: The action these gates wait for.
_FACT = Stored("v0", "K")


def _dispatcher(landed: set):
    """A dispatcher whose one reducer adds what it is told to ``landed``."""

    class _Directory:
        folds = {Stored: lambda action: landed.add((action.host, action.key))}

    dispatcher = Dispatcher()
    dispatcher.compose(_Directory())
    return dispatcher


def test_a_fact_the_directory_already_holds_needs_no_gate():
    """The whole point of the read: never wait for what is already true."""

    async def _ask():
        return Dispatcher().gate(lambda: True, ())

    gate, _trace = run_sim(_ask())
    assert gate is None


def test_a_gate_opens_at_the_commit_that_makes_its_read_true():
    """Committed *after* the waiter parked: it is released, not stranded.

    A commit for one required fact leaves it parked until the probe sees both.
    """

    async def _wait() -> list:
        landed: set = set()
        dispatcher = _dispatcher(landed)
        gate = dispatcher.gate(
            lambda: {("v0", "K"), ("v1", "K")} <= landed,
            (_FACT, Stored("v1", "K")),
        )
        assert gate is not None
        order: list[str] = []

        async def waiter() -> None:
            await gate()
            order.append("released")

        task = asyncio.get_running_loop().create_task(waiter())
        await asyncio.sleep(0)
        dispatcher.dispatch_sync(_FACT)
        await asyncio.sleep(0)
        assert order == [], "released before every fact was true"
        dispatcher.dispatch_sync(Stored("v1", "K"))
        await task
        return order

    order, _trace = run_sim(_wait())
    assert order == ["released"]


def test_only_the_last_named_commit_wakes_the_gate():
    async def _wait() -> int:
        landed: set = set()
        dispatcher = _dispatcher(landed)
        probes = 0

        def holds() -> bool:
            nonlocal probes
            probes += 1
            return {("v0", "K"), ("v1", "K")} <= landed

        gate = dispatcher.gate(holds, (_FACT, Stored("v1", "K")))
        assert gate is not None
        task = asyncio.create_task(gate())
        await asyncio.sleep(0)
        assert probes == 1

        dispatcher.dispatch_sync(Stored("v1", "other"))
        await asyncio.sleep(0)
        assert probes == 1

        dispatcher.dispatch_sync(_FACT)
        await asyncio.sleep(0)
        assert not task.done(), "released with one publication still missing"
        assert probes == 1, "re-probed after a relevant commit"

        dispatcher.dispatch_sync(Stored("v1", "K"))
        await task
        return probes

    probes, _trace = run_sim(_wait())
    assert probes == 1


def test_a_commit_between_the_read_and_the_wait_is_not_lost():
    """The lost-wakeup guard, and the reason the commit is captured first.

    A waiter that read "not yet" and *then* looked for something to wait on would
    miss a commit landing in between and park for the rest of the run. Because the
    event it waits on is the one it captured before reading, that commit sets an
    event it already holds.
    """

    async def _race() -> str:
        landed: set = set()
        dispatcher = _dispatcher(landed)
        reads = []

        def holds() -> bool:
            reads.append(len(landed))
            # The put lands "during" the read: what this read reports is the state
            # from before it, so the answer it gives back is already stale.
            if len(reads) == 1:
                dispatcher.dispatch_sync(_FACT)
                return False
            return ("v0", "K") in landed

        gate = dispatcher.gate(holds, (_FACT,))
        assert gate is not None, "the first read said not yet, which it did"
        await asyncio.wait_for(gate(), timeout=None)
        return "released"

    result, _trace = run_sim(_race())
    assert result == "released", "parked on a commit that had already happened"


def test_released_gates_leave_no_waiter_registration():
    """The action map contains only currently parked waiters."""

    async def _versions() -> dict:
        landed: set = set()
        dispatcher = _dispatcher(landed)
        for version in range(100):
            key = f"W{version}"
            gate = dispatcher.gate(
                lambda k=key: ("v0", k) in landed,
                (Stored("v0", key),),
            )
            assert gate is not None
            task = asyncio.create_task(gate())
            await asyncio.sleep(0)
            dispatcher.dispatch_sync(Stored("v0", key))
            await task
        return dispatcher._waiters

    waiters, _trace = run_sim(_versions())
    assert waiters == {}


def test_a_registration_that_was_evicted_does_not_open_a_later_gate():
    """The stale-routing guard: "committed once" is not "true from now on".

    A volume that registered a key and later dropped it (a new version displacing
    the old) does not hold it. Answering the next requester from the memory of that
    commit would route it to a volume with nothing to serve -- and there is no such
    memory to answer from, because a gate is only ever the caller's own read.
    """

    async def _evict_then_ask():
        landed: set = set()
        dispatcher = _dispatcher(landed)
        holds = lambda: ("v0", "K") in landed          # noqa: E731
        assert dispatcher.gate(holds, (_FACT,)) is not None
        dispatcher.dispatch_sync(_FACT)                     # v0's put landed
        assert dispatcher.gate(holds, (_FACT,)) is None
        landed.discard(("v0", "K"))                    # a newer version displaces it
        return dispatcher.gate(holds, (_FACT,))

    gate, _trace = run_sim(_evict_then_ask())
    assert gate is not None, "answered from memory, not from the directory"
