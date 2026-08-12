"""Deterministic tests for the KV-cache simulation on the *real* TorchStore.

Run from the parent directory: ``python -m pytest kvcache_sim/tests -q``.
Assertions are on the simulation *outcome* (block presence in the real directory,
prefix hit rate, prefill compute, eviction, rejections, TBT, determinism) -- never
on wall-clock timing. Every scenario drives the real ``Controller`` directory + real
per-instance clients on the shared deterministic async engine.
"""

from __future__ import annotations

import asyncio

import pytest

from sim_common import config
from sim_common.async_engine import AsyncEngine, run_sim

from kvcache_sim.control._view import KVView, _longest_prefix_run
from kvcache_sim.workload._accelerator import BLOCK_TOKENS, SimulatedAccelerator
from realsim.simulation import Simulation
from sim_common.cost_model import DEFAULT_PROFILE
from domain import decode_step_time
from kvcache_sim.data._decode import DecodeEngine
from kvcache_sim.data.store import KVStore
from kvcache_sim.control.request import Request
from kvcache_sim.control.scheduler import (
    AdmitDecode, ComputeBusy, DecodeState, LoadBalanceScheduler,
    PrefillFinished, Route,
)
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


def _kv(count: int, block_tokens: int = BLOCK_TOKENS):
    """``count`` KV blocks, exactly as a prefill on this run's accelerator makes them.

    A publish takes the blocks now (the store holds no notion of what one is), so a
    test that wants keys present has to produce KV for them. It does that through
    the same object the serving plane does rather than conjuring a tensor of its
    own, so a test can never publish a block of a size the run would not.
    """
    return SimulatedAccelerator(block_tokens=block_tokens).kv_blocks(count)


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
async def _evict(deployment, inst: str, keys: list) -> None:
    """Evict ``keys`` from ``inst`` the way a full volume does.

    Both halves, because either alone is a state the store never produces: the
    volume drops the bytes (its own ``delete``, which releases what the key owns)
    and then tells the directory. There is no verb for this in the data plane --
    which key to drop is the volume's own decision, taken when a put does not fit
    -- so a test that wants the outcome without the capacity pressure does what
    the volume would have done.
    """
    await deployment.volume_handle(inst).delete_batch.call_one(list(keys))
    await deployment.controller_handle.notify_delete_batch.call_one(
        {inst: list(keys)}
    )


def test_real_directory_prefix_presence_and_eviction():
    topo = _make_topology(2)
    keys = _block_keys_for("m0", [0, 1, 2, 3])

    sim = Simulation(topo)
    store = KVStore(sim.mesh)
    view = KVView(sim.view.directory, sim.topology)

    async def scenario():
        # The data plane publishes/evicts; the control-plane view reads back.
        with sim.mesh.installed():
            await store.publish("s0", list(keys[:3]), _kv(3))  # s0: 3 leading blocks
            await store.publish("s1", list(keys[:1]), _kv(1))  # s1 holds 1
            counts = await view.prefix_lengths(list(keys))
            assert counts == {"s0": 3, "s1": 1}
            await _evict(sim.mesh, "s0", [keys[1]])    # break s0's run at index 1
            counts2 = await view.prefix_lengths(list(keys))
            assert counts2 == {"s0": 1, "s1": 1}
        return True

    try:
        ok = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert ok


# 2b. A pull is all-or-nothing. A plan is made when the request arrives and the
#     pull happens after the prefill queue, so a peer can drop a planned block in
#     between. The fetch then raises and moves nothing, which is what lets the
#     serving plane recompute the whole prefix instead of pulling a hole through
#     it -- half a prefix is not a prefix, and quietly fetching the survivors
#     would charge the request for a reuse it did not get.
def test_a_fetch_whose_block_vanished_raises_and_moves_nothing():
    topo = _make_topology(2)
    keys = _block_keys_for("m0", [0, 1])

    sim = Simulation(topo)
    store = KVStore(sim.mesh)

    async def scenario():
        with sim.mesh.installed():
            await store.publish("s0", list(keys), _kv(len(keys)))
            await _evict(sim.mesh, "s0", [keys[1]])
            before = sim.ledger.transfer_bytes
            with pytest.raises(KeyError):
                await store.fetch("s1", list(keys))
            return sim.ledger.transfer_bytes - before

    try:
        moved = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert moved == 0  # it fails at the locate, before anything crosses


# 3. LRU now lives in the volume -- recency of a node's own data is the one thing
#    that node cannot be wrong about -- and is covered by
#    realsim/tests/test_storage_capacity.py.



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
    from dataclasses import replace as _replace

    from domain import DEFAULT_MODEL, DEFAULT_PROFILE

    from kvcache_sim.workload._serving import BLOCK_TOKENS

    unbounded = run(topo, reqs, "cache_aware").ledger.hit_rate
    big = run(
        topo, reqs, "cache_aware",
        profile=_replace(
            DEFAULT_PROFILE,
            storage_capacity_bytes=DEFAULT_MODEL.block_bytes(100000, BLOCK_TOKENS),
        ),
    ).ledger.hit_rate
    assert abs(unbounded - big) < 1e-9


# 7b. A cache fill is allowed to fail, and the run says when it did.
def test_a_prefix_that_does_not_fit_is_reported_as_not_cached():
    """``publish`` may refuse, and refusing has to be visible in the outcome.

    Only a block that cannot fit *at all* is refused: the transport writes one key
    per put, so a request whose whole working set exceeds the volume still stores
    every block -- it just evicts its own earlier ones. That makes a capacity below
    a single block the one place ``StorageFull`` reaches the serving loop, and the
    place to check the answer is not dropped: the request is still served, but
    nothing is cached and no later request can reuse it.
    """
    from dataclasses import replace as _replace

    from domain import DEFAULT_MODEL, DEFAULT_PROFILE

    from kvcache_sim.workload._serving import BLOCK_TOKENS

    topo, reqs = _make_topology(4), _shared_prefix_workload()
    one_block = DEFAULT_MODEL.block_bytes(1, BLOCK_TOKENS)

    def _at(capacity_bytes: int):
        return run(
            topo, reqs, "cache_aware",
            profile=_replace(
                DEFAULT_PROFILE, storage_capacity_bytes=capacity_bytes
            ),
        ).ledger

    fits = _at(one_block)
    assert fits.unpublished == 0
    assert all(r.published for r in fits.accepted)

    refused = _at(one_block - 1)
    # Every request is still served -- a cache fill failing is not a request
    # failing -- and every one of them reports that it cached nothing.
    assert len(refused.accepted) == len(fits.accepted)
    assert refused.unpublished == len(refused.accepted)
    assert refused.hit_rate == 0.0


# 7c. A request is served where the coordinator says, not where it landed.
def test_a_request_is_served_by_its_plan_host_not_the_host_it_landed_on():
    """Every host routes, so where a request lands does not decide where it runs.

    Arrival is client affinity -- a load balancer's answer, made without looking
    at the cache -- so most requests land somewhere other than the host holding
    their prefix, and the host they land on redirects them. If arrival decided
    placement instead, cache-aware routing would be measuring its own load
    balancer.

    The two counts have to agree: one redirect is traced for exactly the requests
    whose serving host is not their arrival host.
    """
    from kvcache_sim.workload._serving import _affinity

    topo, reqs = _make_topology(4), _shared_prefix_workload()
    result = run(topo, reqs, "cache_aware")

    landed = _affinity(sorted(topo))
    by_id = {r.id: r for r in reqs}
    moved = [
        row for row in result.ledger.accepted
        if row.prefill != landed(by_id[row.id])
    ]
    redirects = [
        line for line in result.trace.render().splitlines() if " REDIR " in line
    ]
    assert moved, "every request happened to land on its own prefill host"
    assert len(moved) == len(redirects)
    # ...and affinity is a function of the conversation alone, so a conversation's
    # requests all land together however their prefixes differ.
    for request in reqs:
        assert landed(request) == landed(by_id[request.id])
    homes = {r.conversation: landed(r) for r in reqs}
    assert all(landed(r) == homes[r.conversation] for r in reqs)


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


# 10. Fan-out sanity: the LRU primitive still honours a bound when given one, and
#     a comfortably sized run still reuses. The *scheduler* no longer passes one:
#     what a volume can hold is the volume's own capacity, enforced where it is
#     known, so control's copy is a recency model rather than a second limit.


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
def _run_decode_batch(n: int, output_tokens: int = 6):
    """Admit ``n`` requests at t=0 on one host; return {id -> worst TBT}.

    Drives the decode engine against a bare clock: it needs no store, no
    directory and no topology, so assembling a Simulation would build a mesh for
    nothing. One engine is one host's decode side, so there is no instance to name.

    Waiting is done the way the serving host waits -- on the completions ``admit``
    answers with -- rather than through an engine-level drain, which no longer
    exists precisely because every admitted request already has somebody holding
    it.
    """
    loop = AsyncEngine()
    res = {}
    eng = DecodeEngine(
        SimulatedAccelerator(),
        max_batch=8,
        on_finish=lambda r, tbt: res.__setitem__(r.id, tbt),
    )

    async def drive():
        await asyncio.gather(*[
            eng.admit(
                Request(id=f"r{i}", arrival=0.0, block_keys=("m0|0",),
                        prompt_tokens=512, output_tokens=output_tokens)
            )
            for i in range(n)
        ])

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


# 12b. A batch always drains, including the part of it that never fit.
#      Over the VRAM cap, so four of the twelve start in ``pending`` and can only
#      run once a slot frees. This is the one place the "every admitted request
#      finishes" claim could fail structurally rather than by arithmetic -- a
#      queued request whose promotion never came would be a caller parked
#      forever, which is a hung run rather than a failed assertion. Waiting on all
#      twelve completions at once is exactly the shape twelve client coroutines
#      have.
def test_a_request_that_did_not_fit_the_batch_still_finishes():
    finished = _run_decode_batch(12)      # max_batch=8 inside the helper
    assert len(finished) == 12
    assert all(tbt > 0 for tbt in finished.values())


# 12c. A request with no decode step to run is retired inside ``admit``, and its
#      caller is released on the same clock instant rather than waiting for a step
#      loop that will never start. The prefill produced the first token, so
#      ``output_tokens=1`` leaves nothing for decode to do.
def test_a_request_with_no_decode_tokens_finishes_immediately():
    finished = _run_decode_batch(3, output_tokens=1)
    assert finished == {"r0": 0.0, "r1": 0.0, "r2": 0.0}


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


# 13b. The handoff between prefill and decode goes through the STORE.
#      A decode host that did not prefill the prompt holds none of its KV, so it
#      has to fetch the chain back out -- a real get_batch, priced like every other
#      transfer. This used to be a method call on the peer object, i.e. free, which
#      flattered disaggregation by exactly the cost that dominates a real
#      prefill/decode-disaggregated deployment.
def test_the_decode_host_fetches_its_kv_out_of_the_store():
    disagg, coupled = run_disaggregation()
    for result in (disagg, coupled):
        ledger = result.ledger
        assert ledger.handoff_bytes > 0
        # Nothing went missing: if it had, the transfer would be uncharged and the
        # bytes above would be understating what the run actually needs to move.
        assert ledger.handoff_misses == 0
        for row in ledger.accepted:
            assert row.decode, f"{row.id} decoded nowhere"

    # A host that prefilled the request decodes it on KV it already holds, so it
    # pays nothing and reports no handoff -- the store's own rule for a local hit,
    # the same one the prefill side follows through ``reuse``. Charging it would
    # bill a storage read for KV that never left, and would report it in a column
    # that means "this crossed a host boundary".
    for row in coupled.ledger.accepted:
        crossed = row.prefill != row.decode
        assert (row.handoff_bytes > 0) is crossed, row.id
    assert any(row.prefill == row.decode for row in coupled.ledger.accepted), (
        "coupled never decoded on the prefill host, so the free path is untested"
    )

    # In the disaggregated run the pools are disjoint, so *every* request's KV
    # really crossed a host boundary -- and the crossing is a transport transfer
    # the mesh charged, not an accounting line this package invented. Both decode
    # instances are the destination of one, which no run could produce before:
    # nothing ever fetched into the decode pool, because nothing had to.
    into = {dst for _src, dst, _label in disagg.ledger.edges}
    assert {"s2", "s3"} <= into
    for row in disagg.ledger.accepted:
        assert row.prefill in ("s0", "s1")
        assert row.decode in ("s2", "s3")


# 13e. The decode leg answers at the LAST TOKEN, and every path through it
#      answers. Both halves matter and only one of them is a feature: a leg that
#      answered early left the run needing a drain pass and left no coroutine on
#      the request to time it, while a leg that never answers is a client parked
#      forever -- a hung run rather than a failed test, which is the worst shape a
#      bug can take here. So the paths are enumerated rather than sampled through
#      a scenario.
def _plan(request: Request, *, prefill: str, decode: str):
    """A minimal accepted plan. This exercises the decode leg, not the router."""
    from kvcache_sim.control.scheduler import Plan

    return Plan(
        request=request, prefill=prefill, decode=decode, match_blocks=0,
        cached_tokens=0, uncached_tokens=request.prompt_tokens, reuse_source=None,
        transfer_bytes=0, queue_wait=0.0, ttft=0.0, done_time=0.0, decode_done=0.0,
    )


def _decode_leg(*, output_tokens: int = 6, prefill: str = "s0",
                published: bool = True, engine: bool = True):
    """Walk one request's decode leg on ``s1``; report when it answered.

    Returns ``(answered_at, events, row)`` -- the sim clock when
    :meth:`ServingHost.decode` returned, the host's own trace lines as
    ``{kind: [time]}`` (it writes a HANDOFF when the KV lands and a DECODE when
    the last token does), and the ledger row the two halves were joined into.

    A real coordinator is wired in because a decode batch *reports itself*: the
    host forwards every batch change to control, so a stub ``None`` would fail on
    the first admission rather than on anything this is testing.
    """
    from kvcache_sim.data.serving import ServingHost
    from kvcache_sim.report.metrics import Metrics, RequestResult

    sim = Simulation(
        _make_topology(2),
        control=LoadBalanceScheduler(
            block_tokens=BLOCK_TOKENS, simulate_decode=True
        ),
        ledger=Metrics(),
    )
    store = KVStore(sim.mesh)
    keys = list(_block_keys_for("m0", [0, 1]))
    request = Request(
        id="r0", arrival=0.0, block_keys=tuple(keys),
        prompt_tokens=2 * BLOCK_TOKENS, output_tokens=output_tokens,
    )
    host = ServingHost(
        "s1", store, sim.coordinator_handle, trace=sim.trace, metrics=sim.ledger,
        decode=(
            DecodeEngine(SimulatedAccelerator(), max_batch=8) if engine else None
        ),
        models_decode=engine,
    )

    async def scenario():
        with sim.mesh.installed():
            if published:
                await store.publish(prefill, keys, _kv(len(keys)))
            # The prefill host opens the row; here there is no prefill host, so the
            # test stands in for it -- the decode side amends, it never creates.
            sim.ledger.add(RequestResult(id=request.id, accepted=True))
            await host.decode(_plan(request, prefill=prefill, decode="s1"))
            return asyncio.get_running_loop().time()

    try:
        answered_at = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    events: dict = {}
    for t, kind, _msg in sim.trace.events:
        events.setdefault(kind, []).append(t)
    return answered_at, events, sim.ledger.results[0]


def test_the_decode_leg_answers_when_the_last_token_lands():
    """The ordinary path: KV fetched from the host that prefilled it, then steps."""
    answered_at, events, row = _decode_leg(output_tokens=6)
    assert len(events["DECODE"]) == 1
    # Answered at the last token -- not at admission, which is where the KV
    # landed and where this used to return.
    assert answered_at == events["DECODE"][0]
    # ...and the five steps between the two really ran (6 output tokens, the
    # first of which prefill produced).
    steps = 5 * decode_step_time(1, DEFAULT_PROFILE)
    assert answered_at - events["HANDOFF"][0] == pytest.approx(steps)
    assert row.tbt > 0                       # ...and the row was written first
    assert row.handoff_bytes > 0


@pytest.mark.parametrize("case,kwargs", [
    # Nothing to decode: retired inside ``admit``, on the instant it arrived.
    ("no decode tokens", dict(output_tokens=1)),
    # The chain was gone (or never cached). The request still decodes -- see the
    # handoff-miss honesty note -- so it still has to finish.
    ("handoff missed", dict(published=False)),
    # The decode host is the prefill host: nothing is fetched, nothing charged.
    ("decodes where it prefilled", dict(prefill="s1")),
])
def test_every_decode_path_answers_at_its_last_token(case, kwargs):
    answered_at, events, _row = _decode_leg(**kwargs)
    assert len(events["DECODE"]) == 1, case
    assert answered_at == events["DECODE"][0], case


def test_a_host_with_no_decode_engine_answers_without_waiting():
    """The one early answer left, and it is not a decode that was skipped.

    A run that does not model decode never sends a client here at all (prefill
    answers ``None`` and the journey ends), so this is the defensive case: a host
    asked to decode with no engine has no last token coming, and waiting for one
    would be waiting forever. It answers the moment the KV is there, which is as
    far as it can honestly get.
    """
    answered_at, events, row = _decode_leg(engine=False)
    assert "DECODE" not in events
    assert answered_at == events["HANDOFF"][0]
    assert row.tbt == 0.0


# 13f. What the client measures, and what it refuses to measure.
def test_the_client_times_every_served_request_end_to_end():
    """Arrival to last token, on the row, for exactly the requests that finished.

    Also the termination proof at scenario scale: every one of these runs returns,
    and a run where any client were parked on a token that never came would not.
    """
    disagg, coupled = run_disaggregation()
    for result in (disagg, coupled):
        rows = result.ledger.accepted
        assert rows and len(rows) == len(result.ledger.results)
        for row in rows:
            # Strictly after arrival, and strictly after the gaps it contains --
            # end-to-end spans the whole request, not one leg of it.
            assert row.latency > 0, row.id
            assert row.latency > row.tbt, row.id
        assert result.ledger.mean_latency > 0
        assert result.ledger.pct_latency(90) >= result.ledger.mean_latency * 0.5

    # A request that was shed has no last token, so it gets no fabricated
    # duration -- and, more to the point, no waiter: 'off' rejects 22 requests
    # after their prefill, and the run still ends.
    off, _early, _predict = run_early_rejection()
    shed = [r for r in off.ledger.results if r.decode_rejected]
    assert shed, "no request was shed, so the assertion would be vacuous"
    assert all(r.latency == 0.0 for r in shed)
    assert all(r.latency > 0.0 for r in off.ledger.accepted)


def test_a_run_that_does_not_model_decode_reports_no_end_to_end():
    """No last token, so no end-to-end latency -- rather than a shorter one.

    The client's walk ends at prefill in these runs, and stamping *that* under the
    same name would mean the column measured two different intervals depending on
    the scenario. So it is left unstamped, and the prefill-side reports do not
    offer the column.
    """
    cache_aware, _baseline = run_shared_prefix()
    assert cache_aware.ledger.accepted
    assert all(r.latency == 0.0 for r in cache_aware.ledger.results)
    assert cache_aware.ledger.mean_latency == 0.0


def test_the_kv_handoff_is_visible_in_end_to_end_and_nowhere_else():
    """The column exists for this: disaggregation wins TBT and loses the wall clock.

    The disaggregated run moves ~3.5x the KV of the coupled one (disjoint pools,
    so every request's chain crosses a host boundary), and it is charged for it on
    the clock. None of that reaches TTFT, which control predicted before any of it
    happened, or TBT, which is measured between tokens the transfer finishes
    before. So the run with the *better* per-token behaviour is the slower one end
    to end, and this is the only pair of columns that can say so.
    """
    disagg, coupled = run_disaggregation()
    assert disagg.ledger.handoff_bytes > 3 * coupled.ledger.handoff_bytes
    assert disagg.ledger.mean_tbt < coupled.ledger.mean_tbt
    assert disagg.ledger.mean_latency > coupled.ledger.mean_latency


# 13c. ...and the two halves of that row were written by two hosts that never
#      spoke. The prefill host recorded the routing decision and the publish; the
#      decode host recorded the handoff and the inter-token gaps. Neither was
#      handed the other's row -- the ledger is the collector, and the join is by
#      request id. A single row carrying both halves is what proves the join ran.
#      Read off the disaggregated run, where the pools are disjoint so every
#      request really was written by two different hosts.
def test_each_host_records_its_own_half_of_a_request():
    disagg, _coupled = run_disaggregation()
    joined = [
        row for row in disagg.ledger.accepted
        if row.prefill and row.decode and row.prefill != row.decode
    ]
    assert len(joined) == len(disagg.ledger.accepted)
    for row in joined:
        assert row.ttft > 0          # the prefill host's half
        assert row.handoff_bytes > 0  # the decode host's half
        assert row.tbt > 0            # ...and the decode host's again, later


# 13d. Nothing on a serving host can reach another serving host. The redirect
#      model's whole structural claim, so it is asserted and not just described:
#      a host answers with an *address*, and the client -- which is run wiring, not
#      capability code -- is what walks it. A ``peers`` argument coming back would
#      fail the first half; a host stashing a peer's bound method would fail the
#      second.
def test_a_serving_host_cannot_reach_another_serving_host():
    import inspect

    from kvcache_sim.data.serving import ServingHost

    accepted = set(inspect.signature(ServingHost.__init__).parameters) - {"self"}
    assert accepted == {
        "me", "store", "coordinator", "trace", "metrics", "prefill", "decode",
        "models_decode",
    }

    sim = Simulation(_make_topology(2))
    store = KVStore(sim.mesh)
    hosts = {
        i: ServingHost(
            i, store, sim.coordinator_handle, trace=sim.trace, metrics=sim.ledger
        )
        for i in sim.ids
    }
    try:
        for host in hosts.values():
            for name, held in vars(host).items():
                assert not isinstance(held, ServingHost), name
                # A bound method of another host is the same reference wearing a
                # callable's clothes, which is how the decode handoff used to work.
                owner = getattr(held, "__self__", None)
                assert owner is None or owner is host or not isinstance(
                    owner, ServingHost
                ), name
    finally:
        sim.loop.close()


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


# 16b. The client seam: a request is redirected, so the round trips it pays are
#      client<->host ones and there are three of them (route, prefill, decode).
#      There is no host-to-host hop left to charge, which is why ``host_rtt``
#      became ``client_rtt`` rather than being deleted.
def test_client_rtt_defaults_to_free_and_byte_identical():
    baseline = run_shared_prefix(seed=1)[0]
    with config.overrides(client_rtt=0.0):
        explicit = run_shared_prefix(seed=1)[0]
    assert explicit.trace.render() == baseline.trace.render()
    assert explicit.ledger.mean_ttft == baseline.ledger.mean_ttft


def test_a_distant_client_delays_the_request_and_costs_reuse():
    """Three round trips per request, paid before each leg can start.

    The same two effects the coordinator hop has, for the same reason: the delay
    pushes a request further down the prefill queue, and routing runs against a
    directory snapshot taken later, by which time the queue it is predicting
    against has moved. It does *not* land in TTFT by one RTT -- TTFT here is
    control's prediction, made before any hop is paid -- so what moves is the
    queueing that prediction is built on.
    """
    free = run_shared_prefix(seed=1)[0]
    with config.overrides(client_rtt=0.5):
        distant = run_shared_prefix(seed=1)[0]
    assert distant.ledger.mean_ttft > free.ledger.mean_ttft
    assert distant.ledger.hit_rate < free.ledger.hit_rate


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
            store = KVStore(sim.mesh)
            await store.publish("s1", list(keys), _kv(len(keys)))
            ranked = await LongestPrefixPolicy().select(sim.view, keys, "s0")
        return empty, ranked

    try:
        empty, ranked = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert empty.sources == ()                  # nobody holds it yet
    assert ranked.sources == ("s1",)            # ...and now the holder is ranked


# --------------------------------------------------------------------------
# The coordinator surface: two members, this application's questions as values.
# --------------------------------------------------------------------------


def _scheduler(n: int = 2):
    """A scheduler attached to a real mesh view, ready to be asked things."""
    sim = Simulation(_make_topology(n))
    sched = LoadBalanceScheduler(block_tokens=512)
    sched.attach(sim.view, sim.transfer_cost)
    return sim, sched


def test_decide_dispatches_each_demand_to_its_own_handler():
    """One member, three questions, told apart by the demand's type."""
    sim, sched = _scheduler()
    request = Request(
        id="r0", arrival=0.0, prompt_tokens=1024, output_tokens=1,
        block_keys=tuple(_block_keys_for("m0", [0, 1])),
    )

    async def scenario():
        with sim.mesh.installed():
            plan = await sched.decide(Route(request))
            admitted = await sched.decide(AdmitDecode(plan))
            tail = await sched.decide(PrefillFinished(plan.prefill, 42.0))
        return plan, admitted, tail

    try:
        plan, admitted, tail = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert plan is not None and plan.prefill in sched.ids
    assert admitted is True                       # no TBT SLO in this run
    assert tail == 42.0                            # the corrected queue tail, echoed


def test_observe_dispatches_each_fact_and_answers_nothing():
    """The learning half: it corrects the model and returns None."""
    sim, sched = _scheduler()
    try:
        assert sched.observe(ComputeBusy("s0", 7.0)) is None
        assert sched.busy_until["s0"] == 7.0
        assert sched.observe(DecodeState("s1", (1.0, 2.0))) is None
        assert sched._occupancy("s1") == 2
    finally:
        sim.loop.close()


def test_a_subclass_answer_is_the_answer_that_runs():
    """The dispatch table binds per instance, so overriding an answer suffices.

    ``functools.singledispatchmethod`` would fail this silently: it captures the
    function registered on the base class, so a subclass redefining the handler is
    ignored and the base answer runs instead. Both schedulers override exactly this
    way, so the trap would be invisible in a passing suite.
    """
    sim = Simulation(_make_topology(2))

    class Fixed(LoadBalanceScheduler):
        async def _decide_route(self, demand):
            return "overridden"

    sched = Fixed(block_tokens=512)
    sched.attach(sim.view, sim.transfer_cost)
    try:
        answer = sim.loop.run_until_complete(sched.decide(Route(request=None)))
    finally:
        sim.loop.close()
    assert answer == "overridden"


@pytest.mark.parametrize("member,payload", [("decide", "Route"), ("observe", 3)])
def test_an_unknown_payload_is_refused_not_guessed(member, payload):
    """A demand this application does not define must fail loudly at the surface."""
    sim, sched = _scheduler()
    try:
        with pytest.raises(TypeError, match="does not answer|is not told"):
            result = getattr(sched, member)(payload)
            if member == "decide":
                sim.loop.run_until_complete(result)
    finally:
        sim.loop.close()


