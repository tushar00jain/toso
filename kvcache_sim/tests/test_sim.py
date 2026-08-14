"""Deterministic tests for the KV-cache simulation on the *real* TorchStore.

Run from the parent directory: ``python -m pytest kvcache_sim/tests -q``.
Assertions are on the simulation *outcome* (block presence in the real directory,
prefix hit rate, prefill compute, eviction, rejections, TBT, determinism) -- never
on wall-clock timing. Every scenario drives the real ``Controller`` directory + real
per-instance clients on the shared deterministic async engine.
"""

from __future__ import annotations

import argparse
import asyncio

import pytest

from sim_common import config
from sim_common.async_engine import AsyncEngine, run_sim

from kvcache_sim.control._view import KVView, _longest_prefix_run
from kvcache_sim.workload._accelerator import (
    BLOCK_TOKENS, SimulatedAccelerator, TOKEN_DTYPE, token_tensor,
)
from realsim.seams.cluster_model_handle import LocalClusterModelHandle
from realsim.seams.cluster_model_service import ClusterModelService
from realsim.simulation import Simulation
from sim_common.cost_model import DEFAULT_PROFILE
from domain import decode_step_time
from kvcache_sim.data._decode import DecodeEngine
from kvcache_sim.data._store import KVStore
from kvcache_sim.control.request import Request
from proposed import ControlPlane, KeySelector
from proposed.selector import KeySelectorChain
from kvcache_sim.control._source import LongestPrefixKeySelector, SpreadReadsKeySelector
from kvcache_sim.control.scheduler import (
    ComputeBusy, DecodeState, LoadBalanceScheduler, PrefillFinished, _LocalOnly,
)
from kvcache_sim.workload._serving import scheduler
from kvcache_sim.workload._generator import _block_keys_for, make_workload
from kvcache_sim.tests._run import (
    run,
    run_disaggregation,
    run_early_rejection,
    run_eviction_sweep,
    run_hotspot,
    run_overload,
    run_shared_prefix,
)
from kvcache_sim.__main__ import KVCacheDemo
from kvcache_sim.workload import scenarios
from kvcache_sim.workload.scenarios import (
    DISAGG_TARGET_TBT,
    _make_topology,
    _shared_prefix_workload,
)


#: The run's own block geometry, for tests that have to count blocks the way the
#: accelerator does (a partial trailing block is a block).
_GEOMETRY = SimulatedAccelerator(block_tokens=BLOCK_TOKENS)


def _kv(count: int, block_tokens: int = BLOCK_TOKENS):
    """``count`` KV blocks, exactly as a prefill on this run's accelerator makes them.

    A publish takes the blocks now (the store holds no notion of what one is), so a
    test that wants keys present has to produce KV for them. It does that through
    the same object the serving plane does rather than conjuring a tensor of its
    own, so a test can never publish a block of a size the run would not.
    """
    return SimulatedAccelerator(block_tokens=block_tokens).kv_blocks(count)


def _turns(conversations) -> list:
    """Every turn of every conversation, flattened, in dialogue order.

    :func:`~kvcache_sim.workload._generator.make_workload` answers with
    conversations now, because a request is a *turn* of one and turn N+1 cannot be
    submitted until turn N has answered. Most assertions below are about requests,
    so this is the same flattening
    :attr:`~kvcache_sim.workload._serving.KVWorkload.requests` performs -- written
    once here rather than as a comprehension at each call site.
    """
    return [turn.request for c in conversations for turn in c.turns]


def _request(**kwargs) -> Request:
    """A ``Request`` whose prompt tensor matches whatever ``prompt_tokens`` says.

    A request carries the prompt itself now, and refuses to be built with one whose
    length disagrees with the count the scheduler prices against. Every test below
    cares about the count and none of them about the ids (there are none -- it is a
    meta tensor), so the prompt is derived here rather than spelled out at a dozen
    call sites.
    """
    return Request(prompt=token_tensor(kwargs["prompt_tokens"]), **kwargs)


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


# 1b. A request carries the prompt, and the two descriptions of it must agree.
#     ``prompt_tokens`` is what the scheduler prices (the prefix match, the
#     uncached suffix, the predicted TTFT) and the tensor is what the forward pass
#     runs over. If they may differ, the run stays internally consistent and
#     measures a prompt nobody submitted, which is the failure mode worth a
#     constructor check.
def test_a_request_refuses_a_prompt_that_is_not_the_length_it_claims():
    with pytest.raises(ValueError, match="prompt of 256"):
        Request(
            id="r0", arrival=0.0, block_keys=_block_keys_for("m0", [0]),
            prompt_tokens=BLOCK_TOKENS, output_tokens=1,
            prompt=token_tensor(256),
        )


# 1c. ...and the block keys are still generated beside the prompt rather than
#     hashed out of it, because a meta tensor has nothing to hash. The compromise
#     is asserted so it cannot be quietly forgotten: two requests with identical
#     prompt *shapes* and different content share no keys, which is exactly what a
#     shape-derived "hash" would get wrong.
def test_block_keys_are_generated_not_derived_from_the_prompt():
    # ``max_turns=1`` makes every draw its own one-turn dialogue, which is the
    # single-shot shape this claim is about: two prompts of identical length and
    # different content. A second turn would be longer than the first, so the
    # shapes would differ for a reason that has nothing to do with hashing.
    a, b = _turns(make_workload(
        num_requests=2, num_conversations=2, system_blocks=1, conv_base_blocks=1,
        query_blocks=1, block_tokens=BLOCK_TOKENS, max_turns=1, seed=7,
    ))
    assert a.prompt.shape == b.prompt.shape       # identical shapes...
    assert a.block_keys[-1] != b.block_keys[-1]   # ...and distinct queries
    # The prompt is real, exactly sized and free to hold -- and has no ids in it,
    # which is why the keys cannot come from it.
    assert a.prompt.device.type == "meta"
    assert a.prompt.numel() == a.prompt_tokens


# 1c-ii. A conversation is turns, and turn N+1 *is* turn N plus what came back
#        plus what the user said next. This is the whole multi-turn claim, checked
#        on the stream rather than through a run: the reusable prefix grows
#        monotonically and the thing in the middle of the growth is the previous
#        turn's own generated-KV keys, taken from the previous turn's own
#        ``continuation_keys`` rather than spelled out again here.
def test_a_turn_is_the_previous_turn_plus_its_output_plus_a_new_message():
    conversations = make_workload(
        num_requests=40, num_conversations=3, system_blocks=2, conv_base_blocks=3,
        query_blocks=2, block_tokens=BLOCK_TOKENS, output_tokens=64, max_turns=6,
        seed=5,
    )
    assert sum(len(c.turns) for c in conversations) == 40   # every turn accounted
    assert max(len(c.turns) for c in conversations) <= 6    # ...and bounded
    multi = [c for c in conversations if len(c.turns) > 1]
    assert multi, "no conversation has a second turn, so nothing here is tested"

    generated = _GEOMETRY.blocks_for(64 - 1)
    for conversation in conversations:
        first = conversation.turns[0].request
        assert len(first.block_keys) == 2 + 3 + 2          # system + base + query
        assert conversation.turns[0].think == 0.0          # its arrival is the item's
        for before, after in zip(conversation.turns, conversation.turns[1:]):
            previous, current = before.request, after.request
            history = previous.block_keys + previous.continuation_keys(generated)
            # The growing prefix, exactly: everything the previous turn was, then
            # everything it produced, then the two blocks of the new message.
            assert current.block_keys[:len(history)] == history
            assert len(current.block_keys) == len(history) + 2
            assert current.prompt_tokens == len(current.block_keys) * BLOCK_TOKENS
            assert current.prompt.numel() == current.prompt_tokens
            assert after.think > 0.0                       # a user paused to read
        # ...and all of them belong to the same conversation as far as a front end
        # that routes on a session id is concerned.
        assert len({t.request.conversation for t in conversation.turns}) == 1


def test_the_conversation_stream_is_a_property_of_the_seed_alone():
    """Same seed, same stream -- and it does not depend on how a run goes.

    The arrival of turn N+1 is emergent (it waits for turn N's answer), so the one
    thing that has to stay fixed is *which* turns exist, what they contain and how
    long each user pauses. That is what makes "same workload, different wiring"
    still a fair comparison between a scenario's configurations, and it is checked
    on the generated stream because the stream is where it is decided.
    """
    kwargs = dict(
        num_requests=60, num_conversations=4, block_tokens=BLOCK_TOKENS, seed=11
    )
    assert make_workload(**kwargs) == make_workload(**kwargs)
    assert make_workload(**{**kwargs, "seed": 12}) != make_workload(**kwargs)


# 1d. The chain continues past the prompt, because the sequence does. Generated
#     tokens extend the sequence, so the KV a decode host produces needs keys, and
#     they are built by extending the prompt's last key rather than hashed for the
#     reason above: there is nothing in a meta token to hash.
def test_continuation_keys_extend_the_chain_and_cannot_collide_with_a_prompt():
    request = _request(
        id="r0", arrival=0.0, block_keys=_block_keys_for("m0", [0, 1]),
        prompt_tokens=2 * BLOCK_TOKENS, output_tokens=64,
    )
    keys = request.continuation_keys(3)
    assert len(keys) == 3
    # Each key contains the whole prefix before it -- the property that makes a
    # prefix-hash chain answer "how much of this sequence do you hold".
    assert keys[0].startswith(request.block_keys[-1])
    assert keys[1].startswith(keys[0]) and keys[2].startswith(keys[1])
    assert len(set(keys)) == 3
    # And no prompt chain can ever name one: the generator's segments are decimal
    # integers, so a ``|g<i>`` segment is a namespace of its own.
    for key in keys:
        assert key.rsplit("|", 1)[1].startswith("g")
    assert request.continuation_keys(0) == ()


# 1e. A generation leaves KV behind it, in whole blocks, and the trailing partial
#     block is charged whole. A paged cache hands out a physical block for the
#     position that first lands in it; the unused remainder is fragmentation the
#     host is really paying for. Dropping it would make decode residency vanish at
#     this model's block size, since nothing here generates 512 tokens.
def test_a_generation_leaves_whole_blocks_of_kv_behind_it():
    acc = SimulatedAccelerator(block_tokens=BLOCK_TOKENS)
    assert acc.generated_kv(0) == []                       # no step, no KV
    assert len(acc.generated_kv(1)) == 1                   # ...one position is a block
    assert len(acc.generated_kv(BLOCK_TOKENS)) == 1
    assert len(acc.generated_kv(BLOCK_TOKENS + 1)) == 2    # one past the boundary
    block, = acc.generated_kv(31)
    # The same block a prompt's KV is made of: same size, same dtype, same cache.
    assert block.numel() * block.element_size() == acc.block_nbytes
    assert block.device.type == "meta"                     # and still free to hold


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
    store = KVStore(sim)
    view = sim.view.derived(KVView)

    async def scenario():
        # The data plane publishes/evicts; the control-plane view reads back.
        with sim.mesh.installed():
            await store.publish("s0", list(keys[:3]), _kv(3))  # s0: 3 leading blocks
            await store.publish("s1", list(keys[:1]), _kv(1))  # s1 holds 1
            counts = view.prefix_lengths(list(keys))
            assert counts == {"s0": 3, "s1": 1}
            await _evict(sim.mesh, "s0", [keys[1]])    # break s0's run at index 1
            counts2 = view.prefix_lengths(list(keys))
            assert counts2 == {"s0": 1, "s1": 1}
        return True

    try:
        ok = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert ok


# 2a. One decision, one directory. A routing decision reads the prefix runs once
#     for its own candidate loop and again from every selector it consults, and
#     all of those must see the same directory or the prices are not comparable.
#     The snapshot is scoped state on the view because that is what those selectors
#     sense through, so it has to be visible for the decision and gone after it.
def test_a_pinned_view_serves_one_snapshot_and_releases_it():
    topo = _make_topology(2)
    keys = _block_keys_for("m0", [0, 1, 2, 3])

    sim = Simulation(topo)
    store = KVStore(sim)
    view = sim.view.derived(KVView)

    async def scenario():
        with sim.mesh.installed():
            await store.publish("s0", list(keys), _kv(len(keys)))
            with view.pinned(list(keys)):
                pinned = view.prefix_lengths(list(keys))
                await _evict(sim.mesh, "s0", [keys[1]])
                held = view.prefix_lengths(list(keys))
            return pinned, held, view.prefix_lengths(list(keys))

    try:
        pinned, held, after = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert pinned == held == {"s0": 4}   # the snapshot, not the eviction under it
    assert after == {"s0": 1}            # ...and the live directory once released


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
    store = KVStore(sim)

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
    convs = _shared_prefix_workload()
    from dataclasses import replace as _replace

    from domain import DEFAULT_MODEL, DEFAULT_PROFILE

    from kvcache_sim.workload._serving import BLOCK_TOKENS

    unbounded = run(topo, convs, "cache_aware").ledger.hit_rate
    big = run(
        topo, convs, "cache_aware",
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

    topo, convs = _make_topology(4), _shared_prefix_workload()
    one_block = DEFAULT_MODEL.block_bytes(1, BLOCK_TOKENS)

    def _at(capacity_bytes: int):
        return run(
            topo, convs, "cache_aware",
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


# 7c. A request is served where control says, not where it landed.
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

    topo, convs = _make_topology(4), _shared_prefix_workload()
    result = run(topo, convs, "cache_aware")

    landed = _affinity(sorted(topo))
    reqs = _turns(convs)
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


# 8. Hotspot: replication trades recompute for KV transfer. It used to also lower
#    TTFT, and on a multi-turn workload it no longer does -- see the docstring.
def test_hotspot_replication_trades_recompute_for_transfer():
    """What replication buys, and what it stopped buying when prefixes started growing.

    Two of the three claims are unchanged and are the mechanism: ``replicate=False``
    never pulls, replication does, and the run that pulls recomputes strictly fewer
    prompt tokens. That is the trade the scenario exists to show and it is robust
    across seeds.

    The third -- that replication also *lowers* TTFT by spreading a hot prefix over
    peers -- is gone, and the assertion for it is not weakened here but replaced,
    because the phenomenon it measured is gone rather than smaller. The old
    workload's "one dominant conversation" was one fixed prefix that every one of
    that conversation's requests shared, so the cache-aware selector really did pile
    all of them on the single instance holding it and replication really did spread
    them. A multi-turn dominant *tenant* is many dialogues, each with its own
    growing history, and the cache-aware selector already scatters them across
    instances by following those histories -- there is no pile left to spread. What
    is shared between them is only the opening, which is a shrinking fraction of a
    conversation that grows, and pulling it now means pulling a long chain rather
    than a fixed dozen blocks. So replication's TTFT effect is noise: it wins on
    some seeds and loses on others, and this asserts what is left, which is that
    both cache-aware configurations beat the load-balancing baseline by a wide
    margin. See the report in the scenario's ``show()`` for the claim that had to
    be withdrawn.
    """
    baseline, no_repl, repl = run_hotspot()
    # Routing to the history still dominates, whether or not it is replicated.
    assert repl.ledger.mean_ttft < baseline.ledger.mean_ttft
    assert no_repl.ledger.mean_ttft < baseline.ledger.mean_ttft
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
                _request(id=f"r{i}", arrival=0.0, block_keys=("m0|0",),
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
    # Both serve the identical load: the TBT SLO is infinite, so nothing is shed.
    for r in (disagg, coupled):
        assert len(r.ledger.accepted) == len(r.ledger.results)
    # A dedicated decode pool holds the TBT target for (nearly) every served
    # request; coupling prefill into decode makes a real fraction miss it.
    assert disagg.ledger.tbt_slo_met(DISAGG_TARGET_TBT) >= 0.95
    assert coupled.ledger.tbt_slo_met(DISAGG_TARGET_TBT) < 0.9
    assert (disagg.ledger.tbt_slo_met(DISAGG_TARGET_TBT)
            > coupled.ledger.tbt_slo_met(DISAGG_TARGET_TBT))


# 14. Both admission modes gate before the prefill; what separates them is the
#     occupancy the gate is fed, and here that shows in routing rather than in TBT.
def test_the_two_admission_modes_gate_before_prefill_and_route_differently():
    """Gating at routing never spends compute on a refusal; foreseeing the load
    changes where decode goes.

    The first claim is structural and is what admission is *for*: both modes ask
    before the prefill runs, so every request that reaches a decode batch was
    admitted while a refusal would still have been free, and both serve the whole
    offered load at this SLO.

    The second is what the lookahead buys, and it is not the TBT column. 'predict'
    routes decode by the load foreseen at prefill completion and 'early' by a
    current occupancy a slow prefill leaves reading empty, so the two put requests
    on different hosts -- visible in the KV each run's decode side ends up holding.
    Attainment is deliberately not asserted a direction: a conversation-per-item
    workload keeps at most one request per open dialogue in flight, so occupancy is
    never far from what a stale snapshot reports and across four seeds the gap is
    noise whose sign changes. Asserting one would be asserting a seed.
    """
    early, predict = run_early_rejection()
    # Nothing is refused after its prefill, because nothing is asked after it: both
    # runs serve every request they were offered.
    for result in (early, predict):
        assert len(result.ledger.accepted) == len(result.ledger.results)
    # The lookahead is not cosmetic: it lands decode elsewhere, and the decode-side
    # residency is where that shows.
    assert early.ledger.decode_blocks != predict.ledger.decode_blocks


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
def _answer(request: Request, *, prefill: str, decode: str):
    """A minimal accepted decision. Exercises the decode leg, not the router."""
    from kvcache_sim.control.scheduler import Plan, Response

    return Response(
        prefill=prefill,
        decode=decode,
        plan=Plan(
            match_blocks=0, cached_tokens=0,
            uncached_tokens=request.prompt_tokens, reuse_source=None,
            transfer_bytes=0, queue_wait=0.0, ttft=0.0, done_time=0.0,
        ),
    )


def _decode_leg(*, output_tokens: int = 6, prefill: str = "s0",
                published: bool = True, engine: bool = True,
                capacity_bytes: float = float("inf"), probe: dict = None):
    """Walk one request's decode leg on ``s1``; report when it answered.

    Returns ``(answered_at, events, row, tokens)`` -- the sim clock when
    :meth:`ServingHost.decode` returned, the host's own trace lines as
    ``{kind: [time]}`` (it writes a HANDOFF when the KV lands, a RESIDE when a set
    of blocks is registered on this host, and a DECODE when the last token does),
    the ledger row the two halves were joined into, and the tokens the leg answered
    with.

    ``probe``, when given, is filled in **inside** the running scenario with what
    the real directory says once the leg is done: ``chain`` and ``generated`` map
    instance -> how long a run of the prompt's keys / this request's continuation
    keys that instance holds. It is a mutable out-parameter rather than a fifth
    return value only because the directory has to be read before the loop closes
    and five of these callers do not care.

    A real control plane is wired in because a decode batch *reports itself*: the
    host forwards every batch change to control, so a stub ``None`` would fail on
    the first admission rather than on anything this is testing.
    """
    from dataclasses import replace as _replace

    from kvcache_sim.data.serving import ServingHost
    from kvcache_sim.report.metrics import Metrics, RequestResult

    sim = Simulation(
        _make_topology(2),
        control=LoadBalanceScheduler(
            block_tokens=BLOCK_TOKENS, simulate_decode=True
        ),
        profile=_replace(
            DEFAULT_PROFILE, storage_capacity_bytes=capacity_bytes
        ),
        ledger=Metrics(),
    )
    store = KVStore(sim)
    keys = list(_block_keys_for("m0", [0, 1]))
    request = _request(
        id="r0", arrival=0.0, block_keys=tuple(keys),
        prompt_tokens=2 * BLOCK_TOKENS, output_tokens=output_tokens,
    )
    host = ServingHost(
        "s1",
        trace=sim.trace, metrics=sim.ledger,
        decode=(
            DecodeEngine(SimulatedAccelerator(), max_batch=8) if engine else None
        ),
        models_decode=engine,
    )
    host.attach(sim)

    async def scenario():
        with sim.mesh.installed():
            if published:
                await store.publish(prefill, keys, _kv(len(keys)))
            # The prefill host opens the row; here there is no prefill host, so the
            # test stands in for it -- the decode side amends, it never creates.
            sim.ledger.add(RequestResult(id=request.id, accepted=True))
            tokens = await host.decode(
                request, _answer(request, prefill=prefill, decode="s1")
            )
            if probe is not None:
                view = sim.view.derived(KVView)
                probe["chain"] = view.prefix_lengths(keys)
                probe["generated"] = view.prefix_lengths(
                    list(request.continuation_keys(1))
                )
            return asyncio.get_running_loop().time(), tokens

    try:
        answered_at, tokens = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    events: dict = {}
    for t, kind, _msg in sim.trace.events:
        events.setdefault(kind, []).append(t)
    return answered_at, events, sim.ledger.results[0], tokens


def test_the_decode_leg_answers_when_the_last_token_lands():
    """The ordinary path: KV fetched, made resident, stepped, and published.

    This used to assert ``answered_at == events["DECODE"][0]`` -- the leg returned
    on the very instant of the last token -- and that equality encoded the absence
    of decode-side residency rather than a property worth keeping. Nothing happened
    after the last token because the KV the batch had just produced went nowhere:
    it was not written, not registered, and cost the decode host nothing. Now it is
    published, so the leg answers *after* the last token by exactly that write, and
    the claim the test is really about survives intact -- the leg does not answer at
    admission, and every decode step is inside the interval.
    """
    answered_at, events, row, _tokens = _decode_leg(output_tokens=6)
    assert len(events["DECODE"]) == 1
    # Two residency publishes: the chain this host pulled in, then the KV its
    # generation appended. The leg answers when the second one lands.
    assert len(events["RESIDE"]) == 2
    assert events["HANDOFF"][0] <= events["RESIDE"][0] < events["DECODE"][0]
    assert answered_at == events["RESIDE"][1]
    # ...and the five steps really ran between the chain landing and the last
    # token (6 output tokens, the first of which prefill produced).
    steps = 5 * decode_step_time(1, DEFAULT_PROFILE)
    assert events["DECODE"][0] - events["RESIDE"][0] == pytest.approx(steps)
    assert row.tbt > 0                       # ...and the row was written first
    assert row.handoff_bytes > 0


def test_the_decode_leg_answers_with_the_tokens_it_generated():
    """Decode's half of the output: ``output_tokens - 1``, one per step it ran.

    The division of labour the engine's docstring has always described, now
    checked against what comes back rather than against a counter: the first token
    is the prefill's (it is what TTFT is the time to) and the remainder is this
    leg's. A request that needs no decode step gets an empty list, which is the
    same statement -- prefill produced the whole output -- rather than a special
    case in the caller.
    """
    for asked, expected in ((6, 5), (2, 1), (1, 0)):
        _at, _events, _row, tokens = _decode_leg(output_tokens=asked)
        assert len(tokens) == expected, asked
        # Real tensors, and free ones: a token id of the run's dtype with no
        # storage behind it, exactly like the KV blocks it was decoded from.
        for token in tokens:
            assert token.device.type == "meta"
            assert token.dtype is TOKEN_DTYPE
            assert token.numel() == 1
        # Distinct objects, one per step -- a batch that handed the same tensor to
        # every member would be indistinguishable from this if they aliased.
        assert len({id(t) for t in tokens}) == len(tokens)


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
    """Every path answers, and never before the last token.

    The equality this used to assert (answered exactly *at* the last token) was
    true only while the KV a generation produced went nowhere. It is now published
    on the way out, so the leg answers at or after the last token -- and where a
    publish happened, at exactly the instant it landed. What the parametrisation is
    guarding is unchanged: none of these three paths is one where the caller is
    left parked forever.
    """
    answered_at, events, _row, _tokens = _decode_leg(**kwargs)
    assert len(events["DECODE"]) == 1, case
    assert answered_at >= events["DECODE"][0], case
    if "RESIDE" in events:
        assert answered_at == events["RESIDE"][-1], case


# 13f. Decoding costs the decode host memory, in both of the ways it can.
#      Two things land on a decode host and neither used to be accounted for: the
#      block chain it pulls in to attend over, and the KV its own generation
#      appends. Both are now published on it, so its volume charges itself for
#      them and the directory knows a second copy is there. Without this a decode
#      host had unbounded free memory and every capacity number in a
#      decode-simulating run was flattered by exactly the residency that never
#      happened.
def test_a_decode_host_is_resident_for_the_chain_it_pulled_and_the_kv_it_made():
    probe: dict = {}
    _at, events, row, _tokens = _decode_leg(output_tokens=6, probe=probe)
    # The prefill host published the 2-block chain; the decode host pulled it in
    # and is now a replica of it -- which is what a read-through cache is.
    assert probe["chain"] == {"s0": 2, "s1": 2}
    # ...and the 5 generated positions are one block, on the decode host alone.
    # Nobody else could hold it: no other host ran a step of this generation.
    assert probe["generated"] == {"s1": 1}
    assert row.decode_blocks == 3          # 2 pulled + 1 generated
    assert row.decode_unpublished is False
    assert len(events["RESIDE"]) == 2      # one per set, in the order they landed


def test_a_host_that_decodes_where_it_prefilled_is_resident_only_for_what_it_made():
    """The local case adds the generation and nothing else, which is the truth.

    Its prompt's blocks were registered here by the prefill and never left, so
    counting them again as newly resident would double-charge one copy. What is
    new is only what the batch generated.
    """
    probe: dict = {}
    _at, _events, row, _tokens = _decode_leg(prefill="s1", probe=probe)
    assert probe["chain"] == {"s1": 2}
    assert probe["generated"] == {"s1": 1}
    assert row.decode_blocks == 1


def test_a_request_that_generates_nothing_leaves_no_kv_behind():
    """``output_tokens=1``: prefill produced the whole answer, decode ran no step.

    No step is no position of KV, so there is no block, no key and no publish --
    the empty case is empty rather than a zero-length block registered under a
    continuation key nothing continues.
    """
    probe: dict = {}
    _at, events, row, _tokens = _decode_leg(output_tokens=1, probe=probe)
    assert probe["generated"] == {}
    assert row.decode_blocks == 2          # the pulled chain, and only that
    assert len(events["RESIDE"]) == 1


def test_a_decode_host_with_no_room_says_so_and_still_answers():
    """A decode-side cache fill may fail, exactly as a prefill-side one may.

    A capacity below a single block is the only place a refusal reaches this code,
    for the reason the prefill-side twin of this test spells out: the transport
    writes one key per put, so any larger volume absorbs an over-sized publish by
    evicting its own earlier blocks. Below one block nothing can be kept at all --
    not the chain the prefill host tried to publish (hence the miss) and not the
    block this generation produced.

    The request still finishes, and that is the point. Publishing at the end of the
    generation is what makes "the decode host had no room" a cache-fill failure
    rather than a request that has to be preempted halfway through an answer: by
    the time it is attempted the answer exists. There is no mid-generation
    preemption modelled here and deliberately none invented.
    """
    from domain import DEFAULT_MODEL

    _at, events, row, tokens = _decode_leg(
        output_tokens=6,
        capacity_bytes=DEFAULT_MODEL.block_bytes(1, BLOCK_TOKENS) - 1,
    )
    assert row.decode_unpublished is True
    assert "NOROOM" in events
    assert len(tokens) == 5                # served in full regardless
    # Nothing was kept, so nothing is counted as resident: the refusal is the
    # separate fact, not a zero hiding inside the block column.
    assert row.decode_blocks == 0
    assert row.handoff_missed is True      # the chain could not be cached either


def test_decode_residency_is_zero_where_decode_is_not_modelled():
    """The other half of the claim: this costs nothing to a prefill-only run.

    A run with ``simulate_decode=False`` never reaches a decode host, so no chain
    is pulled and no KV is generated, and its volumes hold exactly what they held
    before. That is what keeps the four prefill-side scenarios byte-identical.
    """
    aware, _baseline = run_shared_prefix()
    assert aware.ledger.decode_blocks == 0
    assert aware.ledger.decode_unpublished == 0


def test_a_run_that_models_decode_pays_for_it_in_blocks():
    """And the amount is the deployment's *and the conversation's*, not a constant.

    Disaggregation decodes every request away from where it was prefilled, so every
    request drags its whole chain across and then generates on top of it. What
    changed is that "its whole chain" is no longer one number: turn 1 of a dialogue
    is 5 blocks and turn 8 is 19, because turn N+1 carries turn N's prompt, turn N's
    output and a new message. So the expected total is summed off the workload
    rather than written as ``120 x (5 + 1)`` -- and it has to be, since a constant
    here would have to be re-derived by hand every time a conversation's shape
    changes, which is exactly how a test stops describing anything.

    Coupling still decodes most requests where the KV already is, so most of them
    pay only for what they generated.
    """
    disagg, coupled = run_disaggregation()
    generated = _GEOMETRY.blocks_for(12 - 1)     # output_tokens=12 in that scenario
    expected = sum(
        len(r.block_keys) + generated for r in disagg.workload.requests
    )
    assert disagg.ledger.decode_blocks == expected
    assert 0 < coupled.ledger.decode_blocks < disagg.ledger.decode_blocks


def test_a_host_with_no_decode_engine_answers_without_waiting():
    """The one early answer left, and it is not a decode that was skipped.

    A run that does not model decode never sends a client here at all (prefill
    answers ``None`` and the journey ends), so this is the defensive case: a host
    asked to decode with no engine has no last token coming, and waiting for one
    would be waiting forever. It answers as soon as the KV is here and its volume
    has been told, which is as far as it can honestly get.

    That last clause is what changed: the host used to answer on the HANDOFF
    instant, because a fetched chain landed on it and left no trace on its volume.
    It pulled a whole block chain in and its own storage never heard of it. There
    is still no engine and still no last token, so the *shape* of the case is the
    same -- it answers early because there is nothing to wait for -- but "the KV is
    there" now includes saying so.
    """
    answered_at, events, row, tokens = _decode_leg(engine=False)
    assert "DECODE" not in events
    assert events["HANDOFF"][0] < answered_at == events["RESIDE"][0]
    assert row.tbt == 0.0
    # ...and it generated nothing, which is what it answers with. Not a token it
    # did not make and not ``None``: the leg's answer is its output, and this
    # host's output is empty.
    assert tokens == []


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

    # A request that was refused has no last token, so it gets no fabricated
    # duration -- and, more to the point, no waiter: the run still ends. Both
    # outcomes have to be in one run for the contrast to mean anything, so this is a
    # decode-modelling run under a TTFT SLO tight enough to shed about half of it.
    convs = make_workload(
        num_requests=40, num_conversations=10, system_blocks=1, conv_base_blocks=1,
        query_blocks=2, arrival_rate=20.0, block_tokens=BLOCK_TOKENS,
        output_tokens=4, seed=5,
    )
    mixed = run(
        _make_topology(2), convs, "cache_aware",
        simulate_decode=True, max_batch=4, slo_ttft=3.0,
    ).ledger
    refused = [r for r in mixed.results if not r.accepted]
    assert refused and mixed.accepted, "one outcome is missing, so this is vacuous"
    assert all(r.latency == 0.0 for r in refused)
    assert all(r.latency > 0.0 for r in mixed.accepted)


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


# 13h. The payoff the multi-turn workload exists to create: a prefix match that
#      lands on blocks a *generation* produced. Every one of these keys is a
#      ``|g<i>`` continuation key -- a key that can only exist because some turn
#      generated tokens after that exact prompt -- and until conversations grew,
#      the run published them and nothing ever asked for one.
def _generated_hits(result) -> tuple:
    """``(matched blocks, of which generated, matched past the last turn's prompt)``.

    Derived from the ledger and the workload rather than from a new column: what a
    row records is how many prompt *tokens* were served from cache, and blocks are
    that over the block size, so the matched run is the request's own leading keys
    and whether one of them is a generated key is a property of the key.

    The third number is the strict one. A generated key can be published by two
    different things -- the decode host that made the KV, or the *next* turn's
    prefill host, which recomputes it as part of the uncached suffix and publishes
    what it computed -- so a hit on any generated key does not by itself say
    decode's publish was ever read. A match that runs *past the previous turn's
    whole prompt chain* does: the block at that position is the previous turn's own
    output, and at the moment this turn is routed the only thing that can have
    published it is the host that generated it.
    """
    requests = {r.id: r for r in result.workload.requests}
    previous = {}
    for conversation in result.workload.conversations:
        for i, turn in enumerate(conversation.turns):
            previous[turn.request.id] = (
                len(conversation.turns[i - 1].request.block_keys) if i else 0
            )
    matched = generated = fresh = 0
    for row in result.ledger.accepted:
        keys = requests[row.id].block_keys[:row.cached_tokens // BLOCK_TOKENS]
        matched += len(keys)
        generated += sum(1 for k in keys if k.rsplit("|", 1)[1].startswith("g"))
        if previous[row.id] and len(keys) > previous[row.id]:
            fresh += 1
    return matched, generated, fresh


def test_a_later_turn_reuses_the_kv_an_earlier_turn_generated():
    """Generated KV is read, not merely written -- in every scenario.

    The claim this replaces was a caveat in
    :meth:`kvcache_sim.control.request.Request.continuation_keys`: the entries a
    decode host published were findable and nothing looked for them, so publishing
    cost capacity and bought no hit rate. A conversation's turn N+1 walks turn N's
    generated keys on its way to its own new message, so those entries are now
    matched -- ~15% of every matched block in the prefill-side scenarios and ~30%
    in the decode-side ones, where the run also gets a *second* turn's worth of
    growth out of them.
    """
    for result in (*run_shared_prefix(), *run_disaggregation()):
        matched, generated, _fresh = _generated_hits(result)
        assert matched > 0
        assert generated > 0, f"{result.label}: no matched block was generated KV"
        # Not a rounding error's worth: a meaningful share of the reuse.
        assert generated > 0.05 * matched, result.label


def test_only_a_decode_hosts_publish_can_serve_the_previous_turns_output():
    """...and in a run that models decode, that publish is read.

    The strict version of the test above, and the one that closes the loop this
    task opened. A turn that matches *past* the previous turn's whole prompt is
    matching the previous turn's generated block, and at that instant the only
    host that has ever written that key is the host that decoded the previous turn
    -- the next prefill has not run yet. So a non-zero count here is a decode
    host's publish being read by a later request, which is the thing the model
    could not do at all before.

    Zero in a run that does not model decode, and that is the control rather than a
    gap: nothing generated, so nobody published that block, and the prefix run
    stops one block short of it every time. Turn N+1's prefill then computes it and
    publishes it, which is why the *loose* count above is non-zero even there.
    """
    for result in run_disaggregation():
        _matched, _generated, fresh = _generated_hits(result)
        assert fresh > 0, (
            f"{result.label}: no turn ever matched the previous turn's generated "
            f"block, so the decode side's publish is still write-only"
        )
    for result in run_shared_prefix():
        assert _generated_hits(result)[2] == 0


def test_a_conversations_turns_are_strictly_serial():
    """Turn N+1 is not routed until turn N's last token has landed.

    The structural half of multi-turn, and the reason one work item is a whole
    dialogue: turn N+1's prompt contains turn N's answer, so a run that overlapped
    them would be routing a request built out of tokens that did not exist yet.
    Read off the trace, because that is where both instants are: the DECODE line
    is written when a request's last token is emitted and the ROUTE line when the
    next one is placed.
    """
    result = run_disaggregation()[0]
    routed, decoded = {}, {}
    for t, kind, msg in result.trace.events:
        request_id = msg.split()[0]
        if kind == "ROUTE":
            routed.setdefault(request_id, t)
        elif kind == "DECODE":
            decoded[request_id] = t
    overlapped = 0
    for conversation in result.workload.conversations:
        for before, after in zip(conversation.turns, conversation.turns[1:]):
            done = decoded[before.request.id]
            started = routed[after.request.id]
            assert started >= done, (
                f"{conversation.id}: {after.request.id} was routed at {started}, "
                f"before {before.request.id} finished at {done}"
            )
            # ...and the user's pause is inside that gap, not skipped.
            assert started >= done + after.think - 1e-9
            overlapped += 1
    assert overlapped > 0, "no conversation has two turns, so nothing was checked"


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


# 13g. The output is counted off the tokens the run produced, by the only
#      participant that receives all of them. The first token comes back from the
#      prefill host and the rest from the decode host, so a request's length is a
#      client-side join in exactly the way its end-to-end latency is -- and it is
#      now a measurement rather than the ``output_tokens`` the workload asked for,
#      read back out of the request that asked.
@pytest.mark.parametrize("asked", [1, 5])
def test_the_client_counts_the_tokens_the_two_legs_produced(asked):
    convs = make_workload(
        num_requests=6, num_conversations=2, system_blocks=1, conv_base_blocks=1,
        query_blocks=1, block_tokens=BLOCK_TOKENS, output_tokens=asked, seed=3,
    )
    ledger = run(
        _make_topology(2), convs, "cache_aware",
        simulate_decode=True, max_batch=4,
    ).ledger
    assert ledger.accepted
    for row in ledger.accepted:
        assert row.output_tokens == asked, row.id
    # ``asked=1`` is the case that pins down which leg produced what: decode had
    # nothing to do, so the single token in the count can only have come from the
    # prefill host -- which is the one TTFT is the time to.


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
        "me", "trace", "metrics", "prefill", "decode", "models_decode",
    }
    # The ports are not among them: they arrive off the deployment at attach.
    assert set(inspect.signature(ServingHost.attach).parameters) - {"self"} == {
        "deployment",
    }

    sim = Simulation(_make_topology(2))
    hosts = {i: ServingHost(i, trace=sim.trace, metrics=sim.ledger) for i in sim.ids}
    for host in hosts.values():
        host.attach(sim)
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

    early1, predict1 = run_early_rejection(seed=3)
    early2, predict2 = run_early_rejection(seed=3)
    for a, b in ((early1, early2), (predict1, predict2)):
        assert a.trace.render() == b.trace.render()
        assert len(a.ledger.accepted) == len(b.ledger.accepted)
        assert a.ledger.mean_tbt == b.ledger.mean_tbt


# 16. The control seam: control is a service, so the hop is somewhere to
#     charge. At the default (0) it is inline and changes nothing; given a
#     duration it is paid out and back before prefill can start, so it lands in
#     TTFT -- the number this capability exists to move.
def test_control_rtt_defaults_to_free_and_byte_identical():
    baseline = run_shared_prefix(seed=1)[0]
    with config.overrides(control_rtt=0.0):
        explicit = run_shared_prefix(seed=1)[0]
    assert explicit.trace.render() == baseline.trace.render()
    assert explicit.ledger.mean_ttft == baseline.ledger.mean_ttft


def test_control_rtt_lands_in_the_wait():
    """A distant control plane costs the caller time: every turn pays the round trip.

    Measured end to end, on a decode-modelling run. TTFT will not show it, and not
    because the hop is free: TTFT here is control's *predicted* queue-plus-prefill,
    and a conversation is a closed loop, so delaying a turn delays its successor,
    offers the cluster less work per second and shortens its queues. Mean TTFT on
    that workload therefore *falls* monotonically as the hop grows (1.86 -> 1.56 ->
    1.42 at 0 / 0.5 / 2.0), which is the textbook closed-loop response.
    Arrival-to-last-token is the interval that actually contains the hop, and it is
    the one a caller experiences, so it is what is asserted.

    Reuse is left unasserted for the same closed-loop reason, and it is worth being
    explicit that the hop does not cost any: routing does read a directory snapshot
    one hop old, but the pacing hands that back with interest -- the next turn of a
    dialogue arrives later, by which time more of the cluster's publishes have
    landed -- and the hit rate drifts *up* with the hop here (0.843 -> 0.845 ->
    0.849 at 0 / 0.5 / 2.0). The two effects are of the same small size and the sign
    is the workload's, not the seam's.
    """
    free = run_disaggregation(seed=0)[0]
    with config.overrides(control_rtt=0.5):
        distant = run_disaggregation(seed=0)[0]
    assert distant.ledger.mean_latency > free.ledger.mean_latency
    # The comparison still holds -- both schedulers pay the same hop.
    with config.overrides(control_rtt=0.5):
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
    """Three round trips per turn, paid before each leg can start.

    The same two effects the control hop has and measured the same way, for
    the reason the test above spells out: TTFT is a prediction made before any hop
    is paid, and on a self-pacing workload it moves the wrong way anyway. What the
    hops are inside is the caller's own interval, and three of them per turn at
    0.5s put ~1.5s into it.
    """
    free = run_disaggregation(seed=0)[0]
    with config.overrides(client_rtt=0.5):
        distant = run_disaggregation(seed=0)[0]
    assert distant.ledger.mean_latency > free.ledger.mean_latency + 1.5
    assert distant.ledger.hit_rate < free.ledger.hit_rate


# 17. The peer that serves a pull is the peer control priced.
#     Control ranks candidates by prefix and prices the pull at that peer's
#     locality tier (NVLink within a node, RDMA across). The store cannot know
#     that: `locate_volumes` returns every holder and the client takes the first,
#     so for a block several instances hold -- a shared system prompt, or
#     anything replicated -- the bytes could come from a different tier than the
#     one predicted. The run installs LongestPrefixKeySelector in the directory and
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


# 18. The source selector serves either view it can be attached to.
#     The scheduler attaches a KVView, whose snapshot one routing decision pins; a
#     run installing this selector on its own can only attach the plain View the mesh
#     built, because a prefix run is a KV-cache notion the store has no reason to
#     know. The ranking branch is the only place the two differ, and a
#     controller-side call carries an already-chosen source and short-circuits
#     before reaching it -- so nothing but this test holds the plain view up.
def test_the_source_selector_accepts_a_plain_view():
    from kvcache_sim.control._source import LongestPrefixKeySelector
    from realsim.simulation import Simulation

    topo = _make_topology(2)
    sim = Simulation(topo)
    keys = _block_keys_for("m0", [0, 1])
    selector = LongestPrefixKeySelector()
    selector.attach(sim.view)

    async def scenario():
        with sim.mesh.installed():
            empty = await selector.select(keys, "s0")
            store = KVStore(sim)
            await store.publish("s1", list(keys), _kv(len(keys)))
            ranked = await selector.select(keys, "s0")
        return empty, ranked

    try:
        empty, ranked = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert empty.sources == ()                  # nobody holds it yet
    assert ranked.sources == ("s1",)            # ...and now the holder is ranked


# 19. Spreading reads: the opt-in source selector breaks the prefix tie on load.
#     Reuse value is a property of the prefix, so replicas of a hot prefix rank
#     identically and the id tie-break sends every read to the same host.
#     SpreadReadsKeySelector counts the grants it issued -- its own bookkeeping, since
#     nothing in the repo observes per-volume load -- and lets that settle the tie,
#     under a bound that keeps a genuinely longer prefix winning.
def _replicated(holders, blocks, *, num=4):
    """A sim where each of ``holders`` holds ``blocks[holder]`` leading blocks.

    Returns ``(sim, keys)``; the caller drives its own scenario on ``sim.loop`` and
    closes it. Publishing is the same real ``put_batch`` every other test here
    uses, so the prefix runs the selector reads come out of the real directory.
    """
    sim = Simulation(_make_topology(num))
    keys = _block_keys_for("m0", list(range(max(blocks.values()))))

    async def fill():
        with sim.mesh.installed():
            store = KVStore(sim)
            for holder in holders:
                held = keys[: blocks[holder]]
                await store.publish(holder, list(held), _kv(len(held)))

    sim.loop.run_until_complete(fill())
    return sim, keys


def _select_heads(sim, selector, keys, *, count, gap=0.0):
    """The head of ``count`` successive rankings, ``gap`` virtual seconds apart."""
    selector.attach(sim.view)

    async def scenario():
        heads = []
        with sim.mesh.installed():
            for _ in range(count):
                heads.append((await selector.select(keys, "s0")).sources[0])
                if gap:
                    await asyncio.sleep(gap)
        return heads

    return sim.loop.run_until_complete(scenario())


def test_spread_reads_rotates_between_equal_prefix_replicas():
    """Three replicas of one prefix, so the ranking has nothing else to go on."""
    sim, keys = _replicated(["s1", "s2", "s3"], {"s1": 4, "s2": 4, "s3": 4})
    try:
        heads = _select_heads(sim, SpreadReadsKeySelector(), keys, count=6)
        # ...and the same replicas under the default selector, which cannot spread.
        stuck = _select_heads(sim, LongestPrefixKeySelector(), keys, count=6)
    finally:
        sim.loop.close()
    assert heads == ["s1", "s2", "s3", "s1", "s2", "s3"]   # id order, then rotate
    assert stuck == ["s1"] * 6                             # every read, one host


def test_spread_reads_still_prefers_a_materially_longer_prefix():
    """Load may shave off ``max_discount`` blocks and no more.

    s2 holds twice the prefix and is granted every read regardless: the discount is
    bounded, so the cost signal can settle a tie but never outvote reuse.
    """
    sim, keys = _replicated(["s1", "s2"], {"s1": 2, "s2": 4})
    try:
        heads = _select_heads(sim, SpreadReadsKeySelector(max_discount=1), keys, count=5)
        # A discount wide enough to cover the gap does trade reuse away, and only
        # once it has been fully spent -- stated here because it is the knob's
        # meaning, not a defect.
        wide = _select_heads(sim, SpreadReadsKeySelector(max_discount=4), keys, count=3)
    finally:
        sim.loop.close()
    assert heads == ["s2"] * 5
    assert wide == ["s2", "s2", "s1"]


def test_spread_reads_forgets_a_grant_once_its_window_passes():
    """A grant is a window, not a running total, so load cannot accumulate.

    Back-to-back the selector rotates; spaced further apart than the window, every
    read finds an empty tally and the id tie-break answers again -- which is the
    difference between "recently routed there" and "has ever been routed there".
    """
    sim, keys = _replicated(["s1", "s2"], {"s1": 4, "s2": 4})
    try:
        prompt = _select_heads(sim, SpreadReadsKeySelector(window=1.0), keys, count=4)
        aged = _select_heads(
            sim, SpreadReadsKeySelector(window=1.0), keys, count=4, gap=2.0
        )
        # window <= 0 is no memory at all: the default selector's ranking, exactly.
        forgetful = _select_heads(sim, SpreadReadsKeySelector(window=0.0), keys, count=4)
    finally:
        sim.loop.close()
    assert prompt == ["s1", "s2", "s1", "s2"]
    assert aged == ["s1"] * 4
    assert forgetful == ["s1"] * 4


def test_spread_reads_ranks_deterministically():
    """Same grants, same clock, same ranking -- every time, whole list.

    The rank key ends in the instance id, so no two sources can compare equal and
    nothing is left for dict or arrival order to decide. Asserted over the full
    ranking rather than the head, because a caller that rejects the first source
    reads the rest of it.
    """
    async def rankings(sim, keys):
        selector = SpreadReadsKeySelector()
        selector.attach(sim.view)
        out = []
        with sim.mesh.installed():
            for _ in range(5):
                out.append((await selector.select(keys, "s0")).sources)
        return out

    holders, blocks = ["s1", "s2", "s3"], {"s1": 4, "s2": 4, "s3": 3}
    first_sim, keys = _replicated(holders, blocks)
    try:
        first = first_sim.loop.run_until_complete(rankings(first_sim, keys))
    finally:
        first_sim.loop.close()
    second_sim, keys = _replicated(holders, blocks)
    try:
        second = second_sim.loop.run_until_complete(rankings(second_sim, keys))
    finally:
        second_sim.loop.close()
    assert first == second
    assert first[0] == ("s1", "s2", "s3")   # prefix first, id breaks the s1/s2 tie
    assert all(len(set(r)) == len(r) == 3 for r in first)


def test_the_spread_reads_flag_reaches_a_scenario_run():
    """The opt-in entry point, exercised from the flag to the ledger.

    ``python -m kvcache_sim hotspot --spread-reads``: the demo declares the flag,
    the scenario reads it off the parsed command line and gives each cache-aware
    run its *own* selector, and which peer serves a pull changes. What does not
    change is that a *planned* pull is fetched from the peer it was priced against
    -- the ranking sits behind the routed-pull memo in the chain the plane answers a
    fetch with, so a different ranking cannot redirect a pull already decided.
    """
    parser = argparse.ArgumentParser()
    KVCacheDemo().flags(parser)
    args = parser.parse_args(["--spread-reads"])

    aware = scenarios.Hotspot(0).runs(args)[1:]
    # The one ranking the plane holds: it prices against it and answers a fetch with
    # it, so this is where a run's own selector is reachable.
    selectors = [run.control._source for run in aware]
    assert all(isinstance(p, SpreadReadsKeySelector) for p in selectors)
    assert selectors[0] is not selectors[1], "a shared tally would count both runs"

    def pull_sources(result):
        return sorted(
            (r.reuse_source for r in result.ledger.rows if r.reuse_source)
        )

    default = run_hotspot(seed=0)[2]
    spread = run_hotspot(seed=0, args=args)[2]
    assert pull_sources(spread) != pull_sources(default)
    assert _unplanned_edges(spread) == 0


# --------------------------------------------------------------------------
# The two control surfaces: questions to the selector, facts to the model,
# both carrying this application's payloads as values.
# --------------------------------------------------------------------------


def _scheduler(n: int = 2):
    """A scheduler attached to a real mesh view, ready to be asked things."""
    sim = Simulation(_make_topology(n))
    sched = LoadBalanceScheduler(block_tokens=512)
    sched.attach(sim.view)
    return sim, sched


def test_one_answer_names_both_of_a_request_s_hosts():
    """One member, one question, answered before anything runs.

    One answer names the prefill host *and* the decode host, which is what lets the
    whole admission decision be taken at the door: there is no second ask, so there
    is no outcome that costs a prefill. The two selections behind it stay inside the
    scheduler: what leaves is the winner of each, and the price of the one that won.
    """
    sim, sched = _scheduler()
    request = _request(
        id="r0", arrival=0.0, prompt_tokens=1024, output_tokens=1,
        block_keys=tuple(_block_keys_for("m0", [0, 1])),
    )

    async def scenario():
        with sim.mesh.installed():
            # The ranking first: committing the decision moves the load the ranking
            # is read off, so asking afterwards would price a different field.
            return (
                await sched._select_prefill(request),
                await sched.decide(request, "s0"),
            )

    try:
        ranked, response = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert response.prefill in sched.prefill_ids
    assert response.decode in sched.decode_ids
    assert response.plan.ttft > 0.0
    # ...and the prefill host is the head of the selection it came from, which ranked
    # every instance in the pool.
    assert sorted(ranked.sources) == sorted(sched.prefill_ids)
    assert response.prefill == ranked.sources[0]


def test_a_refusal_is_a_decision_not_taken():
    """An SLO miss answers ``None``: there is no decision to read anything off.

    A decider refuses by not deciding, which is a different thing from a selection
    that names nobody -- there is no ranking here to be empty.
    """
    sim = Simulation(_make_topology(2))
    sched = LoadBalanceScheduler(block_tokens=512, slo_ttft=0.0)
    sched.attach(sim.view)
    request = _request(
        id="r0", arrival=0.0, prompt_tokens=1024, output_tokens=1,
        block_keys=tuple(_block_keys_for("m0", [0, 1])),
    )

    async def scenario():
        with sim.mesh.installed():
            return await sched.decide(request, "s0")

    try:
        response = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    assert response is None


def test_notify_dispatches_each_fact_and_answers_nothing():
    """The learning half: each fact lands in the cluster model and answers None.

    Told over the handle a host holds, not to the object, so what is exercised is
    the surface a report really crosses.
    """
    sim, sched = _scheduler()
    cluster = LocalClusterModelHandle(ClusterModelService(sched.cluster))
    assert cluster.model is sched.cluster, "the handle refers to the run's one model"

    async def scenario():
        assert await cluster.notify.call_one(ComputeBusy("s0", 7.0)) is None
        assert sched.cluster.busy_until["s0"] == 7.0
        assert await cluster.notify.call_one(DecodeState("s1", (1.0, 2.0))) is None
        assert sched.cluster.occupancy("s1") == 2
        # A completion raises the tail and never lowers it.
        await cluster.notify.call_one(PrefillFinished("s0", 3.0))
        assert sched.cluster.busy_until["s0"] == 7.0

    try:
        sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()


def test_an_unknown_fact_is_refused_not_guessed():
    """A fact this application does not define fails loudly at the surface."""
    sim, sched = _scheduler()
    try:
        with pytest.raises(TypeError, match="is not told"):
            sim.loop.run_until_complete(
                sched.cluster.notify("not a fact")
            )
    finally:
        sim.loop.close()


def test_a_run_declares_one_plane_and_its_fetch_ranking_selects_over_keys():
    """One plane per capability; a selector is a utility it holds.

    So nothing checks a *plane's* subject -- a run hands it the ports and asks it
    whatever it declares. What is checked is the chain behind the fetch answer, at
    every link, because a link reading those keys as anything else would answer a
    question it was not asked.
    """
    sched = scheduler("cache_aware", _make_topology(2))
    assert isinstance(sched, ControlPlane)
    assert not isinstance(sched, KeySelector)   # a plane is asked, not consulted
    assert {"decide", "sources"} <= set(dir(sched))
    with pytest.raises(TypeError, match="every link must be a KeySelector"):
        KeySelectorChain([LongestPrefixKeySelector(), _LocalOnly()])


def test_one_plane_answers_a_fetch_with_the_pull_it_priced():
    """Why one plane: neither the memo nor the ranking crosses a boundary.

    The peer a pull was priced against is recorded in this plane's own model and read
    back by the member a fetch asks, and the ranking behind that memo is the *same
    object* the pricing ranked with. Two planes had to keep both in step; one holds
    them.
    """
    sched = scheduler("cache_aware", _make_topology(2))
    sim = Simulation(_make_topology(2), control=sched)   # attach builds the chain
    try:
        assert sched.cluster is not None
        assert sched._fetch.selectors[0]._cluster is sched.cluster
        assert sched._fetch.selectors[-1] is sched._reuse.source
    finally:
        sim.loop.close()


