"""The kvcache scenarios: six comparisons, each a :class:`realsim.demo.Scenario`.

Each one fixes a topology and a synthetic request stream, declares the
:class:`~realsim.run.Run` values to compare -- same requests, different wiring --
and narrates the results. Nothing here builds a clock, a mesh or a plane, and
nothing here executes: :meth:`realsim.demo.Demo.main` does that with
:meth:`realsim.run.Run.execute`, the same way for every capability.

Each is parameterized by its seed at construction rather than by a CLI flag, so
``SharedPrefix(seed=1).runs()`` gives a test the same configurations the demo
shows, with no command line involved.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from domain import decode_step_time, DEFAULT_MODEL, DEFAULT_PROFILE
from proposed import Endpoint
from proposed.selector import Balance

from realsim.demo import Console, Scenario
from realsim.run import Result, Run
from sim_common.trace import Trace

from ..control._selector import by_prefix_and_load, LongestPrefixKeySelector
from ..report.metrics import Metrics
from ..report.summary import (
    CacheVsBaselineReport,
    DisaggregationReport,
    EarlyRejectionReport,
    EvictionReport,
    HotspotReport,
)
from ._generator import make_workload
from ._serving import BLOCK_TOKENS, KVWorkload, scheduler, serving_plane

__all__ = [
    "TRACE_LIMIT",
    "EVICTION_CAPACITIES",
    "KV_BLOCKS_PER_INSTANCE",
    "DISAGG_TARGET_TBT",
    "DISAGG_MAX_BATCH",
    "EARLY_SLO_TBT",
    "EARLY_MAX_BATCH",
    "SharedPrefix",
    "Eviction",
    "Hotspot",
    "Overload",
    "Disaggregation",
    "EarlyRejection",
]


#: Per-event trace lines a scenario dumps under ``-v``; these runs are long.
TRACE_LIMIT = 60


def _configure(label: str, topology, conversations, kind: str, **knobs) -> Run:
    """One labelled configuration over ``conversations``.

    Every kvcache run is built here -- by the scenarios below and by the tests --
    so the trace format and the metrics ledger a run reports into are chosen once
    and cannot drift between them.
    """
    # The pools are a deployment fact and go to both halves: the data plane gives
    # a host the engines its pools put on it, and control routes to the
    # same pools. ``coupled`` is a separate, fidelity question -- does this run
    # model a prefill and a decode step on one host as colliding? -- so it goes to
    # the data plane alone, which answers it by sharing a timeline or not. So is
    # ``max_batch``: a VRAM cap is a property of the device the batch runs on, and
    # control never asks how large a batch may get -- only how large it is.
    coupled = knobs.pop("coupled", False)
    max_batch = knobs.pop("max_batch", 8)

    # A per-run machine, so a scenario can give its volumes a different capacity.
    # The default is finite: a serving instance's KV memory is a real bound, and a
    # cache that cannot run out of room is not one -- the eviction it never performs
    # would flatter every hit rate here. It is *store* capacity because that is
    # where the blocks are; a store in general is unbounded, which is why this
    # belongs to the capability and not to the profile every sim shares.
    profile = knobs.pop("profile", _KV_INSTANCE_PROFILE)
    return Run(
        label,
        KVWorkload(topology, conversations),
        # One control plane, asked twice per request that pulls: where to run it, and
        # then which peer serves the fetch that plan implies.
        control=scheduler(kind, topology, **knobs),
        data=serving_plane(
            coupled=coupled,
            simulate_decode=knobs.get("simulate_decode", False),
            max_batch=max_batch,
            prefill_pool=knobs.get("prefill_pool"),
            decode_pool=knobs.get("decode_pool"),
        ),
        profile=profile,
        trace=Trace(time_width=8, kind_width=7),
        ledger=Metrics(),
    )


def _make_topology(num: int, per_node: int = 2) -> Dict[str, Endpoint]:
    """Build ``num`` instances laid out ``per_node`` per node (distinct hosts).

    Same-node instances exchange KV over NVLink; cross-node over RDMA -- so *where*
    a reusable prefix lives changes the cost of pulling it. The instance id is also
    its storage-volume id in the real directory.
    """
    topo: Dict[str, Endpoint] = {}
    for i in range(num):
        node = f"N{i // per_node}"
        topo[f"s{i}"] = Endpoint(id=f"s{i}", host=f"h{i}", node=node)
    return topo


def _subset(topology: Dict[str, Endpoint], ids: List[str]) -> Dict[str, Endpoint]:
    """Return the sub-topology for ``ids`` (order-stable)."""
    return {i: topology[i] for i in ids}


def _shared_prefix_workload(seed: int = 0):
    """Many conversations sharing a hot system prompt + per-conv context."""
    return make_workload(
        num_requests=200, num_conversations=8, system_blocks=4,
        conv_base_blocks=4, query_blocks=2, zipf_s=1.1, arrival_rate=2.5,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )



#: Per-instance cache capacities the eviction sweep walks.
#: What one serving instance sets aside for KV, in blocks. Chosen where the
#: eviction sweep below has already plateaued: large enough that capacity is not
#: what any other scenario is measuring, finite because the hardware is.
KV_BLOCKS_PER_INSTANCE = 256
_KV_INSTANCE_PROFILE = replace(
    DEFAULT_PROFILE,
    storage_capacity_bytes=DEFAULT_MODEL.block_bytes(
        KV_BLOCKS_PER_INSTANCE, BLOCK_TOKENS
    ),
)

# In blocks, converted to the volume's byte capacity. The smallest must still fit
# one request's own blocks: a volume now enforces a real limit rather than a model
# one, so a put that cannot fit even after evicting everything is refused, not
# absorbed.
#
# That constraint is what moved the floor from 8 to 48, and the input to it is what
# changed rather than the rule. The largest prompt here used to be
# system(2) + conversation(4) + query(2) = 8 blocks; a conversation's last turn now
# carries its whole history, so it is system(2) + conversation(4) + 8 turns of
# (query(2) + one block of output) minus the first turn's output = 29 blocks -- and
# the floor has to clear that with room for another request's reads to interleave,
# not merely equal it (32 does not; 48 does).
#
# Below that floor the sweep stops measuring eviction and starts measuring
# something this model does not do correctly, which is worth naming because it is a
# real hole rather than a tuning matter. A publish is one ``put_batch``, and
# torchstore registers a batch's keys with the directory *after* every one of them
# has landed -- while the volume evicts, and reports its evictions, key by key as
# they land. So a volume with less slack than the batch is writing drops a key out
# of the batch it is still landing, reports that drop before the key has ever been
# registered, and is then registered for it anyway. The directory ends the run
# naming a volume for blocks that volume threw away, which routes a later read at
# nothing. It is self-healing in behaviour (the read raises and the request
# recomputes, the ``RESTALE`` path) and it is not self-healing in the directory,
# and closing it means changing when a batch is registered, which is upstream of
# this repo. What this constant can do is keep the sweep in the regime it is about:
# a request's own working set fits, and what is measured is whether everybody
# else's does.
EVICTION_CAPACITIES = (48, 64, 96, 128, 192, 256)



def _hotspot_workload(seed: int = 0):
    """One dominant conversation (extreme skew) -> a single hot instance."""
    return make_workload(
        num_requests=160, num_conversations=4, system_blocks=6,
        conv_base_blocks=6, query_blocks=2, zipf_s=2.2, arrival_rate=3.0,
        block_tokens=BLOCK_TOKENS, output_tokens=64, seed=seed,
    )





# --------------------------------------------------------------------------- #
# Decode-side scenarios: TBT under batched decode.
# --------------------------------------------------------------------------- #

# Target TBT for the disaggregation scenario: 5 x the batch=1 baseline step time.
DISAGG_TARGET_TBT = 5 * decode_step_time(1, DEFAULT_PROFILE)
DISAGG_MAX_BATCH = 8



# TBT SLO for the early-rejection scenario: 3 x baseline step time.
EARLY_SLO_TBT = 3 * decode_step_time(1, DEFAULT_PROFILE)
EARLY_MAX_BATCH = 8



class SharedPrefix(Scenario):
    """Cache-aware vs load-balance on the shared-prefix workload (ample capacity)."""

    name = "shared_prefix"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        convs = _shared_prefix_workload(self.seed)
        return [
            _configure("cache_aware", topo, convs, "cache_aware"),
            _configure("load_balance", topo, convs, "load_balance"),
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("SHARED PREFIX: conversations sharing a hot system prompt + context")
        console.info("directory: real torchstore.controller.Controller (off-actor)")
        console.info("4 instances (2 nodes), 200 turns over 30 multi-turn conversations")
        console.info("(<=8 turns each, Zipf skew over how many turns a tenant contributes).")
        console.info("Turn N+1 is turn N's prompt plus turn N's OUTPUT plus a new message, so")
        console.info("the reusable prefix grows every turn and the last turn of a dialogue is")
        console.info("31 blocks against the first turn's 10. Cache-aware routes a turn to the")
        console.info("instance holding that history (or pulls it once), so a conversation is")
        console.info("prefilled ~once; load-balance scatters the turns, recomputing the whole")
        console.info("history on whichever instance it lands on -- which is why the gap here")
        console.info("is far wider than it was on a single-turn stream: what a miss costs now")
        console.info("grows with the conversation.")
        console.trace(results[0].trace, limit=TRACE_LIMIT)
        console.summary(CacheVsBaselineReport("shared_prefix", results))


class Eviction(Scenario):
    """One cache-aware run per capacity; the report reads the hit-rate curve off
    their ledgers. Each run is labelled with its capacity."""

    name = "eviction"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        convs = make_workload(
            num_requests=400, num_conversations=12, system_blocks=2,
            conv_base_blocks=4, query_blocks=2, zipf_s=1.05, arrival_rate=2.5,
            block_tokens=BLOCK_TOKENS, output_tokens=64, seed=self.seed,
        )
        return [
            _configure(
                str(cap), topo, convs, "cache_aware",
                profile=replace(
                    DEFAULT_PROFILE,
                    storage_capacity_bytes=DEFAULT_MODEL.block_bytes(
                        cap, BLOCK_TOKENS
                    ),
                ),
            )
            for cap in EVICTION_CAPACITIES
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("EVICTION: hit rate vs cache capacity (LRU)")
        console.info("400 turns over 55 conversations. As per-instance capacity grows, the")
        console.info("hot working set fits and the prefix hit rate rises, then plateaus")
        console.info("(the ~55% -> ~83% shape). Too-small caches also force more KV")
        console.info("re-fetch (fabric).")
        console.info("The sweep starts at 48 blocks rather than 8 because a conversation's")
        console.info("last turn is a 29-block chain: below that a volume cannot hold one")
        console.info("request's own working set and the curve stops being about eviction.")
        console.info("")
        console.info(EvictionReport(results).render())


class Hotspot(Scenario):
    """Compare (a) baseline, (b) cache-aware no-replication, (c) cache-aware.

    ``--spread-reads`` puts the two cache-aware runs' source ranking under a
    :class:`~proposed.selector.Balance` and folds the two dimensions that leaves with
    :func:`~kvcache_sim.control._selector.by_prefix_and_load`, so longest-prefix-then-id
    becomes longest-prefix-minus-recent-load. This scenario is where it can matter: extreme
    skew replicates one prefix, and every replica of it ranks identically on prefix
    alone. Off by default, and only here -- it changes which replica serves a read, so
    it is not byte-identical.
    """

    name = "hotspot"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        convs = _hotspot_workload(self.seed)
        spread = getattr(args, "spread_reads", False)

        def source():
            """A *fresh* selector per run: one object attached twice senses only the
            view it was attached to last, so two runs would share one load reading."""
            return Balance(LongestPrefixKeySelector()) if spread else None

        # One fold for both runs: it is arithmetic over a key and holds nothing.
        fold = by_prefix_and_load() if spread else None
        return [
            _configure("baseline", topo, convs, "load_balance"),
            _configure("no_replication", topo, convs, "cache_aware", replicate=False,
                 source_selector=source(), source_fold=fold),
            _configure("replication", topo, convs, "cache_aware",
                 balance_threshold=1.2, replicate=True, source_selector=source(),
                 source_fold=fold),
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("HOTSPOT: extreme skew -> hot-block replication trades compute for KV")
        console.info("One dominant tenant (Zipf s=2.2 over 4 ranks, so rank 0 draws ~3 turns in")
        console.info("4), 21 dialogues. Both cache-aware columns route a turn to whoever holds")
        console.info("its history and beat the load-balancing baseline 2.7-3.9x on TTFT.")
        console.info("Replication (balance_threshold 1.2) additionally spreads a shared prefix")
        console.info("to peers read-through, buying strictly less recompute and paying for it")
        console.info("in KV fabric bytes.")
        console.trace(results[2].trace, limit=TRACE_LIMIT)
        console.summary(HotspotReport(results))
        console.info("(replication swaps recompute for cheap KV transfer when spreading a")
        console.info(" hot prefix to a peer -> fewer prefill tokens, more fabric bytes.)")
        console.info("A CLAIM WITHDRAWN, and it is the one this scenario used to lead with:")
        console.info("that replication also cuts p90 TTFT. It does not on a multi-turn stream")
        console.info("-- it wins on some seeds and loses on others, and the TTFT columns above")
        console.info("are noise between the two cache-aware runs. The hotspot it was spreading")
        console.info("was an artifact of one-shot requests: a dominant *conversation* whose")
        console.info("every request carried one identical fixed prefix really did pile onto the")
        console.info("single instance holding it. A dominant *tenant* is many dialogues, each")
        console.info("with its own growing history, so cache-aware routing already scatters")
        console.info("them and there is no pile left to spread; what they still share is the")
        console.info("12-block opening, a shrinking fraction of a prompt that grows. This is")
        console.info("not retuned to look otherwise: restoring the phenomenon means a workload")
        console.info("with one very long dialogue, which is a scenario change to argue for")
        console.info("separately.")


class Overload(Scenario):
    """High arrival rate + a TTFT SLO -> some requests must be rejected."""

    name = "overload"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        convs = make_workload(
            num_requests=300, num_conversations=6, system_blocks=6,
            conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=9.0,
            block_tokens=BLOCK_TOKENS, output_tokens=64, seed=self.seed,
        )
        slo = 6.0
        return [
            _configure("cache_aware", topo, convs, "cache_aware", slo_ttft=slo),
            _configure("load_balance", topo, convs, "load_balance", slo_ttft=slo),
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("OVERLOAD: high arrival + TTFT SLO -> rejections")
        console.info("300 turns over 39 conversations at a high rate, TTFT SLO 6.0. Prefix")
        console.info("reuse shortens prefill, freeing capacity, so cache-aware admits more")
        console.info("requests (fewer rejections) than the load-balancing baseline. Multi-turn")
        console.info("widens that gap: what the baseline recomputes on a miss is a whole")
        console.info("conversation, so it sheds ~28% more of the same offered load.")
        console.info("A refused turn does not end its conversation -- the next turn is offered")
        console.info("anyway -- so both columns are shedding from the same 300, which is what")
        console.info("makes the counts comparable.")
        console.summary(CacheVsBaselineReport("overload", results))


class Disaggregation(Scenario):
    """Disaggregating prefill from decode protects TBT (Mooncake's headline).

    Both configs run with admission disabled (``slo_tbt=inf``, so the TBT gate can
    refuse nobody) and every request is served and measured. Decode capacity is
    fixed (two instances, ``DISAGG_MAX_BATCH`` each); the only difference is whether
    those two instances also do prefill. Returns ``[disaggregated, coupled]``.
    """

    name = "disaggregation"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)  # s0..s3
        convs = make_workload(
            num_requests=120, num_conversations=8, system_blocks=2,
            conv_base_blocks=2, query_blocks=1, zipf_s=1.1, arrival_rate=1.2,
            block_tokens=BLOCK_TOKENS, output_tokens=12, seed=self.seed,
        )
        common = dict(
            simulate_decode=True, slo_tbt=float("inf"), max_batch=DISAGG_MAX_BATCH,
        )
        return [
            _configure("disaggregated", topo, convs, "cache_aware",
                 prefill_pool=["s0", "s1"], decode_pool=["s2", "s3"], **common),
            _configure("coupled", _subset(topo, ["s2", "s3"]), convs, "cache_aware",
                 coupled=True, **common),
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("DISAGGREGATION: dedicated decode pool protects TBT from prefill")
        console.info("Two decode instances, VRAM cap 8, TBT target %.3f. Admission is",
                     DISAGG_TARGET_TBT)
        console.info("disabled (no TBT SLO gate), so BOTH configs serve every request -- the")
        console.info("contrast is purely the TBT-target attainment among served requests, not")
        console.info("a rejection count. The only difference is prefill placement: disaggregated")
        console.info("prefills on a separate pool (s0/s1) so decode (s2/s3) keeps its own")
        console.info("compute timeline; coupled runs prefill AND decode on s2/s3, so a prefill")
        console.info("can collide with a decode step and spike that request's inter-token gap.")
        console.trace(results[0].trace, limit=TRACE_LIMIT)
        console.summary(DisaggregationReport(results, DISAGG_TARGET_TBT))
        console.info("Attainment is the fraction of served requests whose *worst* inter-token")
        console.info("gap stayed under the target. Disaggregation isolates decode from prefill,")
        console.info("so served requests hold TBT; coupling lets long prefills stall decode, so")
        console.info("a large fraction of served requests blow the target -- same load admitted.")
        console.info("What isolation costs is the handoff row: a decode host that did not")
        console.info("prefill the prompt fetches its whole KV chain back out of the store, which")
        console.info("is a real transfer on the clock and the reason both columns pay it here")
        console.info("(coupled picks a decode host by load, not by who prefilled). It delays")
        console.info("every request without widening any inter-token gap, so it shows up in")
        console.info("neither TBT column -- it shows up end to end, which the client measures")
        console.info("from arrival to last token and which is more than half the disaggregated")
        console.info("column. That is the trade, and it changes sign: disaggregation wins every")
        console.info("per-token number here and loses the wall clock to pay for it. Both halves")
        console.info("grew with multi-turn, because what a handoff moves is the conversation's")
        console.info("whole history and that is what now grows: ~2.3x the handoff bytes of the")
        console.info("single-turn stream for the same 120 requests.")
        console.info("The decode KV row is the other half of that bill, and it is memory rather")
        console.info("than time: a decode host holds the chain it pulled in AND the KV its")
        console.info("generation appends, so it accumulates blocks it never prefilled. The")
        console.info("disaggregated column holds ~3x the coupled one for the same load --")
        console.info("coupling decodes most requests where they were prefilled, so the chain is")
        console.info("already there and only the generated blocks are new. Neither runs out of")
        console.info("room at this capacity; a decode pool sized for its own KV and not for")
        console.info("everybody else's would.")


class EarlyRejection(Scenario):
    """What the TBT gate is fed: current occupancy (``early``) or foreseen
    (``predict``)."""

    name = "early_rejection"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def runs(self, args=None) -> List[Run]:
        topo = _make_topology(4)
        convs = make_workload(
            num_requests=160, num_conversations=6, system_blocks=6,
            conv_base_blocks=4, query_blocks=2, zipf_s=1.3, arrival_rate=20.0,
            block_tokens=BLOCK_TOKENS, output_tokens=32, seed=self.seed,
        )
        common = dict(
            simulate_decode=True, slo_tbt=EARLY_SLO_TBT, max_batch=EARLY_MAX_BATCH
        )
        return [
            _configure(mode, topo, convs, "cache_aware", early_rejection=mode, **common)
            for mode in ("early", "predict")
        ]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        console.section("EARLY REJECTION: gate decode admission on the load you foresee")
        console.info("Heavy decode load with a tight TBT SLO of %.3f. Both cache-aware runs",
                     EARLY_SLO_TBT)
        console.info("gate at routing, before the prefill runs, so a refusal costs no compute;")
        console.info("they differ only in the decode occupancy the gate is fed. 'early' reads")
        console.info("the last report; 'predict' rolls it forward to this request's prefill")
        console.info("completion, counting the prefills already promised that will have landed")
        console.info("by then. Neither rejects anything at this load, so what the columns show")
        console.info("is where each one's decode selection put the requests.")
        console.trace(results[1].trace, limit=TRACE_LIMIT)
        console.summary(EarlyRejectionReport(results, EARLY_SLO_TBT))
        console.info("A CLAIM WITHDRAWN: that TBT attainment separates 'predict' (routes on the")
        console.info("load foreseen at prefill completion) from 'early' (routes on a current")
        console.info("occupancy a slow prefill leaves reading empty). It does not on a")
        console.info("multi-turn stream, and the reason is structural rather than a matter of")
        console.info("degree. A conversation is a closed loop -- a user cannot send turn N+1")
        console.info("before turn N answers -- so at most one request per open dialogue is ever")
        console.info("in flight, and this scenario's 160 turns arrive as ~22 concurrent")
        console.info("conversations instead of as a burst at rate 20. Occupancy is then never")
        console.info("far from what a stale snapshot reports, the two selectors route almost")
        console.info("identically, and across four seeds the attainment gap is noise whose sign")
        console.info("changes. Restoring the mechanism means offering the concurrency these")
        console.info("constants intended -- many more, shorter conversations -- and no such")
        console.info("change tried so far restores it across seeds, so it is left broken and")
        console.info("said out loud rather than tuned into looking fixed.")
        console.info("The decode KV row is the difference nobody went looking for, and it is the")
        console.info("cost of the routing each one picked: a request decoded away from its")
        console.info("prefill host drags that host's whole chain onto the decode host's volume")
        console.info("and leaves it there, and that chain is a conversation's history rather")
        console.info("than a fixed 12 blocks. This capacity pressure is real: these volumes do")
        console.info("evict under it, which the eviction sweep would show if it modelled decode.")
