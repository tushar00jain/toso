"""The kvcache workload, and the wiring a run installs around it.

Two separate things, deliberately:

* :class:`KVWorkload` is *the work* -- a stream of conversations, one
  :class:`~realsim.runner.WorkItem` per conversation at its first turn's arrival
  time. It builds no store, no scheduler and no plane;
* :func:`scheduler` and :func:`serving_plane` are the *capability wiring*, one
  per plane: the scheduler over the view, and the store plus one
  :class:`~kvcache_sim.data.serving.ServingHost` per instance over it. Both are
  factories because they reach for the view, the mesh and the ledger, none of
  which exists before the stack does.

Three things a deployment would not need are built here, because in a deployment
they are not built at all. One is
:class:`~kvcache_sim.workload._accelerator.SimulatedAccelerator`: what a forward
pass costs, how it is made to take that long, and what KV it leaves behind are the
run's answers under simulation and the model's in production, so the capability is
handed a port and this supplies the implementation. It is why the store below is
built with nothing but the deployment -- what a KV block *is* arrives at the store
as an argument, from the accelerator, per request. The other two are the client:
:func:`_affinity` decides which host a request lands on and :class:`_Client` takes
it there and follows the redirects it gets back. Production has a client SDK, an
ingress proxy or DNS doing that, none of which is part of the serving system --
which is why they are here rather than in ``data/``, whose test for membership is
whether a thing advances the clock or moves bytes, and whose contents are what
would lift into a deployment unchanged.

The client stands in for a fourth thing that a deployment does not have on any of
its machines either: the **user**. A conversation's turns are serial because a
reply cannot be written before the answer arrives, so somebody has to hold the
dialogue open between them, and it is the caller.

The client is where the request's itinerary lives
-------------------------------------------------
A serving host answers with an *address* rather than acting on it: the host a
request lands on says which host should prefill it, and that host says which host
will decode it (see :mod:`kvcache_sim.data.serving`). Somebody has to walk that
chain, and it is the same somebody who walks a ``307`` -- the client. So
:class:`_Client` makes three calls per request where the old wiring made one and
let the hosts call each other twice, and it reaches each host through a
:class:`~realsim.seams.link.LocalEndpoint` over a shared
:class:`~realsim.seams.link.ServiceHop`, so those three round trips are charged
(:attr:`sim_common.config.SimConfig.client_rtt`, ``0.0`` by default, which keeps a
hop inline and the run byte-identical).

Walking the chain is also what makes the client the only thing that can time the
request end to end, so it does: the last leg returns at the last token, and the
client stamps ``now - request.arrival`` onto the row. That replaced a drain pass
the run used to need -- when decode outlived the request's coroutine, something
had to keep the loop alive for the tail, and nothing was left holding the request
to measure it.

They are two functions because they are two services. The plane factory does not
build the scheduler; it takes ``sim.placement_handle`` and ``sim.cluster_handle``,
the handles :meth:`realsim.run.Run.execute` put in front of whatever
:func:`scheduler` returned and of the model it decides against. A scenario names
both functions on a :class:`~realsim.run.Run`: same workload, different wiring,
which is exactly what "cache-aware vs load-balance" means.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Callable, Dict, List, Optional
from zlib import crc32

from domain import DEFAULT_MODEL, DEFAULT_PROFILE
from proposed import Endpoint, KeySelector
from realsim.runner import ItemDispatch, WorkItem
from realsim.seams.link import LocalEndpoint, ServiceHop
from realsim.simulation import Simulation
from realsim.run import Workload
from sim_common import config

from ..control._cluster import KVClusterModel
from ..control._source import LongestPrefixKeySelector
from ..control.request import Request
from ..control.scheduler import (
    CacheAwareScheduler, FetchRouting, LoadBalanceScheduler, predicts_decode,
)
from ._accelerator import BLOCK_TOKENS, SimulatedAccelerator
from ..data._decode import DecodeEngine
from ..data._prefill import PrefillEngine
from ..data.serving import ServingHost
from ..data.store import KVStore
from ..report.metrics import Metrics

# ``BLOCK_TOKENS`` is imported rather than declared: how much of a prompt one KV
# block covers is the engine's cache-page size, so it lives with the accelerator
# that lays the KV out (:mod:`kvcache_sim.workload._accelerator`). It is re-exported
# here because the *scheduler* has to be told the same number -- it prices a prefix
# match in blocks -- and a run that told the two different numbers would route on
# one geometry and store in another.

__all__ = ["BLOCK_TOKENS", "KVWorkload", "scheduler", "serving_plane"]


class KVWorkload(Workload):
    """A stream of **conversations** over a set of serving instances.

    One work item per conversation, released when its *first* turn arrives. Which
    scheduler serves them is not this object's business -- see
    :func:`serving_plane`.

    One item per conversation rather than per request, because a conversation is a
    closed loop: turn N+1 is turn N's prompt plus turn N's output plus a new
    message, so it cannot be submitted until turn N has answered, and a user
    cannot type a reply to an answer they have not seen. The item is therefore the
    dialogue and the client walks its turns in order (:class:`_Client`), which
    leaves :meth:`realsim.runner.Runner.run`'s ``gather`` giving concurrency
    *across* conversations while turns *within* one stay strictly serial. Nothing
    in the harness had to learn about any of this: a work item was always allowed
    to be a coroutine that lives a while, and the client only became able to hold
    one open to its last token when the decode leg started answering there.

    The consequence worth naming is that only the first turn has a release time.
    Every later turn arrives when the run says it does, so a run's arrival process
    is now partly its own output -- see :mod:`kvcache_sim.workload._generator`.
    """

    def __init__(self, topology: Dict[str, Endpoint], conversations) -> None:
        super().__init__(topology)
        self.conversations = list(conversations)

    @property
    def requests(self) -> List[Request]:
        """Every turn of every conversation, flattened, in dialogue order.

        What a reader of the *outcome* wants: the ledger has one row per request,
        so anything reconciling rows against the work that produced them (the
        residency invariant, the tests that check where a request was served) needs
        the requests, not the dialogues holding them. Derived rather than stored,
        so there is one list of turns and it lives on the conversations.
        """
        return [r for c in self.conversations for r in c.requests]

    def items(self, sim: Simulation) -> List[WorkItem]:
        """One item per conversation; the client walks that dialogue's turns."""
        return [
            WorkItem(id=c.id, release_time=c.arrival, payload=c)
            for c in self.conversations
        ]


def scheduler(
    kind: str,
    topology: Dict[str, Endpoint],
    *,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "early",
    source_selector: Optional[KeySelector[None]] = None,
) -> List[object]:
    """This run's **two control planes**, as objects a scenario can just declare.

    ``kind`` is ``"cache_aware"`` (the scheduler under test) or ``"load_balance"``
    (the baseline). Knobs only: the stack's ports arrive later through
    :meth:`~proposed.plane.ControlPlane.attach`, which is what lets these be values
    rather than factories the harness must call at the right moment.

    Two planes because kvcache decides in two places, and where a plane is reached
    from is its type. The :class:`~realsim.run.Run` fronts the
    :class:`~proposed.selector.AnySelector` with a
    :class:`~realsim.seams.placement_handle.LocalPlacementHandle` (it decides
    compute placement) and installs the :class:`~proposed.selector.KeySelector` in the
    directory (it answers the store's routing question). They share the cluster
    model, which is why it is built here: the scheduler prices a pull and records
    it there, and the chain answers the fetch with it.

    **Ordered.** The scheduler is last because both planes bring up the one
    ``source_selector``, and the scheduler's attach is the one that leaves it sensing
    through the pinned :class:`~kvcache_sim.control._view.KVView` a routing decision
    reads (:meth:`~kvcache_sim.control.scheduler._Scheduler.attach`).

    ``source_selector`` is the one knob that is an object rather than a value: which
    peer serves a prefix gap is a :class:`~proposed.selector.KeySelector`, and it keeps
    state across the decisions it makes
    (:class:`~kvcache_sim.control._source.SpreadReadsKeySelector`). ``None`` -- the
    default -- is :class:`~kvcache_sim.control._source.LongestPrefixKeySelector`. Give
    each run its own: two runs sharing one would tally each other's grants and
    neither would reproduce alone.
    """
    if kind not in ("cache_aware", "load_balance"):
        raise ValueError(f"unknown scheduler kind {kind!r}")
    source = source_selector if source_selector is not None else LongestPrefixKeySelector()
    # Over ALL instances: the prefill and decode pools may each be a subset.
    cluster = KVClusterModel(
        sorted(topology), lookahead=predicts_decode(simulate_decode, early_rejection)
    )
    knobs = dict(
        block_tokens=BLOCK_TOKENS,
        profile=DEFAULT_PROFILE,
        slo_ttft=slo_ttft,
        slo_tbt=slo_tbt,
        simulate_decode=simulate_decode,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        early_rejection=early_rejection,
        cluster=cluster,
    )
    if kind == "cache_aware":
        # The ranking goes to the scheduler only because its reuse placement is
        # what names a peer; the baseline never pulls and never asks one.
        placement = CacheAwareScheduler(
            balance_threshold=balance_threshold, replicate=replicate,
            source_selector=source, **knobs
        )
    else:
        placement = LoadBalanceScheduler(**knobs)
    return [FetchRouting(cluster, source), placement]


class _ServingEndpoints:
    """One serving host as a **client** reaches it: endpoint-shaped, hop-charged.

    Standing in for whatever a client SDK holds for a serving instance -- a
    connection, a stub, a Monarch handle over its actor -- as one
    :class:`~realsim.seams.link.LocalEndpoint` per member, over a shared
    :class:`~realsim.seams.link.ServiceHop`. A client is off the box, so reaching a
    host is a boundary like reaching the directory or the control plane and is
    charged like one: free by default, so a run that does not ask for the fidelity
    is byte-identical.

    All three of a host's members are here, because the client calls all three --
    that is what a redirect chain is. Nothing else in the run holds one of these:
    hosts do not, which is the whole point of the redirect model. This class
    replaces the peer handle a host used to be given, and it is deliberately not
    the same object under a new name: what changed is not the shape of the
    reference but *who* is entitled to hold one.
    """

    def __init__(self, host: ServingHost, hop: ServiceHop) -> None:
        self.route = LocalEndpoint(host.route, hop)
        self.prefill = LocalEndpoint(host.prefill, hop)
        self.decode = LocalEndpoint(host.decode, hop)


def _affinity(ids: List[str]) -> Callable[[Request], str]:
    """Which host a request lands on: same conversation, same host.

    The load balancer a deployment has and a simulation has to stand in for. Client
    affinity rather than round robin because it is what a real front end does with
    a session id, and because it is the arrival selector redirected least: a
    conversation's requests share a prefix, so the host that served the last one is
    usually the host that should serve the next.

    ``request.conversation`` is the **tenant**, not the individual dialogue, and
    that is what makes the sentence above true twice over now: a tenant's dialogues
    all open with the same per-tenant context, and each dialogue's own growing
    history belongs to exactly one tenant, so keeping a tenant together keeps both
    on one volume. Routing per dialogue would scatter the shared opening across
    every instance and buy nothing back.

    Note what it deliberately does *not* do: it never looks at the block keys. An
    arrival selector that routed by cache contents would be doing control's job
    with none of control's information, and the comparison this whole
    package exists to make would be measuring itself.

    Deterministic across runs and platforms: ``crc32`` of the conversation id, not
    Python's salted ``hash``.
    """
    ordered = sorted(ids)

    def landed(request: Request) -> str:
        return ordered[crc32(request.conversation.encode()) % len(ordered)]

    return landed


class _Client:
    """The thing outside the cluster that holds a conversation and follows redirects.

    Stands in for what a deployment already has in front of its hosts -- a client
    SDK doing client-side balancing, an ingress proxy, DNS -- and therefore for
    something that is deleted rather than moved when this ships. It holds no selector
    of its own beyond ``landed``, and it never asks where a request should *run*:
    it asks a host, and does what it is told.

    It stands in for one more thing, and that one has no server-side counterpart at
    all: the **user**. A conversation is a turn, an answer, a pause, and another
    turn, and the only participant present for all four is the caller. So
    :meth:`submit` is a loop over a dialogue's turns and :meth:`_turn` is the
    single-request walk that used to be the whole object.

    Three legs per turn, which is nearly the rest of it::

        plan = await hosts[landed(request)].route(request)      # "prefill is B"
        decode, first = await hosts[plan.prefill].prefill(plan)  # "decode is C"
        rest = await hosts[decode].decode(plan)                 # ...returns at the
                                                                # last token
        metrics.completed(request.id, now - request.arrival, 1 + len(rest))

    The last line is the rest of it, and it is the one measurement this object is
    entitled to make. A client does not learn the hit rate or the handoff bytes --
    each host records its own half of those into the run's ledger, and a client
    that had to be told them would be part of the serving system. But how long the
    request took is not a fact about a host: it spans two of them and the
    redirects between, so the only participant present at both ends is the caller,
    which is also who a latency SLO is written for. Server-side timing and
    client-side timing are different numbers in every real deployment, and this is
    the client-side one.

    The same argument makes it the only thing that holds the whole **answer**. The
    output arrives in two pieces from two machines -- the first token out of the
    prefill's last position, the remaining ``output_tokens - 1`` out of the decode
    batch -- and no host ever holds both. So the client concatenates them, and how
    many tokens the request produced becomes something the run *counted* rather
    than the ``output_tokens`` the workload asked for read back out of the
    request. The two agree today, and the point is that they are now two numbers
    that could disagree: one is what was asked for and one is what was made.

    That the stamp is even possible is what the decode leg's shape now buys. It
    used to answer at *admission* -- the batch stepped on afterwards as its own
    task -- so this coroutine returned long before the last token and the run
    needed a separate drain pass to keep the loop alive for the tail. Now the leg
    answers when the request is done, so waiting for the answer and waiting for
    the run to finish are the same act, and the drain is gone rather than
    replaced.

    Note what it carries. The :class:`~kvcache_sim.control.scheduler.Plan` -- a
    value control issued, which is exactly the kind of thing a client is
    handed and hands back (a routing token, a session ticket) and which nothing
    here reads for a decision.

    It also does not second-guess an address. ``prefill`` answers with the decode
    host rather than the client reading ``plan.decode``, because whether there is a
    next leg at all is the serving host's answer and not a field to interpret --
    ``None`` means the journey ends here.

    Args:
        hosts: ``instance id -> _ServingEndpoints``. References, not objects: a
            client is off the box, so each of the three legs is a charged round
            trip rather than a free method call.
        landed: which host a request arrives at.
        metrics: the run's ledger, written to exactly once per *completed*
            request and never read. Not a hop: the stamp is taken on this side of
            the wire and reporting it is the harness collecting what the client
            already knew, not a call the client makes into the cluster.
    """

    def __init__(
        self,
        hosts: Dict[str, _ServingEndpoints],
        landed: Callable[[Request], str],
        metrics: Metrics,
    ) -> None:
        self.hosts = hosts
        self.landed = landed
        self.metrics = metrics

    async def submit(self, item) -> None:
        """Walk one conversation: pause, submit a turn, wait for it, repeat.

        The whole of the multi-turn model on this side, and it is a ``for`` loop
        with an ``await`` in it. Turn N+1's prompt *contains* turn N's answer, so
        it cannot exist until turn N has one; a client that submitted the two
        concurrently would be a client that knew what the model was going to say.
        The pause in front of each turn is the user reading and typing
        (:class:`~kvcache_sim.workload._generator.Turn`), and it is slept here
        rather than scheduled by the runner because when it *starts* is when the
        previous answer landed, which nothing outside this coroutine knows.

        The arrival stamped on each turn is taken here, for the same reason: what
        the generator could state was the instant the turn would arrive if the
        system answered instantly, and what a latency has to be measured from is
        when the request really entered. The two agree exactly for a conversation's
        first turn.

        A refused turn does not end the conversation. The user was told no, and
        the model of a user this workload has does not retry and does not give up
        -- see the generator for why the alternative (ending the dialogue) makes a
        rejection count incomparable between the two configurations a scenario is
        rejecting differently.
        """
        conversation = item.payload
        loop = asyncio.get_running_loop()
        for turn in conversation.turns:
            if turn.think:
                await asyncio.sleep(turn.think)
            await self._turn(replace(turn.request, arrival=loop.time()))

    async def _turn(self, request: Request) -> None:
        """Take one turn through as many hosts as it is redirected to.

        Returns when the turn is finished, which for a run that models decode
        means its last token has landed -- which is also what makes it safe for
        :meth:`submit` to treat this as "the user now has the answer". Every early
        return is a journey that ended, not one abandoned: the host that ended it
        recorded why, and neither leaves anything decoding behind this coroutine.
        """
        plan = await self.hosts[self.landed(request)].route.call_one(request)
        if plan is None:
            return  # refused at the door; the host that refused recorded it
        # The prefill leg answers with both: the first token it produced, and the
        # address of whoever generates the rest. The token comes back even when the
        # address does not -- a prefill that happened is a prefill that emitted one.
        decode, first_token = await self.hosts[plan.prefill].prefill.call_one(plan)
        if decode is None:
            # Nothing after prefill: no decode modelled, or it was shed. The client
            # holds one token and the run models no more of this request, so there
            # is nothing to report -- the same reason the latency below is not
            # stamped on this path, and not a place to invent a shorter one.
            return
        output = [first_token, *await self.hosts[decode].decode.call_one(plan)]
        # Stamped only here, on the one path where a last token exists. A refused
        # request has no end-to-end latency and a prefill-only run has no last
        # token, so both leave the field at its default rather than reporting a
        # shorter interval under the same name.
        #
        # The token count travels with it because it is the same kind of fact: what
        # the client received, counted off the tokens themselves rather than read
        # back off ``request.output_tokens``, which is what was *asked for*.
        self.metrics.completed(
            request.id,
            asyncio.get_running_loop().time() - request.arrival,
            output_tokens=len(output),
        )


def serving_plane(
    *,
    coupled: bool = False,
    simulate_decode: bool = False,
    max_batch: int = 8,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
) -> Callable[[Simulation], ItemDispatch]:
    """Build the factory for this run's **data plane**: one host per instance.

    The pools are the deployment's answer to which hosts run what, and giving a
    host an engine is how it is told.

    ``coupled`` is the remaining question, and it is a *fidelity* one rather than a
    placement one: a host in both pools really has one accelerator, but a run may
    model its prefill and decode as not contending, which several scenarios here do
    and which is the historical default. It is answered by handing the two engines
    one timeline or two, so the host reads it off the objects instead of being told
    twice.

    ``simulate_decode`` is not that question. It asks whether this *run* models the
    request's second half at all, which is a scenario's choice and not a fact about
    any host; a host is told so it knows whether a finished prefill is the end of
    the request.
    """

    def build(sim: Simulation) -> ItemDispatch:
        # The simulation *is* the deployment: it vends the client for an instance
        # and holds the directory, and that is the whole of what the store needs.
        # What a KV block is -- and how big one is -- is the accelerator's answer
        # below, so there is nothing left for the run to hand the store.
        store = KVStore(sim.mesh)
        hop = ServiceHop(config.current().client_rtt)
        hosts: Dict[str, ServingHost] = {}

        prefills = set(prefill_pool) if prefill_pool else set(sim.ids)
        decodes = (set(decode_pool) if decode_pool else set(sim.ids)) if simulate_decode else set()
        for instance in sorted(sim.ids):
            def accelerator() -> SimulatedAccelerator:
                return SimulatedAccelerator(
                    profile=DEFAULT_PROFILE,
                    model=DEFAULT_MODEL,
                    block_tokens=BLOCK_TOKENS,
                )

            compute = accelerator()
            # One accelerator when the run models the collision; two when it does
            # not, which is what ``coupled=False`` on a host in both pools means.
            prefill_compute = compute if coupled else accelerator()
            hosts[instance] = ServingHost(
                instance, store, sim.placement_handle, sim.cluster_handle,
                trace=sim.trace, metrics=sim.ledger,
                prefill=(
                    PrefillEngine(prefill_compute)
                    if instance in prefills else None
                ),
                decode=(
                    DecodeEngine(compute, max_batch=max_batch)
                    if instance in decodes else None
                ),
                models_decode=simulate_decode,
            )
        # The endpoints exist only on the client side, so there is no cycle to
        # close: every host was fully constructed above, knowing nothing about any
        # other, and this loop just gives the client a way to reach each of them.
        client = _Client(
            {i: _ServingEndpoints(h, hop) for i, h in sorted(hosts.items())},
            _affinity(sorted(sim.ids)),
            sim.ledger,
        )
        # What the runner drives is a dispatcher, not a plane: the executing half
        # of this capability is the hosts, and there are several. The rows are
        # published at rejection, at acceptance, or when the last decode token
        # lands -- never one per item, so the harness must not write them.
        #
        # Nothing to drain after the items, either, and that is a property of the
        # client above rather than an omission: its coroutine now lives until its
        # request's last token, so the runner's ``gather`` over every item is
        # already a wait for every decode batch to empty.
        return ItemDispatch(client.submit, writes_own_outcomes=True)

    return build
