"""The kvcache workload, and the wiring a run installs around it.

Two separate things, deliberately:

* :class:`KVWorkload` is *the work* -- a stream of conversations, one
  :class:`~realsim.runner.WorkItem` per conversation at its first turn's arrival
  time. It builds no store, no scheduler and no plane;
* :func:`scheduler` and :func:`serving_plane` are the *capability wiring*, one per
  plane: the control plane over the view, and the store plus one
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
it there and holds the dialogue. Production has a client SDK, an
ingress proxy or DNS doing that, none of which is part of the serving system --
which is why they are here rather than in ``data/``, whose test for membership is
whether a thing advances the clock or moves bytes, and whose contents are what
would lift into a deployment unchanged.

The client stands in for a fourth thing that a deployment does not have on any of
its machines either: the **user**. A conversation's turns are serial because a
reply cannot be written before the answer arrives, so somebody has to hold the
dialogue open between them, and it is the caller.

What the client does *not* hold is the addresses
-----------------------------------------------
A serving host answers with an *address* when the work belongs somewhere else, and says
in its own declaration where in that answer the address is
(:mod:`kvcache_sim.data.serving`), so going there is machinery: a
:class:`~proposed.routed.RoutedPlane` over
:meth:`~proposed.deployment.Deployment.plane_handle`, calling the same member at the
host it was sent to. Every call it makes crosses the client-to-host boundary and is
charged there (:attr:`sim_common.config.SimConfig.client_rtt`, ``0.0`` by default, which
keeps a hop inline and the run byte-identical), so a turn costs two round trips, or
three when its prefill answers with an address -- and never more, because the host that
address names is told the decision that was already made rather than making its own.

What is left is the client's own, and it is two things: which host a turn lands on
(:func:`_affinity`), which no host can answer, and the end-to-end stamp. The last
call returns at the last token, so the caller is the one participant present at both
ends -- and server-side timing is a different number.

They are two functions because they are the two halves. The plane factory does not
build the scheduler; it takes ``sim.control_plane_handle`` and
``sim.dispatcher_handle``, the handles :meth:`realsim.run.Run.execute` put in front of
whatever :func:`scheduler` returned and of the dispatcher its hosts report into. A
scenario names both functions on a :class:`~realsim.run.Run`: same workload, different
wiring,
which is exactly what "cache-aware vs load-balance" means.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Callable, Dict, List, Optional
from zlib import crc32

from domain import DEFAULT_MODEL, DEFAULT_PROFILE
from proposed import ControlPlane, Endpoint, RoutedPlane
from realsim.runner import ItemDispatch, WorkItem
from realsim.simulation import Simulation
from realsim.run import Workload

from ..control.request import Request
from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ._accelerator import BLOCK_TOKENS, SimulatedAccelerator
from ..data._decode import DecodeEngine
from ..data._prefill import PrefillEngine
from ..data.serving import ServingHost
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
    *,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "early",
    source: str = "prefix",
) -> ControlPlane:
    """This run's **control plane**, as an object a scenario can just declare.

    ``kind`` is ``"cache_aware"`` (the scheduler under test) or ``"load_balance"``
    (the baseline). Knobs only: the stack's ports arrive later through
    :meth:`~proposed.plane.ControlPlane.attach`, which is what lets this be a value
    rather than a factory the harness must call at the right moment.

    One plane, asked where a request should run and, later, which peer serves the
    fetch that plan implies. Its sensors are the plane's own, built where it learns
    its instances (:meth:`~kvcache_sim.control.scheduler._Scheduler.attach`).

    ``source`` names which peers a fetch is answered from -- ``"prefix"`` or
    ``"spread"`` -- and the plane builds it, so two runs configured alike still get a
    ranking each: one object attached twice senses only the view it was attached to
    last, and neither run would reproduce alone.
    """
    if kind not in ("cache_aware", "load_balance"):
        raise ValueError(f"unknown scheduler kind {kind!r}")
    knobs = dict(
        block_tokens=BLOCK_TOKENS,
        profile=DEFAULT_PROFILE,
        slo_ttft=slo_ttft,
        slo_tbt=slo_tbt,
        simulate_decode=simulate_decode,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        early_rejection=early_rejection,
        source=source,
    )
    if kind == "cache_aware":
        # The same ranking twice over: the reuse axis names a peer while pricing, and
        # a fetch is answered with it. The baseline never pulls, so it prices against
        # nobody -- and still answers a fetch with the ranking.
        return CacheAwareScheduler(
            balance_threshold=balance_threshold, replicate=replicate, **knobs
        )
    return LoadBalanceScheduler(**knobs)


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
    """The thing outside the cluster that holds a conversation and times its turns.

    Stands in for what a deployment already has in front of its hosts -- a client
    SDK doing client-side balancing, an ingress proxy, DNS -- and therefore for
    something that is deleted rather than moved when this ships. It never asks where
    a request should *run*: it calls the host the request landed on and is sent
    wherever that host says (:class:`~proposed.routed.RoutedPlane`).

    It stands in for one more thing, and that one has no server-side counterpart at
    all: the **user**. A conversation is a turn, an answer, a pause, and another
    turn, and the only participant present for all four is the caller. So
    :meth:`submit` is a loop over a dialogue's turns and :meth:`_turn` is one turn.

    Two things are genuinely the caller's, and they are all that is left here. Where
    a turn lands, because no host can answer that. And how long the turn took, which
    is not a fact about a host either: it spans two of them and the reroutes
    between, so the only participant present at both ends is the caller, which is
    also who a latency SLO is written for. Server-side timing and client-side timing
    are different numbers in every real deployment, and this is the client-side one.
    A client learns neither the hit rate nor the handoff bytes -- each host records
    its own half of those, and a client that had to be told them would be part of the
    serving system.

    The same argument makes it the only thing that holds the whole **answer**. The
    output arrives in two pieces from two machines -- the first token out of the
    prefill's last position, the remaining ``output_tokens - 1`` out of the decode
    batch -- and no host ever holds both. So the client concatenates them, and how
    many tokens the request produced becomes something the run *counted* rather
    than the ``output_tokens`` the workload asked for read back out of the
    request. The two agree today, and the point is that they are now two numbers
    that could disagree: one is what was asked for and one is what was made.

    Args:
        plane: the run's serving hosts as a caller reaches them, each call a charged
            round trip rather than a free method call.
        landed: which host a request arrives at.
        metrics: the run's ledger, written to exactly once per *completed*
            request and never read. Not a hop: the stamp is taken on this side of
            the wire and reporting it is the harness collecting what the client
            already knew, not a call the client makes into the cluster.
    """

    def __init__(
        self,
        plane: RoutedPlane,
        landed: Callable[[Request], str],
        metrics: Metrics,
    ) -> None:
        self.plane = plane
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
        """Take one turn through as many hosts as it is sent to.

        Returns when the turn is finished, which for a run that models decode
        means its last token has landed -- which is also what makes it safe for
        :meth:`submit` to treat this as "the user now has the answer". A turn that
        ended early ended, it was not abandoned: the host that ended it recorded why,
        and neither path leaves anything decoding behind this coroutine.
        """
        prefilled = await self.plane.prefill(request, at=self.landed(request))
        if prefilled is None or prefilled.decode is None:
            # Refused at the door, or a run that models no second half. Either way the
            # journey ended where a host ended it, and the stamp below belongs to the
            # requests that produced a last token.
            return
        # The two halves of the answer, from the two hosts that made them: the first
        # token out of the prefill's last position, the rest out of the decode batch.
        output = [
            prefilled.token,
            *await self.plane.decode(request, prefilled.response, at=prefilled.decode),
        ]
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
        # The simulation *is* the deployment (:class:`~proposed.deployment.Deployment`):
        # it vends the client for an instance, holds the directory, and carries the
        # control services a host reaches. So each host is handed it and takes
        # what it needs (:meth:`~kvcache_sim.data.serving.ServingHost.attach`) --
        # nothing here plumbs a port.
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
                instance,
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
            hosts[instance].attach(sim)
            # ...and fronted, so a caller off the box can reach it. There is no cycle
            # to close: a host knows nothing about any other, and what the service
            # refuses is precisely a host that did (``proposed.routed.peerless``).
            sim.front_plane(instance, hosts[instance])
        client = _Client(
            RoutedPlane(sim, ServingHost),
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
