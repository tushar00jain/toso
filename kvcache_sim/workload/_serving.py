"""The kvcache workload, and the wiring a run installs around it.

Two separate things, deliberately:

* :class:`KVWorkload` is *the work* -- a request stream, one
  :class:`~realsim.runner.WorkItem` per request at its arrival time. It builds no
  store, no scheduler and no plane;
* :func:`coordinator` and :func:`serving_plane` are the *capability wiring*, one
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

They are two functions because they are two services. The plane factory does not
build the scheduler; it takes ``sim.coordinator_handle``, the handle
:meth:`realsim.run.Run.execute` put in front of whatever :func:`coordinator`
returned. A scenario names both on a :class:`~realsim.run.Run`: same workload,
different wiring, which is exactly what "cache-aware vs load-balance" means.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Optional
from zlib import crc32

from domain import DEFAULT_MODEL, DEFAULT_PROFILE
from proposed import Endpoint
from realsim.runner import ItemDispatch, WorkItem
from realsim.seams.link import LocalEndpoint, ServiceHop
from realsim.simulation import Simulation
from realsim.run import Workload
from sim_common import config

from ..control.request import Request
from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ._accelerator import BLOCK_TOKENS, SimulatedAccelerator
from ..data._decode import DecodeEngine
from ..data._prefill import PrefillEngine
from ..data.serving import ServingHost
from ..data.store import KVStore

# ``BLOCK_TOKENS`` is imported rather than declared: how much of a prompt one KV
# block covers is the engine's cache-page size, so it lives with the accelerator
# that lays the KV out (:mod:`kvcache_sim.workload._accelerator`). It is re-exported
# here because the *scheduler* has to be told the same number -- it prices a prefix
# match in blocks -- and a run that told the two different numbers would route on
# one geometry and store in another.

__all__ = ["BLOCK_TOKENS", "coordinator", "KVWorkload", "serving_plane"]


class KVWorkload(Workload):
    """A request stream over a set of serving instances.

    One work item per request, released at its arrival time. Which scheduler
    serves them is not this object's business -- see :func:`serving_plane`.
    """

    def __init__(self, topology: Dict[str, Endpoint], requests) -> None:
        super().__init__(topology)
        self.requests = requests

    def items(self, sim: Simulation) -> List[WorkItem]:
        """One item per request; the serving plane runs each one's lifecycle."""
        return [
            WorkItem(id=r.id, release_time=r.arrival, payload=r)
            for r in self.requests
        ]


def coordinator(
    kind: str,
    *,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    max_batch: int = 8,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "off",
) -> object:
    """This run's **control plane**, as an object a scenario can just declare.

    ``kind`` is ``"cache_aware"`` (the coordinator under test) or
    ``"load_balance"`` (the baseline). Knobs only: the stack's ports arrive later
    through :meth:`~kvcache_sim.control.scheduler._Base.attach`, which is what
    lets this be a value rather than a factory the harness must call at the right
    moment. The :class:`~realsim.run.Run` installs it in the directory (it answers
    the store's routing question) *and* fronts it with a
    :class:`~realsim.seams.coordinator_handle.LocalCoordinatorHandle` (it decides compute
    placement) -- one object, reached through the seam of whichever service is
    asking.
    """
    if kind not in ("cache_aware", "load_balance"):
        raise ValueError(f"unknown scheduler kind {kind!r}")
    knobs = dict(
        block_tokens=BLOCK_TOKENS,
        profile=DEFAULT_PROFILE,
        slo_ttft=slo_ttft,
        slo_tbt=slo_tbt,
        simulate_decode=simulate_decode,
        max_batch=max_batch,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        early_rejection=early_rejection,
    )
    if kind == "cache_aware":
        return CacheAwareScheduler(
            balance_threshold=balance_threshold, replicate=replicate, **knobs
        )
    return LoadBalanceScheduler(**knobs)


class _ServingEndpoints:
    """One serving host as a **client** reaches it: endpoint-shaped, hop-charged.

    Standing in for whatever a client SDK holds for a serving instance -- a
    connection, a stub, a Monarch handle over its actor -- as one
    :class:`~realsim.seams.link.LocalEndpoint` per member, over a shared
    :class:`~realsim.seams.link.ServiceHop`. A client is off the box, so reaching a
    host is a boundary like reaching the directory or the coordinator and is
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
    a session id, and because it is the arrival policy redirected least: a
    conversation's requests share a prefix, so the host that served the last one is
    usually the host that should serve the next.

    Note what it deliberately does *not* do: it never looks at the block keys. An
    arrival policy that routed by cache contents would be doing the coordinator's
    job with none of the coordinator's information, and the comparison this whole
    package exists to make would be measuring itself.

    Deterministic across runs and platforms: ``crc32`` of the conversation id, not
    Python's salted ``hash``.
    """
    ordered = sorted(ids)

    def landed(request: Request) -> str:
        return ordered[crc32(request.conversation.encode()) % len(ordered)]

    return landed


class _Client:
    """The thing outside the cluster that submits a request and follows redirects.

    Stands in for what a deployment already has in front of its hosts -- a client
    SDK doing client-side balancing, an ingress proxy, DNS -- and therefore for
    something that is deleted rather than moved when this ships. It holds no policy
    of its own beyond ``landed``, and it never asks where a request should *run*:
    it asks a host, and does what it is told.

    Three legs, which is the whole object::

        plan   = await hosts[landed(request)].route(request)   # "prefill is B"
        decode = await hosts[plan.prefill].prefill(plan)       # "decode is C"
        await hosts[decode].decode(plan)

    Note what it carries and what it does not. It carries the
    :class:`~kvcache_sim.control.scheduler.Plan` -- a value the coordinator issued,
    which is exactly the kind of thing a client is handed and hands back (a routing
    token, a session ticket) and which nothing here reads for a decision. It does
    *not* carry a measurement row: each host records its own half into the run's
    ledger, and the client never learns the outcome, because a client that had to
    be told the hit rate would be part of the serving system.

    It also does not second-guess an address. ``prefill`` answers with the decode
    host rather than the client reading ``plan.decode``, because the plan is what
    control *predicted* and the prefill host may have been refused the admission
    since -- ``None`` means the journey ends here, and the host that ended it has
    already said why in the ledger.

    Args:
        hosts: ``instance id -> _ServingEndpoints``. References, not objects: a
            client is off the box, so each of the three legs is a charged round
            trip rather than a free method call.
        landed: which host a request arrives at.
        drains: ``instance id -> ServingHost.drain``, the one thing that is not a
            request. Draining is the harness waiting for the simulated cluster to
            go quiet, not a client operation, so it is not on the endpoints and
            pays no hop: a real client simply closes its connection and leaves the
            hosts to finish.
    """

    def __init__(
        self,
        hosts: Dict[str, _ServingEndpoints],
        landed: Callable[[Request], str],
        drains: Dict[str, Callable[[], Awaitable[None]]],
    ) -> None:
        self.hosts = hosts
        self.landed = landed
        self.drains = drains

    async def submit(self, item) -> None:
        """Take one request through as many hosts as it is redirected to."""
        request: Request = item.payload
        plan = await self.hosts[self.landed(request)].route.call_one(request)
        if plan is None:
            return  # refused at the door; the host that refused recorded it
        decode = await self.hosts[plan.prefill].prefill.call_one(plan)
        if decode is None:
            return  # nothing after prefill: no decode modelled, or it was shed
        await self.hosts[decode].decode.call_one(plan)

    async def drain(self) -> None:
        """Every host's decode has to finish before the run is over."""
        for instance in sorted(self.drains):
            await self.drains[instance]()


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
                instance, store, sim.coordinator_handle,
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
            {i: h.drain for i, h in sorted(hosts.items())},
        )
        # What the runner drives is a dispatcher, not a plane: the executing half
        # of this capability is the hosts, and there are several. The rows are
        # published at rejection, at acceptance, or when the last decode token
        # lands -- never one per item, so the harness must not write them.
        return ItemDispatch(
            client.submit, on_drain=client.drain, writes_own_outcomes=True
        )

    return build
