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

Four things a deployment would not need are built here, because in a deployment
they are not built at all. One is
:class:`~kvcache_sim.workload._accelerator.SimulatedAccelerator`: what a forward
pass costs and how it is made to take that long is the run's answer under
simulation and the model's in production, so the capability is handed a port and
this supplies the implementation. One is the peer references a host reaches another host
through -- in production a handle to a remote actor, here a
:class:`~realsim.seams.link.LocalEndpoint` pair over a
:class:`~realsim.seams.link.ServiceHop`, so the boundary is charged. The other two
are the load balancer: :func:`_affinity` decides which host a request lands on and
:class:`_LoadBalancer` hands it there. Production has a client SDK, an ingress proxy
or DNS doing that, none of which is part of the serving system -- which is why they
are here rather than in ``data/``, whose test for membership is whether a thing
advances the clock or moves bytes, and whose contents are what would lift into a
deployment unchanged.

They are two functions because they are two services. The plane factory does not
build the scheduler; it takes ``sim.coordinator_handle``, the handle
:meth:`realsim.run.Run.execute` put in front of whatever :func:`coordinator`
returned. A scenario names both on a :class:`~realsim.run.Run`: same workload,
different wiring, which is exactly what "cache-aware vs load-balance" means.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from zlib import crc32

import torch

from domain import DEFAULT_MODEL, DEFAULT_PROFILE, Model
from proposed import Endpoint
from realsim.runner import ItemDispatch, WorkItem
from realsim.seams.link import LocalEndpoint, ServiceHop
from realsim.seams.transport import TensorDescriptor
from realsim.simulation import Simulation
from realsim.run import Workload
from sim_common import config

from ..control.request import Request
from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ._accelerator import SimulatedAccelerator
from ..data._decode import DecodeEngine
from ..data._prefill import PrefillEngine
from ..data.serving import ServingHost
from ..data.store import KVStore

#: Tokens per KV block. Fixed for every scenario so runs stay comparable.
BLOCK_TOKENS = 512

__all__ = ["BLOCK_TOKENS", "coordinator", "KVWorkload", "serving_plane"]


def _sim_block_carrier(
    block_tokens: int = BLOCK_TOKENS, model: Model = DEFAULT_MODEL
):
    """What one KV block is stored as **under simulation**.

    A metadata-only carrier: a uint8 descriptor whose length *is* the block's
    modeled byte size, so the bytes the transport charges cannot drift from the
    bytes the scheduler predicted. Zero real storage.

    This is the one piece a real deployment chooses differently -- it stores the
    KV tensors -- which is why it lives with the run rather than in
    :mod:`kvcache_sim.data.store`.
    """
    return TensorDescriptor(
        shape=(model.block_bytes(1, block_tokens),), dtype=torch.uint8
    )


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


class _LocalHostHandle:
    """A reference to another serving host, endpoint-shaped and hop-charged.

    What a host holds for a peer, standing in for Monarch's handle over that
    host's actor: one :class:`~realsim.seams.link.LocalEndpoint` per member a peer
    may call, over a shared :class:`~realsim.seams.link.ServiceHop`. Reaching
    another host is a boundary like reaching the directory or the coordinator, and
    is charged like one -- free by default (``host_rtt`` is ``0.0``), so a run that
    does not ask for the fidelity is byte-identical.

    Only the two members a *peer* calls are exposed. ``receive`` is not one of
    them: a request arrives from a client, and a host that could hand another host
    an unrouted request would be a second router.
    """

    def __init__(self, host: ServingHost, hop: ServiceHop) -> None:
        self.serve = LocalEndpoint(host.serve, hop)
        self.admit_decode = LocalEndpoint(host.admit_decode, hop)


def _affinity(ids: List[str]) -> Callable[[Request], str]:
    """Which host a request lands on: same conversation, same host.

    The load balancer a deployment has and a simulation has to stand in for. Client
    affinity rather than round robin because it is what a real front end does with
    a session id, and because it is the arrival policy that forwards least: a
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


class _LoadBalancer:
    """Which host a request lands on, and handing it there. Not a serving decision.

    Stands in for what a deployment already has in front of its hosts -- a client
    SDK doing client-side balancing, an ingress proxy, DNS -- and therefore for
    something that is deleted rather than moved when this ships. It holds no
    policy of its own beyond ``landed``, and it never asks where a request should
    *run*: that is the coordinator's answer, and the host it delivers to will go
    and get it.

    Args:
        hosts: ``instance id -> ServingHost``. Held as objects rather than
            references because this is outside every host -- the client side of
            the deployment -- and so pays no hop that every scheduler would not
            pay equally.
        landed: which host a request arrives at.
    """

    def __init__(
        self,
        hosts: Dict[str, ServingHost],
        landed: Callable[[Request], str],
    ) -> None:
        self.hosts = hosts
        self.landed = landed

    async def deliver(self, item) -> None:
        """Hand the request to whichever host it arrived at."""
        request: Request = item.payload
        await self.hosts[self.landed(request)].receive(request)

    async def drain(self) -> None:
        """Every host's decode has to finish before the run is over."""
        for instance in sorted(self.hosts):
            await self.hosts[instance].drain()


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
        # and holds the directory. All the run adds is the block carrier.
        store = KVStore(
            sim.mesh, block_tokens=BLOCK_TOKENS, carrier=_sim_block_carrier()
        )
        hop = ServiceHop(config.current().host_rtt)
        hosts: Dict[str, ServingHost] = {}
        handles: Dict[str, _LocalHostHandle] = {}

        def peers(instance: str) -> Any:
            return handles[instance]

        prefills = set(prefill_pool) if prefill_pool else set(sim.ids)
        decodes = (set(decode_pool) if decode_pool else set(sim.ids)) if simulate_decode else set()
        for instance in sorted(sim.ids):
            def accelerator() -> SimulatedAccelerator:
                return SimulatedAccelerator(profile=DEFAULT_PROFILE, model=DEFAULT_MODEL)

            compute = accelerator()
            # One accelerator when the run models the collision; two when it does
            # not, which is what ``coupled=False`` on a host in both pools means.
            prefill_compute = compute if coupled else accelerator()
            hosts[instance] = ServingHost(
                instance, store, sim.coordinator_handle,
                peers=peers,
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
        # After every host exists, because a handle names one: the lookup is
        # deferred through ``peers`` so the cycle is closed by the time anyone
        # calls, and no host holds another host's object.
        for instance, host in hosts.items():
            handles[instance] = _LocalHostHandle(host, hop)
        balancer = _LoadBalancer(hosts, _affinity(sorted(sim.ids)))
        # What the runner drives is a dispatcher, not a plane: the executing half
        # of this capability is the hosts, and there are several. The rows are
        # published at rejection, at acceptance, or when the last decode token
        # lands -- never one per item, so the harness must not write them.
        return ItemDispatch(
            balancer.deliver, on_drain=balancer.drain, writes_own_outcomes=True
        )

    return build
