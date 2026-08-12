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
they are not built at all. One is the peer references a host reaches another host
through -- in production a handle to a remote actor, here a
:class:`~realsim.seams.link.LocalEndpoint` pair over a
:class:`~realsim.seams.link.ServiceHop`, so the boundary is charged. The other two
are the load balancer: :func:`_affinity` decides which host a request lands on and
:class:`_Arrivals` delivers it there. Production has a client SDK, an ingress proxy
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
from proposed import DataPlane, Endpoint
from realsim.runner import WorkItem
from realsim.seams.link import LocalEndpoint, ServiceHop
from realsim.seams.transport import TensorDescriptor
from realsim.simulation import Simulation
from realsim.run import Workload
from sim_common import config

from ..control.request import Request
from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
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


class _Arrivals(DataPlane):
    """Deliver each request to the host it lands on. That is the whole job.

    The harness-facing :class:`~proposed.plane.DataPlane`, and deliberately almost
    nothing: *where a request arrives* is a load balancer's answer -- DNS, a
    client's affinity, a round robin -- and *where it should run* is the
    coordinator's. Neither is a serving decision, which is why this is wiring and
    not part of the capability: a deployment deletes it and keeps the hosts.

    Args:
        hosts: ``instance id -> ServingHost``. Held as objects rather than
            references because this stands in for the client side of the
            deployment, which is outside every host -- and so pays no hop that
            every scheduler would not pay equally.
        arrival_host: which host a request lands on.
    """

    #: Rows are published at rejection, at acceptance, or when the last decode
    #: token lands -- never one per item, so the harness must not write them.
    writes_own_outcomes = True

    def __init__(
        self,
        hosts: Dict[str, ServingHost],
        arrival_host: Callable[[Request], str],
    ) -> None:
        self.hosts = hosts
        self.arrival_host = arrival_host

    async def execute(self, item) -> None:
        """Hand the request to whichever host it arrived at."""
        request: Request = item.payload
        await self.hosts[self.arrival_host(request)].receive(request)

    async def drain(self) -> None:
        """Every host's decode has to finish before the run is over."""
        for host in sorted(self.hosts):
            await self.hosts[host].drain()


def serving_plane(
    *,
    coupled: bool = False,
    simulate_decode: bool = False,
    max_batch: int = 8,
    decode_pool: Optional[List[str]] = None,
) -> Callable[[Simulation], "_Arrivals"]:
    """Build the factory for this run's **data plane**: one host per instance.

    ``coupled`` says whether prefill shares a host's decode compute -- a deployment
    fact, so it belongs to the host, not to the scheduler. The decode settings are
    passed here *and* to :func:`coordinator` from the one scenario that declares
    them, rather than one reading them off the other.

    ``decode_pool`` needs no telling now: a host that is never named as a plan's
    decode instance is never handed one, so only the coordinator has to know which
    hosts decode.
    """

    def build(sim: Simulation) -> "_Arrivals":
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

        for instance in sorted(sim.ids):
            hosts[instance] = ServingHost(
                instance, store, sim.coordinator_handle,
                peers=peers,
                trace=sim.trace, metrics=sim.ledger,
                coupled=coupled,
                simulate_decode=simulate_decode,
                max_batch=max_batch,
                profile=DEFAULT_PROFILE,
                model=DEFAULT_MODEL,
            )
        # After every host exists, because a handle names one: the lookup is
        # deferred through ``peers`` so the cycle is closed by the time anyone
        # calls, and no host holds another host's object.
        for instance, host in hosts.items():
            handles[instance] = _LocalHostHandle(host, hop)
        return _Arrivals(hosts, _affinity(sorted(sim.ids)))

    return build
