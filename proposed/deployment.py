"""How application code reaches the store it is deployed against.

A capability's data plane calls ordinary torchstore APIs, but it has to get the
client from somewhere, and in a simulation "somewhere" is a harness object holding
many clients at once. Depending on that harness directly would make the data plane
unliftable: real code cannot import the simulator.

So the data plane asks for a :class:`Deployment` instead. In production this is
one process with one client and one controller; in the simulator it is the mesh,
which resolves the node and does the bookkeeping a real deployment would not need.
Either way the application code is the same.

The *payload* types stay untyped here -- a ``StorageInfo``, a ``Request``, a
tensor are torchstore's, and this package cannot import torchstore, being what
torchstore would gain rather than something layered on top of it. The **surface**
is not payload, so it is declared: :class:`Controller` says which methods the
directory service offers and with what arguments. What a caller *holds* is a
reference to that service rather than the service itself -- a different shape,
declared by Monarch, and left untyped here for the reason given on
:attr:`Deployment.controller_handle`. Leaving that as ``Any``
made the directory look like it had no interface at all, which in turn made
:class:`~proposed.policy.Policy` look like the primary one instead of a hook
consulted inside this surface.

The endpoint indirection a caller goes through (``locate_volumes.call_one(...)``
rather than ``locate_volumes(...)``) is Monarch's own, and Monarch declares it --
``Endpoint[P, R]`` with ``call_one`` / ``call``. It is not restated here: this
surface declares the methods, which is where the signatures are.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

__all__ = [
    "Controller", "Coordinator", "Deployment", "StorageFull", "StorageVolume",
]


class StorageFull(Exception):
    """A volume refused a write because it has no room for it.

    Part of the storage surface, not of a simulator: a bounded volume can always
    be asked for more than it has, and a caller has to be able to tell "there was
    no room" from a bug. Declared here so a data plane can catch it without
    importing whatever is enforcing the bound.

    It is raised only when the volume has already asked what to drop
    (:meth:`proposed.policy.Policy.evict`) and still cannot fit the write -- so
    catching it means *this data does not fit anywhere on this volume*, not
    *try again later*. A cache fill answers that by not caching; a durable write
    has to fail.
    """


class Controller(Protocol):
    """The directory service, as a caller reaches it.

    Named for what it replaces. torchstore has no type for this today, but it has
    a *name*: ``api.py``'s ``_controller() -> Controller`` annotates the spawned
    handle as the actor class, and ``spmd.py`` does the same, while ``LocalClient``
    takes it unannotated altogether. So this is that type, written down.

    Declared as **methods**, matching the actor class, because that is where the
    signatures live. A caller does not invoke them directly: Monarch's
    ``@endpoint`` turns each into an ``EndpointProperty``, and a handle exposes it
    as an ``Endpoint[P, R]`` reached through ``call_one`` (one actor) or ``call``
    (a mesh) -- so the call site reads
    ``controller.locate_volumes.call_one(keys, missing_ok=True)``. That
    indirection is Monarch's own type (``monarch._src.actor.endpoint.Endpoint``)
    and is not restated here; annotating the handle with the actor's surface is
    the same convention torchstore already uses.

    The difference between this protocol and torchstore's class *is* the ask, and
    it is two things. One cannot be declared: ``locate_volumes`` gains a hook that
    consults a :class:`~proposed.policy.Policy`, which changes no signature, so it
    is stated here instead. The other can be, and is: :meth:`locate_raw`, the same
    directory read with that hook skipped. Every other member below already exists
    upstream, spelled the same way.

    A caller does not hold one of these -- it holds a :class:`ControllerHandle`.
    torchstore's ``Controller`` implements this; under simulation the two bodies
    Monarch will not let us invoke off-actor are mirrored privately inside
    :class:`realsim.seams.controller_handle.LocalControllerHandle`.
    """

    async def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """``{key -> {volume_id -> StorageInfo}}``, routed."""

    async def locate_raw(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """The same read, *unrouted*: no policy consulted.

        A controller implementing this proposal needs both reads. This is the one it
        hands its own policy, through a :class:`~proposed.view.View`: a policy
        sensing the directory must see it as it *is*, and reading it back through
        ``locate_volumes`` would re-enter the hook the policy is being called from.

        It is on this surface and not a protocol of its own because the object that
        has it is the object that answers ``locate_volumes`` -- one directory
        service, read two ways. A caller that is not the controller's own policy has
        no reason to reach for it.
        """

    async def evict_for(
        self, storage_volume_id: str, need_bytes: int
    ) -> Sequence[str]:
        """``storage_volume_id`` is out of room: which keys should it drop?

        The second thing this surface would have to gain, and the second place it
        consults a :class:`~proposed.policy.Policy` -- the first being the routing
        hook inside ``locate_volumes``. A storage volume knows it is full and knows
        nothing about what is worth keeping; the directory knows who holds what and is
        where a policy is installed, so the volume asks *here* rather than holding a
        control plane of its own.

        Answers nothing when no policy is installed, which leaves the volume to
        refuse the put as an unbounded store always did.
        """

    async def notify_put_batch(
        self, requests: Sequence[Any], storage_volume_id: str
    ) -> None:
        """Register that ``storage_volume_id`` now holds the keys in ``requests``."""

    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
        """Deregister one key from one volume."""

    async def notify_delete_batch(
        self, volume_to_keys: Dict[str, List[str]]
    ) -> None:
        """Deregister ``{volume_id -> keys}``, idempotently."""

    async def keys(self, prefix: Optional[str] = None) -> List[str]:
        """Every registered key, or those under ``prefix``."""


class StorageVolume(Protocol):
    """The store's *storage* service, as a caller reaches it.

    The third service in a deployment, beside :class:`Controller` (which knows who
    holds what) and :class:`Coordinator` (which decides): the one that actually holds
    bytes. torchstore has this class -- ``torchstore.storage_volume.StorageVolume``,
    an actor whose endpoints each delegate to an ``InMemoryStore`` -- and, as with
    ``Controller``, never declares the surface a caller depends on. So this is that
    surface, written down.

    Declared as **methods**, matching the actor class, for the reason given on
    :class:`Controller`: that is where the signatures live. A caller reaches each one
    through an endpoint (``volume.put.call_one(buffer, requests)``), which is
    Monarch's own type and is not restated here.

    Every member below already exists upstream. Two of its endpoints are deliberately
    left out -- ``get_meta`` and ``get_id`` -- because nothing in this proposal
    reaches for them; they are the store's own bookkeeping, and declaring a surface
    wider than the ask would misstate the ask.

    What a *deployment* would have to gain is nothing here: unlike ``Controller``,
    this surface is complete as it stands. What the simulator adds around it --
    per-volume residency, a byte capacity, and asking
    :meth:`proposed.policy.Policy.evict` before refusing a put that does not fit --
    is behaviour behind these same signatures, which is why it needs no member of its
    own. A real volume enforcing a real disk would answer the same way.
    """

    async def put(self, transport_buffer: Any, requests: Sequence[Any]) -> None:
        """Store what ``requests`` name, reading bytes through ``transport_buffer``.

        Raises :class:`StorageFull` if the write does not fit and could not be made
        to fit; nothing lands in that case.
        """

    async def get(self, transport_buffer: Any, requests: Sequence[Any]) -> Any:
        """Fill ``transport_buffer`` with what ``requests`` name, and return it."""

    async def handshake(
        self, transport_buffer: Any, requests: Sequence[Any]
    ) -> List[Any]:
        """Agree the transport for a transfer before either side moves bytes."""

    async def delete(self, key: str) -> None:
        """Drop one key's bytes."""

    async def delete_batch(self, keys: List[str]) -> None:
        """Drop several keys' bytes."""

    async def reset(self) -> None:
        """Drop everything this volume holds."""


class Coordinator(Protocol):
    """A control plane that runs as its own service, as a caller reaches it.

    The mirror of :class:`Controller`, for the other service a caller talks to.
    Where that one is a directory torchstore already has, this one is a coordinator
    an application spawns: it holds the cluster-wide picture a single host cannot --
    every instance's queue, cache and load -- and serializes the decisions that read
    it.

    Every member is ``async``, like :class:`Controller`'s -- a service handles a
    message, and whether the *sender* waits for the answer is the sender's choice
    (``call_one`` versus ``broadcast``), not something the surface decides.

    Declared as methods, like :class:`Controller`, because that is where the
    signatures live. A caller holds a reference rather than the object, so the call
    goes through an endpoint (``decide.call_one(demand)``,
    ``observe.broadcast(fact)``); upstream that reference is Monarch's handle over an
    ``Actor`` carrying one ``@endpoint`` per member below, each forwarding to the
    plain object -- decorate the shim, not the deciding logic, or its members become
    ``EndpointProperty`` descriptors that cannot be invoked off-actor. That tax is on
    display in :mod:`realsim.seams.controller_service`, which exists because
    torchstore decorated ``Controller``'s own methods.

    Two members, matching :class:`proposed.coordinator.Coordinator` exactly. It used
    to name one member per question a KV-cache scheduler asks -- ``schedule``,
    ``decode_admission``, three ``observe_*`` -- which put one application's
    vocabulary in the store's contract, and in the seam that carries it, and left the
    author's half and this half to be kept in step by hand. With the questions moved
    into the *payload*, there is nothing application-specific in either half to
    drift, and a second application reaches its coordinator across the same two
    endpoints without a line changing here or in the seam.

    The payloads are ``Any`` for the reason given at the top of this module: a
    demand, an answer and a fact are the application's types, and this package cannot
    import an application any more than it can import torchstore.
    """

    async def decide(self, demand: Any) -> Optional[Any]:
        """Ask the control plane to answer ``demand``; ``None`` is a refusal."""

    async def observe(self, fact: Any) -> None:
        """Report ``fact``. The reply carries nothing; ``broadcast`` and do not wait."""


class Deployment(Protocol):
    """The store, as application code sees it."""

    def client_for(self, node_id: str) -> Any:
        """The torchstore client for ``node_id``, ready to be driven.

        A deployment that runs one node per process ignores the argument and
        returns its own client. A harness running many nodes in one process
        resolves the node and attributes the work to it.
        """
        ...

    @property
    def controller_handle(self) -> Any:
        """A reference to the directory service: the calls a caller makes.

        Untyped on purpose. What a caller holds is not a :class:`Controller` but a
        *reference* to one, and Monarch already declares that shape --
        ``@endpoint`` makes each method an ``Endpoint`` reached through ``call_one``
        or ``call``, so the call reads
        ``controller_handle.locate_volumes.call_one(keys, missing_ok=True)``.
        Restating it here would either lose the keyword names a caller uses or
        duplicate a generic this package cannot import. Read the argument lists off
        :class:`Controller`, which is the surface behind the reference: in
        production Monarch's handle over the ``Controller`` actor, under simulation
        :class:`realsim.seams.controller_handle.LocalControllerHandle` over a
        :class:`realsim.seams.controller_service.ControllerService`.
        """
        ...
