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
:attr:`Deployment.controller_handle`.

The endpoint indirection a caller goes through (``locate_volumes.call_one(...)``
rather than ``locate_volumes(...)``) is Monarch's own, and Monarch declares it --
``Endpoint[P, R]`` with ``call_one`` / ``call``. It is not restated here: this
surface declares the methods, which is where the signatures are.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

__all__ = [
    "ClusterModel", "Controller", "Deployment", "Key",
    "StorageFull", "StorageVolume", "VolumeId",
]

#: What the store is asked *for*: the directory's own noun, and the subject a
#: :class:`~proposed.selector.KeySelector` declares (``subject_type``).
Key = str

#: What the store answers *with*: a storage volume's directory identity, and what
#: every :class:`~proposed.selector.Selection` ranks, whatever its subject.
VolumeId = str


class StorageFull(Exception):
    """A volume refused a write because it has no room for it.

    Part of the storage surface, not of a simulator: a bounded volume can always
    be asked for more than it has, and a caller has to be able to tell "there was
    no room" from a bug. Declared here so a data plane can catch it without
    importing whatever is enforcing the bound.

    It is raised only when the volume has already made what room it could and
    still cannot fit the write -- so
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
    it is two things. One is :meth:`locate_raw`, a directory read with nothing
    applied to it, which is what a control plane senses through. The other cannot
    be declared as a member: ``locate_volumes`` gains an optional **source
    preference** -- a list of volume ids its caller hands it -- and applies it to
    the answer (:func:`proposed.selector.prefer`) before returning. The store
    consults nobody to do that; it reorders a value it was given, which is why
    nothing here declares a plane. Every other member already exists upstream,
    spelled the same way.

    A caller does not hold one of these -- it holds a reference
    (:attr:`Deployment.controller_handle`). torchstore's ``Controller`` implements
    this; under simulation the endpoint bodies Monarch will not let us invoke
    off-actor are mirrored inside
    :class:`realsim.seams.controller_service.ControllerService`.
    """

    def locate_raw(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """``{key -> {volume_id -> StorageInfo}}``, with nothing applied to it.

        A controller implementing this proposal needs both reads. This is the one a
        control plane senses through, via a :class:`~proposed.view.View`: what it
        sees has to be the directory as it *is*, not an answer already reordered
        for somebody.

        **Not a coroutine, and that is load-bearing.** A directory read cannot
        suspend, so everything a control plane does between reading the directory
        and writing its own bookkeeping runs to completion before the next
        requester's does -- which is what lets a routing decision be a
        read-modify-write with no lock (``dedup_sim.control.routing``) and lets a
        set of priced candidates be comparable
        (``kvcache_sim.control.scheduler``). It is a plain local method for the same
        reason it can be one: the caller is a control plane sensing through a view
        built over the directory in its own process, so nothing crosses a boundary
        and there is nothing to wait for.
        """

    async def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """``{key -> {volume_id -> StorageInfo}}``, the caller's preference applied.

        That preference is the ask (see the class docstring): upstream, one optional
        parameter on the read path, applied to the located map before the client
        picks a volume per key. The simulator carries it in a coroutine binding
        instead, because the real client has no such parameter yet
        (:func:`realsim.seams.factory.bind_prefer`).

        A store that is handed no preference answers exactly as it does today, and
        one that is handed a preference still consults nothing: whoever asked a
        control plane did so itself, before calling this.
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


class ClusterModel(Protocol):
    """An application's picture of its own cluster, as a caller reaches it.

    The peer of :class:`Controller` on the application's side. That one holds
    residency -- which volume holds which key -- and is written as volumes publish
    and evict; this one holds what no directory can see: how deep an instance's
    queue is, who is working and until when, what has been promised and not yet
    done. It is written as the hosts report what they did.

    One member, because being told is the only thing a *caller* does to a model.
    What decides against it is the application's own control plane, and what it
    may show is read off the model itself: the reads are the application's, since a
    queue tail and a decode occupancy are one application's vocabulary, and
    :class:`~proposed.view.View` is the *store's* sensor, which cannot see load at
    all.

    The fact is ``Any`` for the reason given at the top of this module: this
    package cannot import an application, so what a host reports is the
    application's own type.
    """

    async def notify(self, fact: Any) -> None:
        """Fold ``fact`` in. The reply carries nothing.

        Awaited, like :meth:`Controller.notify_put_batch` and for the same reason:
        a reporter whose next question must be decided against this fact gets that
        ordering from the reply. Sending it one-way would order it only at the
        sender, and over any distance at all the question would arrive first.
        """


class StorageVolume(Protocol):
    """The store's *storage* service, as a caller reaches it.

    The third service in a deployment, beside :class:`Controller` (which knows who
    holds what) and the application's own control plane
    (which decides): the one that actually holds
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
    per-volume residency, a byte capacity, and evicting its coldest before refusing
    a put that does not fit -- is behaviour behind these same signatures, which is
    why it needs no member of its own beyond :meth:`touch`. A real volume enforcing a
    real disk would answer the same way.
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

    async def touch(self, keys: List[str]) -> None:
        """Report a read of ``keys`` that did not come through this volume.

        The one member here torchstore does not have, and the only thing the store
        needs in order to evict well: a volume can only see the accesses that reach
        it, so a caller reading data it already holds -- a serving instance reusing
        its own KV -- leaves no trace, and the volume drops exactly the blocks being
        reused. Nothing moves and nothing is charged; this is recency, not a read.

        A deployment where every hit goes through the store does not need it. This
        exists because that is not the only shape a cache has.
        """

    async def delete(self, key: str) -> None:
        """Drop one key's bytes."""

    async def delete_batch(self, keys: List[str]) -> None:
        """Drop several keys' bytes."""

    async def reset(self) -> None:
        """Drop everything this volume holds."""


class Deployment(Protocol):
    """The store and the control plane, as application code sees them.

    What a :meth:`~proposed.plane.DataPlane.attach` is handed: everything an
    executing half reaches that is not its own. A capability's data plane is written
    against this and against nothing a harness owns, which is what lets the same
    plane run under the simulator and over a real deployment.
    """

    def client_for(
        self, node_id: str, *, prefer: Optional[Sequence[VolumeId]] = None
    ) -> Any:
        """The torchstore client for ``node_id``, ready to be driven.

        A deployment that runs one node per process ignores the id and returns its
        own client. A harness running many nodes in one process resolves the node
        and attributes the work to it.

        ``prefer`` is the source preference the reads made through this client apply
        (:func:`proposed.selector.prefer`) -- volume ids, best first, typically what
        a data plane just got back from :attr:`control_plane_handle`. ``None`` is no
        preference: the read is exactly the ordinary one. Upstream this belongs on
        ``get`` / ``get_batch`` rather than here; it is a keyword on the member that
        already binds the caller's identity because the real client has no such
        parameter yet, and threading it through every call site of a client this
        object vends would be the same edit twice.
        """
        ...

    def volume_handle(self, node_id: str) -> Any:
        """A reference to ``node_id``'s storage volume: the calls a caller makes.

        Untyped for the same reason as :attr:`controller_handle` -- what a caller
        holds is a reference, whose shape Monarch declares. Read the argument lists
        off :class:`StorageVolume`.

        Here so a data plane can tell its *own* volume something the store cannot
        observe (``touch``). Reaching another node's volume through this would be
        going around the client, which is what the client is for.
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

    @property
    def control_plane_handle(self) -> Any:
        """A reference to this capability's control plane: everything it may be asked.

        One plane, and one endpoint per member *it* declares
        (:class:`~proposed.plane.ControlPlane`), because what a capability answers is
        the capability's to name. Which volumes should serve this read, where this
        request should run, and whatever a capability written next needs are members
        on the same reference:

            selection = await control_plane_handle.sources.call_one(keys, me)
            await client_for(me, prefer=selection.sources).get_batch(keys)

        Including whatever member it wants a *write* reported over -- a plane that
        withholds an answer until a peer holds a key has to hear that the put landed,
        and the caller that made the put is what tells it.

        Untyped for :attr:`controller_handle`'s reason: a reference, whose shape
        Monarch declares. Here rather than handed to a data plane separately -- a
        host reaching its control plane is reaching *this deployment's* control plane.
        ``None`` when the deployment stands up no such service, which is a run that
        decides nothing.
        """
        ...

    @property
    def cluster_handle(self) -> Any:
        """A reference to the model this application's hosts report into.

        The other half of what a host says to control -- a question goes to
        :attr:`control_plane_handle`, a fact goes here
        (:class:`ClusterModel`). ``None`` when the control plane keeps no model.
        """
        ...
