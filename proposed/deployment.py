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

from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

__all__ = [
    "Controller", "Deployment", "Key", "Sensor", "StorageFull",
    "StorageVolume", "VolumeId",
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
    """Directory metadata, including pending publication state."""

    def _locate(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        *,
        prefer: Optional[Sequence[VolumeId]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Read live metadata synchronously, with an optional source preference."""

    async def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        prefer: Optional[Sequence[VolumeId]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Read live metadata through the controller endpoint."""

    async def notify_put_batch(
        self,
        requests: Sequence[Any],
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        """Declare or land one batch and return its publication id."""

    def _notify_put(
        self, request: Any, storage_volume_id: str, *, pending: bool = True
    ) -> int:
        """Declare or land one request synchronously."""

    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
        """Deregister one key from one volume."""

    async def notify_delete_batch(
        self,
        volume_to_keys: Optional[Dict[str, List[str]]] = None,
        *,
        pub: Optional[int] = None,
    ) -> None:
        """Deregister live rows or retire one publication."""

    def serving_union(
        self, requests: Sequence[Any]
    ) -> FrozenSet[Tuple[int, VolumeId]]:
        """Live and pending sources overlapping any requested region."""

    def greedy_cover(
        self,
        requests: Sequence[Any],
        ranked: Iterable[Tuple[int, VolumeId]],
    ) -> List[Tuple[int, VolumeId]]:
        """Greedy minimum cover over ranked publications."""

    async def keys(self, prefix: Optional[str] = None) -> List[str]:
        """Every registered key, or those under ``prefix``."""


class Sensor(ABC):
    """Facts a capability holds between calls, and its decisions read.

    A base a sensor derives, where the services in this module are
    :class:`Protocol`s: a protocol is for a surface reached across a boundary, of which
    a caller has the shape and not the object, while this is what a capability's own
    object declares itself to be -- and a member-less protocol would declare nothing,
    every object satisfying it structurally.

    The peer of :class:`Controller` on the application's side. That one holds
    residency -- which volume holds which key -- and is written as volumes publish
    and evict; a sensor holds what no directory can see: how deep an instance's
    queue is, who is working and until when, what has been promised and not yet
    done.

    Sensor reads are the application's own vocabulary: a queue tail and a decode
    occupancy belong to whoever decides against them. A selector declares the sensor
    types it reads. :attr:`folds` is empty until a sensor accepts reported actions.

    Nothing reaches one from outside the process that holds it. A fact a host
    reports is an action, dispatched into the one
    :class:`proposed.dispatch.Dispatcher` a run fronts, which folds it by calling the
    reducer a sensor declares (:class:`proposed.dispatch.Reducer`) -- so a sensor is
    written by the folds it publishes and read by the decisions above it, and neither
    is a surface.
    """

    folds: Mapping[type, Callable[[Any], None]] = {}


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
        self,
        node_id: str,
        *,
        prefer: Optional[Sequence[VolumeId]] = None,
    ) -> Any:
        """The torchstore client for ``node_id``, ready to be driven.

        A deployment that runs one node per process ignores the id and returns its
        own client. A harness running many nodes in one process resolves the node
        and attributes the work to it.

        ``prefer`` is the source preference the reads made through this client apply
        (:func:`proposed.selector.prefer`) -- volume ids, best first, typically what
        a data plane just got back from :attr:`control_plane_handle`. ``None`` is no
        preference: the read is exactly the ordinary one.

        Upstream it belongs on the client's own read members, ``get`` / ``get_batch``.
        It is a keyword on the member that already binds the caller's identity because
        the real client has no such parameter yet, and threading it through every call
        site of a client this object vends would be the same edit twice.
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

    def plane_handle(self, node_id: str) -> Any:
        """A reference to ``node_id``'s **data plane**: the calls a caller makes.

        The sibling of :attr:`control_plane_handle`, and untyped for its reason: what
        a caller holds is a reference, one endpoint per member the plane declares, and
        Monarch declares that shape. Read the argument lists off the plane.

        Here for whoever follows an address a plane answered with -- a caller, never
        another host (:func:`proposed.routed.routed`,
        :class:`proposed.routed.RoutedPlane`).
        A plane that redirects nobody needs nothing here.
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
    def dispatcher_handle(self) -> Any:
        """A reference to where this application's hosts report their facts.

        The other half of what a host says to control -- a question goes to
        :attr:`control_plane_handle`, a fact goes here as an action
        (:class:`proposed.dispatch.Dispatcher`, which folds it into every sensor it
        moves and commits them together). ``None`` when nothing outside the control
        plane's own process writes anything.
        """
        ...
