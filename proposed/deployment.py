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

__all__ = ["Controller", "Coordinator", "Deployment"]


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
    goes through an endpoint (``schedule.call_one(request)``,
    ``observe_decode_state.broadcast(...)``); upstream that reference is Monarch's
    handle over an ``Actor`` carrying one ``@endpoint`` per member below, each
    forwarding to the plain object -- decorate the shim, not the deciding logic, or
    its members become ``EndpointProperty`` descriptors that cannot be invoked
    off-actor. That tax is on display in
    :mod:`realsim.seams.controller_service`, which exists because torchstore
    decorated ``Controller``'s own methods.

    The payloads are ``Any`` for the reason given at the top of this module: a
    request, a plan and a completion are the application's types, and this package
    cannot import an application any more than it can import torchstore.
    """

    async def schedule(self, request: Any) -> Optional[Any]:
        """Decide what to do with ``request``; ``None`` rejects it."""

    async def complete(self, plan: Any) -> Any:
        """What the executing half must do once it has carried ``plan`` out."""

    async def decode_admission(self, plan: Any) -> bool:
        """Whether the accepted ``plan`` may enter the stage it is queued for."""

    async def observe_prefill_done(self, inst: str, now: float) -> float:
        """Report the clock the real work reached; answer with the corrected model."""

    async def observe_compute_busy(self, inst: str, until: float) -> None:
        """Report a resource occupied until ``until``."""

    async def observe_decode_state(
        self, inst: str, finishes: Sequence[float]
    ) -> None:
        """Report what is still running on ``inst``, as finish estimates."""


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
