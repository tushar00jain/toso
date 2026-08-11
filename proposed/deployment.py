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
is not payload, so it is declared: :class:`Controller` says which endpoints the
directory service offers and with what arguments, and :class:`ControllerHandle` is
what a caller actually holds -- an endpoint per method, which is a different shape
and a different type. Leaving that as ``Any``
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

__all__ = ["Controller", "ActorEndpoint", "ControllerHandle", "Deployment"]


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

    Every member below already exists on the real ``Controller``. What this surface
    would have to *gain* is one thing, deliberately not listed: the hook that
    consults a :class:`~proposed.policy.Policy` inside ``locate_volumes``. The
    difference between this protocol and torchstore's class is the ask.

    A caller does not hold one of these -- it holds a :class:`ControllerHandle`.
    torchstore's ``Controller`` implements this; under simulation the two bodies
    Monarch will not let us invoke off-actor are mirrored privately inside
    :class:`realsim.seams.controller_handle.FakeControllerHandle`.
    """

    async def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """``{key -> {volume_id -> StorageInfo}}``, routed."""

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


class ActorEndpoint(Protocol):
    """One method of a service, as a caller reaches it.

    Monarch declares this (``monarch._src.actor.endpoint.Endpoint``, generic over a
    ``ParamSpec``); it is restated here only because this package cannot import the
    runtime it proposes against. Deliberately *not* generic: hand-written parameter
    lists would lose the keyword names a caller actually uses, and the signatures
    already live on :class:`Controller`, which is the honest place to read them.
    """

    async def call_one(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke it on a single actor."""

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke it on every actor in a mesh."""


class ControllerHandle(Protocol):
    """What a caller *holds*: an endpoint per method of :class:`Controller`.

    The two are different shapes, and both are real. ``@endpoint`` turns each
    method of the service into an ``EndpointProperty``, so a handle offers
    ``locate_volumes`` as an object and the call reads
    ``locate_volumes.call_one(keys, missing_ok=True)`` -- which is what real
    ``LocalClient`` code does, so nothing may collapse it back into a method.

    This is the type of :attr:`Deployment.controller_handle`, and the one
    :class:`realsim.seams.controller_handle.FakeControllerHandle` implements. Read
    the argument lists off :class:`Controller`.
    """

    locate_volumes: ActorEndpoint
    notify_put_batch: ActorEndpoint
    notify_delete: ActorEndpoint
    notify_delete_batch: ActorEndpoint
    keys: ActorEndpoint


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
    def controller_handle(self) -> ControllerHandle:
        """The controller's endpoint surface: the directory calls."""
        ...
