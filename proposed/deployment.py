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
is not payload, so it is declared: :class:`Controller` below says which endpoints
a controller offers and with what arguments. Leaving that as ``Any``
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

__all__ = ["Controller", "Deployment"]


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

    Under simulation
    :class:`realsim.seams.controller_handle.FakeControllerHandle` stands here.
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
    def controller_handle(self) -> Controller:
        """The controller's endpoint surface: the directory calls."""
        ...
