"""A set of real storage volumes + co-located real clients on one directory.

:class:`Mesh` is ``realsim``'s multi-client wiring primitive: the pieces every
multi-node scenario needs before it can express a capability, assembled once.

    mesh = Mesh(topology, profile=profile, trace=trace)
    with mesh.installed():
        mesh.bind_source("s0")
        await mesh.client("s0").put_batch(...)

It owns, for a run:

* one controller adapter (the real ``Controller`` directory, real ``Trie`` or the
  dict shim -- see :func:`~realsim.adapters.real_controller.make_controller_adapter`)
  and its :class:`~realsim.seams.controller_handle.LocalControllerHandle`;
* one :class:`~realsim.seams.volume_service.VolumeService` per node (reached
  through a :class:`~realsim.seams.volume_handle.LocalVolumeHandle`), each
  backed by a real ``InMemoryStore``;
* one :class:`~realsim.adapters.real_client.RealClientAdapter` -- hence one real
  ``LocalClient`` -- per node, co-located with that node's volume;
* one shared :class:`~sim_common.resources.ResourceRegistry`, so concurrent
  transfers contend for the same links/stores; and
* the single shared ``create_transport_buffer`` substitution, which resolves the
  calling client's source endpoint from the contextvar in
  :mod:`realsim.seams.factory`.

Why it exists
-------------
``create_transport_buffer`` is a process-wide module global, so concurrent
clients cannot each install their own factory -- a multi-client drive needs *one*
factory that resolves the caller's source endpoint dynamically. That constraint,
plus the volume/adapter/registry wiring above, is identical for every
multi-client scenario and independent of the capability under test. It used to
live inside a burst-shaped read coordinator, so capabilities that are not bursts
-- ``kvcache_sim``'s continuous arrival stream -- could not reuse it and had to
re-derive the wiring underneath. Extracting it here lets a capability package
hold only capability code.

Accounting hook
---------------
:attr:`Mesh.on_transfer` is an optional callback invoked as
``(kind, src_id, dst_id, nbytes, cost)`` for every transfer the shared factory's
transports charge. It is read at call time, so a consumer constructed *after* the
mesh -- a run's ledger, which needs the mesh to exist first -- can claim it.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from realsim.adapters.real_client import RealClientAdapter
from realsim.adapters.real_controller import make_controller_adapter
from realsim.seams import factory
from realsim.seams.transport import Endpoint, InMemoryTransport
from realsim.seams.volume_handle import LocalVolumeHandle
from realsim.seams.volume_service import VolumeService
from proposed import View
from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
    ProfileTransferCost,
)
from sim_common.resources import ResourceRegistry
from sim_common.trace import Trace

__all__ = ["OnTransfer", "Mesh"]

# A transfer-accounting callback: (kind, src_id, dst_id, nbytes, cost).
OnTransfer = Callable[[str, str, str, int, float], None]


class Mesh:
    """Real volumes + co-located real clients over one shared directory.

    Args:
        topology: ``node_id -> Endpoint``. The node id is also its storage-volume
            id in the real directory; the endpoint's ``.id`` is the transfer
            identity the cost model prices locality against.
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`
            supplying every cost constant (defaults to
            :data:`~sim_common.cost_model.DEFAULT_PROFILE`). Also supplies each
            volume's byte capacity (``storage_capacity_bytes``).
        trace: shared :class:`~sim_common.trace.Trace` for transfer events.
        registry: shared :class:`~sim_common.resources.ResourceRegistry`
            (``None`` -> one built from the ambient ``contention`` config, whose
            ``"none"`` default is inert and byte-identical to no contention).
        real_directory: controller directory backing (``None`` -> the ambient
            :data:`sim_common.config.SimConfig.real_directory`, default the real
            ``Trie``; ``False`` -> the dict shim). Changes no metric.
        on_transfer: optional transfer-accounting callback (see module docstring).
    """

    def __init__(
        self,
        topology: Dict[str, Endpoint],
        *,
        profile: Optional[MachineProfile] = None,
        trace: Optional[Trace] = None,
        registry: Optional[ResourceRegistry] = None,
        real_directory: Optional[bool] = None,
        on_transfer: Optional[OnTransfer] = None,
    ) -> None:
        self.topology: Dict[str, Endpoint] = dict(topology)
        self.ids: List[str] = sorted(self.topology)
        self.profile = profile if profile is not None else DEFAULT_PROFILE
        self.trace = trace if trace is not None else Trace()
        # One shared resource layer for the whole run (default "none" -> inert).
        self.registry = (
            registry if registry is not None else ResourceRegistry.from_config()
        )
        self.on_transfer = on_transfer

        # The directory service: an adapter owning the real ``Controller`` and
        # the endpoint in front of it (``.controller`` / ``.handle``). Reach the
        # endpoint through :attr:`controller_handle`, which is the name the
        # ``Deployment`` port uses, rather than a second alias here.
        self.directory = make_controller_adapter(real_directory)
        # A mesh answers for itself -- every holder, directory order -- unless the
        # caller of a read named the sources it prefers (``client_for(prefer=...)``).
        # Nothing decides in here: who to prefer is settled before the read.
        #
        # Each volume's byte capacity comes from the run's profile
        # (``storage_capacity_bytes``, default unbounded); the seam enforces it
        # against the aggregate resident working set, evicting its own coldest
        # before refusing a put that does not fit. It is handed the controller
        # handle so it can tell the directory what it dropped -- under the node
        # id, because that is the volume's *directory* identity: it is what the
        # co-located client registers its puts under (``client_volume_id``), so
        # it is the only name a ``notify_delete_batch`` can be matched against.
        # The endpoint's ``.id`` is a different identity (the transfer one) and
        # naming a dropped key with it would be silently ignored by the real
        # ``Controller._notify_delete``, whose ``missing_ok`` swallows a volume
        # id it does not know.
        self.volumes: Dict[str, LocalVolumeHandle] = {
            vid: LocalVolumeHandle(
                VolumeService(
                    volume_id=vid,
                    profile=self.profile,
                    controller=self.controller_handle,
                )
            )
            for vid in self.ids
        }
        # One real LocalClient per node, co-located with that node's volume.
        self.adapters: Dict[str, RealClientAdapter] = {
            vid: RealClientAdapter(
                self.controller_handle,
                volume_handles=self.volumes,
                client_volume_id=vid,
                topology=self.topology,
                profile=self.profile,
                trace=self.trace,
                registry=self.registry,
            )
            for vid in self.ids
        }

    # -- accessors ---------------------------------------------------------- #
    @cached_property
    def view(self) -> View:
        """A :class:`~proposed.view.View` over this mesh's directory + topology.

        Built here rather than in ``proposed`` so the proposal never has to know
        what a :class:`Mesh` is.

        Senses through the directory *service*, not the handle in front of it:
        ``locate_raw`` is the read with no caller's preference folded into it, so a
        control plane ranks the directory rather than an answer somebody has already
        ranked. Note what that also means -- a control plane's directory reads do not
        cross the handle, so they are not charged the hop a real one would pay.

        One per mesh, because a view carries the scope one decision pins its directory
        read in (:meth:`~proposed.view.View.pinned`): a second view over the same mesh
        would be a second scope, so a reader holding it would walk the directory again
        in the middle of a decision that had pinned it. Safe to cache because the ports
        under it are fixed for the run -- the topology is copied at construction and
        the directory adapter is built once.
        """
        return View(
            self.directory.service,
            self.topology,
            ProfileTransferCost(self.topology, self.profile).get_time,
        )

    def adapter(self, node_id: str) -> RealClientAdapter:
        """The :class:`RealClientAdapter` co-located with ``node_id``."""
        return self.adapters[node_id]

    def client_for(
        self, node_id: str, *, prefer: Optional[Sequence[str]] = None
    ) -> Any:
        """:class:`proposed.deployment.Deployment` -- the client for ``node_id``.

        Binds the calling coroutine to ``node_id`` first, so a capability's data
        plane never has to know that many clients share this process. A real
        deployment has one client and no binding to do.

        ``prefer`` is bound the same way and for a different reason: the real client
        has no source-preference parameter, so a caller that has one to give says so
        here and the directory read applies it
        (:func:`realsim.seams.factory.bind_prefer`). Bound on every call, ``None``
        included, so a client vended without one reads as an unrouted client does
        rather than inheriting the last preference this coroutine expressed.
        """
        self.bind_source(node_id)
        factory.bind_prefer(prefer)
        return self.client(node_id)

    def volume_handle(self, node_id: str) -> Any:
        """:class:`proposed.deployment.Deployment` -- ``node_id``'s volume."""
        return self.volumes[node_id]

    @property
    def controller_handle(self) -> Any:
        """:class:`proposed.deployment.Deployment` -- the directory endpoints.

        The one way a *caller* reaches the directory service: every client is built
        with it. A ``View`` goes to the service behind it instead (see :attr:`view`).
        """
        return self.directory.handle

    def client(self, node_id: str) -> Any:
        """The real ``LocalClient`` co-located with ``node_id``."""
        return self.adapters[node_id].client

    def endpoint(self, node_id: str) -> Endpoint:
        """``node_id``'s locality :class:`~sim_common.topology.Endpoint`."""
        return self.topology[node_id]

    # -- the shared transport factory --------------------------------------- #
    def bind_source(self, node_id: str) -> None:
        """Charge the calling coroutine's transfers as originating at ``node_id``.

        Must be called before driving a client under :meth:`installed`, in the
        coroutine that will run the operation: the shared factory has no other way
        to tell which of the mesh's clients it is building a transport for.
        """
        factory.bind_source(self.topology[node_id])

    def _dispatch_transfer(
        self, kind: str, src_id: str, dst_id: str, nbytes: int, cost: float
    ) -> None:
        """Forward a transfer to :attr:`on_transfer`, read at call time."""
        if self.on_transfer is not None:
            self.on_transfer(kind, src_id, dst_id, nbytes, cost)

    def _build(self, storage_volume_ref) -> InMemoryTransport:
        """The shared ``create_transport_buffer``: one transport per operation."""
        return InMemoryTransport(
            storage_volume_ref,
            src=factory.current_source(),
            dst=self.topology[storage_volume_ref.volume_id],
            profile=self.profile,
            trace=self.trace,
            on_transfer=self._dispatch_transfer,
            registry=self.registry,
        )

    @contextmanager
    def installed(self) -> Iterator["Mesh"]:
        """Install this mesh's shared transport factory for the duration of the block.

        Only one substitution may be active process-wide; see
        :func:`realsim.seams.factory.installed`.
        """
        with factory.installed(self._build, owner=self):
            yield self
