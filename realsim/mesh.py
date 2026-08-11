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
  and its :class:`~realsim.seams.controller_handle.FakeControllerHandle`;
* one :class:`~realsim.seams.volume_handle.FakeVolumeHandle` per node, each
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
from typing import Any, Callable, Dict, Iterator, List, Optional

from realsim.adapters.real_client import RealClientAdapter
from realsim.adapters.real_controller import make_controller_adapter
from realsim.seams import factory
from realsim.seams.transport import Endpoint, InMemoryTransport
from realsim.seams.volume_handle import FakeVolumeHandle
from proposed import View
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile
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
        policy: Optional[Any] = None,
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
        # A policy installed here IS a control plane, and this is where it runs:
        # inside the endpoint's locate_volumes (see FakeControllerHandle._route).
        # ``None`` is the naive answer -- every holder, directory order -- which
        # is what the real directory returns unaided, so an unrouted mesh pays
        # nothing for the hook.
        if policy is not None:
            self.controller_handle.install_policy(policy, self.view)
        # Each volume's byte capacity comes from the run's profile
        # (``storage_capacity_bytes``, default unbounded); the seam enforces it
        # against the aggregate resident working set.
        self.volumes: Dict[str, FakeVolumeHandle] = {
            vid: FakeVolumeHandle(volume_id=self.topology[vid].id, profile=self.profile)
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
    @property
    def view(self) -> View:
        """A :class:`~proposed.view.View` over this mesh's directory + topology.

        Built here rather than in ``proposed`` so the proposal never has to know
        what a :class:`Mesh` is.
        """
        return View(self.controller_handle, self.topology)

    def adapter(self, node_id: str) -> RealClientAdapter:
        """The :class:`RealClientAdapter` co-located with ``node_id``."""
        return self.adapters[node_id]

    def client_for(self, node_id: str) -> Any:
        """:class:`proposed.deployment.Deployment` -- the client for ``node_id``.

        Binds the calling coroutine to ``node_id`` first, so a capability's data
        plane never has to know that many clients share this process. A real
        deployment has one client and no binding to do.

        """
        self.bind_source(node_id)
        return self.client(node_id)

    @property
    def controller_handle(self) -> Any:
        """:class:`proposed.deployment.Deployment` -- the directory endpoints.

        The one way to reach the directory service: a client is built with it, a
        ``View`` reads through it, and an installed policy is consulted inside it.
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

        Binds two identities for the same client: the locality endpoint the cost
        model prices against, and the directory volume id a routing policy is
        asked about (see :func:`realsim.seams.factory.bind_requester`).
        """
        factory.bind_source(self.topology[node_id])
        factory.bind_requester(node_id)

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
