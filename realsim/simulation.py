"""The one place the whole stack is assembled: :class:`Simulation`.

Every layer the design doc draws is built here, in that order, once. A capability
supplies only what is capability-specific -- a control plane, a data plane, a
workload --
and never re-stitches the stack underneath it. Before this existed each capability
wired its own clock, ledger, mesh and runner, and the shapes drifted apart: one
made an ``AsyncEngine`` directly and one went through ``run_sim``; one hooked
transfer accounting and one did not; one built its transfer-cost estimate from the
same profile and topology as its mesh with nothing holding the two together.

    sim = Simulation(topology, control=DedupKeySelector())
    results = sim.run(my_workload, plane=my_plane)

What it builds, top to bottom (compare the stack in the design doc):

* the **trace** and the **ledger** -- one of each per run;
* the **clock**: a :class:`~sim_common.async_engine.AsyncEngine` whose virtual
  time makes sleeps free and exact;
* the **mesh**: the real controller directory, a real volume + co-located real
  client per node, the shared resource registry, and the one shared
  ``create_transport_buffer``. A ``control`` plane that answers the store's
  routing question is installed in the real ``locate_volumes`` body here;
* the **view** the control plane senses through, over that same directory;
* the **transfer-cost estimate**, from the *same* topology and profile the mesh
  charges against, so a scheduler cannot predict against one model while the
  transport charges another.

:meth:`Simulation.run` then puts a :class:`~realsim.runner.Runner` over it and
drives a :class:`~realsim.run.Workload`'s items on the clock. It assembles;
the workload supplies the work; :meth:`realsim.run.Run.execute` pairs the two.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from proposed import Endpoint, KeySelector, View
from sim_common import config
from sim_common.async_engine import AsyncEngine
from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
    ProfileTransferCost,
)
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.mesh import Mesh
from realsim.seams.cluster_model_handle import LocalClusterModelHandle
from realsim.seams.cluster_model_service import ClusterModelService
from realsim.seams.link import ServiceHop
from realsim.seams.control_plane_handle import LocalControlPlaneHandle
from realsim.seams.control_plane_service import ControlPlaneService
from realsim.runner import ItemDispatch, Runner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from realsim.run import Workload

__all__ = ["Simulation"]


class Simulation:
    """One assembled stack: clock, mesh, view, ledger, cost estimate.

    Args:
        topology: ``node_id -> Endpoint``. The node id is also its storage-volume
            id in the real directory.
        control: the plane that runs **in the directory service**, installed in the
            real controller's ``locate_volumes``, so a caller that just does
            ``client.get(K)`` is routed. Must be a
            :class:`~proposed.selector.KeySelector`, which is what claims a subject
            the directory can hand down. ``None`` leaves the directory answering
            for itself.
        placement: the plane the **application's own hosts ask**, fronted by a
            :class:`~realsim.seams.control_plane_handle.LocalControlPlaneHandle` as
            :attr:`control_plane_handle` and reached as its own service. Any
            :class:`~proposed.plane.ControlPlane`: what it answers with is between
            it and the hosts that ask. ``None`` for a capability that decides only
            inside the directory.

            Two arguments rather than a list, so each plane's role is the parameter
            it was passed as. dedup is ``control`` alone; kvcache is both, and the
            pair is the interesting case: a scheduler its hosts ask over the handle,
            and beside it a chain in the directory answering "which peer serves this
            prefix". They share the model the first writes and the second reads, so
            the peer control priced is the peer that serves the pull, with nothing
            threaded through the data plane to say so.

            A model of the cluster (:attr:`~proposed.plane.ControlPlane.cluster`) is
            fronted too, as :attr:`cluster_handle`, so the hosts report into it
            directly; one of the two planes may keep one.
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`;
            supplies every cost constant and each volume's byte capacity.
        trace: shared :class:`~sim_common.trace.Trace` (created if omitted).
        ledger: shared :class:`~sim_common.report.Ledger` (created if omitted).
            A capability with a richer outcome row passes its own subclass.
        real_directory: controller directory backing (``None`` -> ambient config).
        quiet: opt out of per-event tracing (``None`` -> ambient config).
        random_seed: engine ready-queue seed; ``None`` is FIFO and reproducible.
    """

    def __init__(
        self,
        topology: Dict[str, Endpoint],
        *,
        control: Optional[Any] = None,
        placement: Optional[Any] = None,
        profile: Optional[MachineProfile] = None,
        trace: Optional[Trace] = None,
        ledger: Optional[Ledger] = None,
        real_directory: Optional[bool] = None,
        quiet: Optional[bool] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.profile = profile if profile is not None else DEFAULT_PROFILE
        self.trace = trace if trace is not None else Trace()
        self.ledger = ledger if ledger is not None else Ledger()

        # The clock, created on first use: assembling a stack should not have the
        # side effect of standing up an event loop, and a caller may only want
        # the mesh (a test inspecting a carrier, say).
        self._loop: Optional[AsyncEngine] = None
        self._quiet = quiet
        self._random_seed = random_seed

        # The store: real controller + real volumes + real clients, and one shared
        # transport factory. What runs in the selector hook inside locate_volumes is
        # installed below, once the control plane has been attached: a selector
        # over this run's directory cannot exist before this run's directory does.
        self.mesh = Mesh(
            topology,
            profile=self.profile,
            trace=self.trace,
            real_directory=real_directory,
        )
        # Every transfer the transports charge lands in the run's one ledger.
        self.mesh.on_transfer = self.ledger.record_transfer

        # What the control plane may look at, and what it may price against --
        # both derived from the objects above rather than rebuilt beside them.
        self.view: View = self.mesh.view
        self.transfer_cost = ProfileTransferCost(self.mesh.topology, self.profile)

        # The planes, each named by the argument that says which side reaches it.
        # Both are attached before any wiring below: attach hands over the view and
        # the transfer-cost estimate that everything below is derived from, and a
        # plane may not be brought up against a stack another plane has not
        # finished joining.
        #
        # ``control`` first and ``placement`` second, which matters when the two
        # share a selector: the last attach is the view it senses through, and a
        # decision that pins a snapshot must not consult a ranking reading past it
        # (:meth:`~kvcache_sim.control.scheduler._Scheduler.attach`).
        self.control_plane_handle: Optional[Any] = None
        self.cluster_handle: Optional[Any] = None
        planes = tuple(p for p in (control, placement) if p is not None)
        for plane in planes:
            plane.attach(self.view, self.transfer_cost)
        # The store's own question, answered inside locate_volumes by the seam in
        # front of the directory (LocalControllerHandle). The subject a directory
        # hands down is keys, so being a KeySelector is the claim that makes a plane
        # installable there -- checked, because installing one that reads its subject
        # as anything else would answer a question the directory did not ask.
        if control is not None:
            if not isinstance(control, KeySelector):
                raise TypeError(
                    f"{type(control).__name__} is installed in locate_volumes, so "
                    f"it must be a KeySelector; pass an application's own plane as "
                    f"placement="
                )
            self.mesh.controller_handle.install_selector(control)
        # What reaching a control plane costs. Resolved once, here, because this is
        # the one place a run's control services are built -- the same reason
        # ``make_controller_adapter`` resolves the directory's. One distance for
        # all of them: a model is held by the control plane that reads it, so a
        # host reaching any of them crosses the same boundary.
        hop = ServiceHop(config.current().control_rtt)
        # The application's own question, fronted as a service of its own. A
        # capability that runs solely in the directory passes no ``placement`` and
        # gets none.
        if placement is not None:
            self.control_plane_handle = LocalControlPlaneHandle(
                ControlPlaneService(placement), hop=hop
            )
        # The other half of what a host says to control: a question goes to the
        # placement, a fact goes to the model it corrects. Only a plane that keeps
        # one has it (``ControlPlane.cluster``), and it is read after ``attach``
        # because that is when a model learns which cluster it is of.
        modelled = [p for p in planes if p.cluster is not None]
        if len(modelled) > 1:
            raise TypeError(
                f"both control planes keep a cluster model "
                f"({', '.join(type(p).__name__ for p in modelled)}): the hosts "
                f"report into one, so a run has one to front"
            )
        if modelled:
            self.cluster_handle = LocalClusterModelHandle(
                ClusterModelService(modelled[0].cluster), hop=hop
            )

    @property
    def loop(self) -> AsyncEngine:
        """The run's virtual clock."""
        if self._loop is None:
            self._loop = AsyncEngine(
                trace=self.trace, quiet=self._quiet, random_seed=self._random_seed
            )
        return self._loop

    # -- proposed.Deployment: what a data plane is attached to --------------- #
    # The store's three members delegate to the mesh, so an assembled stack *is* a
    # :class:`~proposed.deployment.Deployment`: a capability's data plane is handed
    # this one object (:meth:`~proposed.plane.DataPlane.attach`) and finds on it the
    # store to call, the control plane to ask (:attr:`control_plane_handle`) and the
    # model to report into (:attr:`cluster_handle`).
    def client_for(self, node_id: str) -> Any:
        """The torchstore client for ``node_id``, ready to be driven."""
        return self.mesh.client_for(node_id)

    def volume_handle(self, node_id: str) -> Any:
        """A reference to ``node_id``'s storage volume."""
        return self.mesh.volume_handle(node_id)

    @property
    def controller_handle(self) -> Any:
        """A reference to the directory service."""
        return self.mesh.controller_handle

    # -- the pieces a capability wires itself to ---------------------------- #
    @property
    def topology(self) -> Dict[str, Endpoint]:
        """``node_id -> Endpoint``."""
        return self.mesh.topology

    @property
    def ids(self) -> List[str]:
        """Node ids, sorted."""
        return self.mesh.ids

    def origins(self, *volume_ids: str) -> "Simulation":
        """Mark volumes as holding data before the run (fabric accounting).

        A transfer served by one of these is a fabric byte -- the cost a routing
        selector exists to cut. Returns self so it can be chained onto construction.
        """
        self.ledger.origins.update(volume_ids)
        return self

    # -- driving it ---------------------------------------------------------- #
    def run(
        self,
        workload: "Workload",
        *,
        dispatch: Optional[ItemDispatch] = None,
    ) -> Dict[str, Any]:
        """Run ``workload`` on this stack; return ``item id -> result``.

        The workload supplies the items and whatever precedes them on the clock;
        the dispatch says how an item is run and whether it publishes its own
        outcome rows, so neither is passed here.

        Closes the loop when done; a :class:`Simulation` runs once.
        """
        dispatch = dispatch if dispatch is not None else ItemDispatch()
        runner = Runner(
            self.mesh,
            dispatch=dispatch,
            ledger=None if dispatch.writes_own_outcomes else self.ledger,
        )
        items = workload.items(self)

        async def _go() -> Dict[str, Any]:
            await workload.prepare(self)
            return await runner.run(items)

        try:
            return self.loop.run_until_complete(_go())
        finally:
            self.loop.close()
