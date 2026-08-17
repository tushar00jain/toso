"""The one place the whole stack is assembled: :class:`Simulation`.

Every layer the design doc draws is built here, in that order, once. A capability
supplies only what is capability-specific -- a control plane, a data plane, a
workload -- and never re-stitches the stack underneath it. Before this existed each capability
wired its own clock, ledger, mesh and runner, and the shapes drifted apart: one
made an ``AsyncEngine`` directly and one went through ``run_sim``; one hooked
transfer accounting and one did not; one priced reads outside the stable run facts.

    sim = Simulation(topology, control=Dedup())
    results = sim.run(my_workload, plane=my_plane)

What it builds, top to bottom (compare the stack in the design doc):

* the **trace** and the **ledger** -- one of each per run;
* the **clock**: a :class:`~sim_common.async_engine.AsyncEngine` whose virtual
  time makes sleeps free and exact;
* the **mesh**: the real controller directory, a real volume + co-located real
  client per node, the shared resource registry, and the one shared
  ``create_transport_buffer``;
* the **control services**: the one ``control`` plane, fronted by a handle a
  caller reaches it through, and the sensor its hosts write, fronted beside it;
* the **directory sensor** the control plane reads;
* the **environment** holding that topology and the same profile the mesh charges
  against, so a scheduler cannot predict against another model.

:meth:`Simulation.run` then puts a :class:`~realsim.runner.Runner` over it and
drives a :class:`~realsim.run.Workload`'s items on the clock. It assembles;
the workload supplies the work; :meth:`realsim.run.Run.execute` pairs the two.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from proposed import ControlPlane, DirectorySensor, Endpoint, Environment
from sim_common import config
from sim_common.async_engine import AsyncEngine
from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
)
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.mesh import Mesh
from realsim.seams.control_plane_handle import LocalControlPlaneHandle
from realsim.seams.control_plane_service import ControlPlaneService
from realsim.seams.data_plane_handle import LocalDataPlaneHandle
from realsim.seams.data_plane_service import DataPlaneService
from realsim.seams.link import ServiceHop
from realsim.seams.dispatcher_handle import LocalDispatcherHandle
from realsim.seams.dispatcher_service import DispatcherService
from realsim.runner import ItemDispatch, Runner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from realsim.run import Workload

__all__ = ["Simulation"]


class Simulation:
    """One assembled stack: clock, mesh, environment, sensors and ledger.

    Args:
        topology: ``node_id -> Endpoint``. The node id is also its storage-volume
            id in the real directory.
        control: this run's one :class:`~proposed.plane.ControlPlane`, fronted by a
            :class:`~realsim.seams.control_plane_handle.LocalControlPlaneHandle` as
            :attr:`control_plane_handle`. Whatever it declares is what a data plane
            may ask it -- which volumes should serve a read, where a request should
            run, both -- and nothing here names any of them. ``None`` for a run that
            decides nothing: the directory then answers for itself, because nothing
            hands it a preference.

            One plane, because a capability's answers are formed from one picture. It
            is dedup's routing and kvcache's scheduler alike, and in kvcache's case
            that is the interesting part: the same object prices a pull and then
            answers the fetch with the peer it priced, so the peer control chose is
            the peer that serves the read, with nothing threaded through the data
            plane and nothing to keep in step between two objects.

            Where its hosts report their facts
            (:attr:`~proposed.plane.ControlPlane.dispatcher`) is fronted too, as
            :attr:`dispatcher_handle`, so they report into it directly.
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
        # transport factory. Nothing decides inside it -- a read applies the
        # preference its caller was given, and the plane that gives one is fronted
        # below.
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
        self.environment = Environment(
            topology=self.mesh.topology,
            profile=self.profile,
        )
        self.directory_sensor: DirectorySensor = self.mesh.directory_sensor

        # The plane receives the stable environment and the directory sensor before
        # its service is fronted.
        self.control_plane_handle: Optional[Any] = None
        self.dispatcher_handle: Optional[Any] = None
        # What reaching one of this run's *hosts* costs, and one distance for all of
        # them: a caller is off the box whichever host it is calling. The planes
        # themselves arrive later (:meth:`front_plane`).
        self._plane_hop = ServiceHop(config.current().client_rtt)
        self._planes: Dict[str, Any] = {}
        if control is not None:
            if not isinstance(control, ControlPlane):
                raise TypeError(
                    f"{type(control).__name__} is this run's control plane, so it "
                    f"must be a ControlPlane: a run holds the deciding object and "
                    f"hands it the stack's ports, which a selector or a bare "
                    f"callable has nowhere to receive"
                )
            control.attach(
                self.environment, {DirectorySensor: self.directory_sensor}
            )
        # What reaching a control plane costs. Resolved once, here, because this is
        # the one place a run's control services are built -- the same reason
        # ``make_controller_adapter`` resolves the directory's. One distance for both
        # of them: the dispatcher is held by the control plane whose sensors it folds
        # into, so a caller reaching either crosses the same boundary.
        hop = ServiceHop(config.current().control_rtt)
        if control is not None:
            # Fronted as a service, with an endpoint per member the plane declares --
            # read off the plane, not named here.
            self.control_plane_handle = LocalControlPlaneHandle(
                ControlPlaneService(control), hop=hop
            )
            # The other half of what a host says to control: a question goes to the
            # plane, a fact goes to the dispatcher the plane declares
            # (``ControlPlane.dispatcher``, which folds one action into every sensor it
            # moves). Read after ``attach``, because that is when a plane builds it.
            if control.dispatcher is not None:
                self.dispatcher_handle = LocalDispatcherHandle(
                    DispatcherService(control.dispatcher), hop=hop
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
    # store to call, the control plane to ask (:attr:`control_plane_handle`, whatever
    # that plane declares) and the dispatcher to report into
    # (:attr:`dispatcher_handle`). A *caller* finds one thing more on it, and no host
    # does: :meth:`plane_handle`, whichever host an address names.
    def client_for(
        self,
        node_id: str,
        *,
        prefer: Optional[Sequence[str]] = None,
    ) -> Any:
        """The torchstore client for ``node_id``, ready to be driven.

        ``prefer`` names the volumes its reads should be served by, best first --
        what :attr:`control_plane_handle` just answered.
        """
        return self.mesh.client_for(node_id, prefer=prefer)

    def volume_handle(self, node_id: str) -> Any:
        """A reference to ``node_id``'s storage volume."""
        return self.mesh.volume_handle(node_id)

    def front_plane(self, node_id: str, plane: Any) -> None:
        """Put a service in front of ``node_id``'s data plane, so a caller can reach it.

        The sibling of what ``__init__`` does for the control plane, called from the
        capability's own wiring instead, because that is where a run's hosts are built
        -- after the stack they attach to exists.
        """
        self._planes[node_id] = LocalDataPlaneHandle(
            DataPlaneService(plane), hop=self._plane_hop
        )

    def plane_handle(self, node_id: str) -> Any:
        """A reference to ``node_id``'s data plane, as a caller off the box holds it."""
        return self._planes[node_id]

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
