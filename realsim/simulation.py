"""The one place the whole stack is assembled: :class:`Simulation`.

Every layer the design doc draws is built here, in that order, once. A capability
supplies only what is capability-specific -- a policy, a data plane, a workload --
and never re-stitches the stack underneath it. Before this existed each capability
wired its own clock, ledger, mesh and runner, and the shapes drifted apart: one
made an ``AsyncEngine`` directly and one went through ``run_sim``; one hooked
transfer accounting and one did not; one built its transfer-cost estimate from the
same profile and topology as its mesh with nothing holding the two together.

    sim = Simulation(topology, policy=DedupPolicy())
    results = sim.run(my_workload, plane=my_plane)

What it builds, top to bottom (compare the stack in the design doc):

* the **trace** and the **ledger** -- one of each per run;
* the **clock**: a :class:`~sim_common.async_engine.AsyncEngine` whose virtual
  time makes sleeps free and exact;
* the **mesh**: the real controller directory, a real volume + co-located real
  client per node, the shared resource registry, and the one shared
  ``create_transport_buffer``. A ``policy``, if given, is installed in the real
  ``locate_volumes`` body here;
* the **view** the control plane senses through, over that same directory;
* the **transfer-cost estimate**, from the *same* topology and profile the mesh
  charges against, so a scheduler cannot predict against one model while the
  transport charges another.

:meth:`Simulation.run` then puts a :class:`~realsim.runner.Runner` over it and
drives a :class:`~realsim.run.Workload`'s items on the clock. It assembles;
the workload supplies the work; :func:`realsim.run.execute` pairs the two.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from proposed import DataPlane, Endpoint, Policy, View
from sim_common.async_engine import AsyncEngine
from sim_common.cost_model import (
    DEFAULT_PROFILE,
    MachineProfile,
    ProfileTransferCost,
)
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.mesh import Mesh
from realsim.runner import Runner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from realsim.run import Workload

__all__ = ["Simulation"]


class Simulation:
    """One assembled stack: clock, mesh, view, ledger, cost estimate.

    Args:
        topology: ``node_id -> Endpoint``. The node id is also its storage-volume
            id in the real directory.
        policy: optional :class:`~proposed.policy.Policy`, installed in the real
            controller's ``locate_volumes``. ``None`` leaves the directory
            answering for itself, which is what the naive policy says anyway.
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
        policy: Optional[Policy] = None,
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
        self.policy = policy

        # The clock, created on first use: assembling a stack should not have the
        # side effect of standing up an event loop, and a caller may only want
        # the mesh (a test inspecting a carrier, say).
        self._loop: Optional[AsyncEngine] = None
        self._quiet = quiet
        self._random_seed = random_seed

        # The store: real controller + real volumes + real clients, one shared
        # transport factory, and the policy hook inside locate_volumes.
        self.mesh = Mesh(
            topology,
            profile=self.profile,
            trace=self.trace,
            policy=policy,
            real_directory=real_directory,
        )
        # Every transfer the transports charge lands in the run's one ledger.
        self.mesh.on_transfer = self.ledger.record_transfer

        # What the control plane may look at, and what it may price against --
        # both derived from the objects above rather than rebuilt beside them.
        self.view: View = self.mesh.view
        self.transfer_cost = ProfileTransferCost(self.mesh.topology, self.profile)

    @property
    def loop(self) -> AsyncEngine:
        """The run's virtual clock."""
        if self._loop is None:
            self._loop = AsyncEngine(
                trace=self.trace, quiet=self._quiet, random_seed=self._random_seed
            )
        return self._loop

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
        policy exists to cut. Returns self so it can be chained onto construction.
        """
        self.ledger.origins.update(volume_ids)
        return self

    # -- driving it ---------------------------------------------------------- #
    def run(
        self,
        workload: "Workload",
        *,
        plane: Optional[DataPlane] = None,
    ) -> Dict[str, Any]:
        """Run ``workload`` on this stack; return ``item id -> result``.

        The workload supplies the items and whatever precedes them on the clock;
        the plane says whether it publishes its own outcome rows and whether it
        has work outliving the items, so none of that is passed here.

        Closes the loop when done; a :class:`Simulation` runs once.
        """
        plane = plane if plane is not None else DataPlane()
        drains = type(plane).drain is not DataPlane.drain
        runner = Runner(
            self.mesh,
            plane=plane,
            ledger=None if plane.writes_own_outcomes else self.ledger,
            drain=plane.drain if drains else None,
        )
        items = workload.items(self)

        async def _go() -> Dict[str, Any]:
            await workload.prepare(self)
            return await runner.run(items)

        try:
            return self.loop.run_until_complete(_go())
        finally:
            self.loop.close()
