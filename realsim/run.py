"""One configuration to execute, and the one way to execute it.

A scenario is almost never a single run -- it is a comparison. Dedup runs the
same burst unrouted and then once per fan-out cap; kvcache runs the same request
stream under two schedulers. What differs between those runs is not the workload,
it is the *policy* and the *data plane*.

:class:`Run` is that difference, as data: a label, the workload, and the pieces
the capability installs around it. :func:`execute` is the only code that turns
one into a :class:`Result`, so three capabilities cannot drift in how they wire a
stack -- which is what a per-capability ``harness.py`` used to allow, each with
its own signature (``run_burst(num_readers, ...)`` vs ``run(topology, requests,
kind, ...)``).

    runs = [Run("baseline", burst),
            Run("cap=1", burst, policy=DedupPolicy(1, trace=t), plane=make_plane)]
    results = [execute(r) for r in runs]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from proposed import DataPlane, Policy
from sim_common.cost_model import MachineProfile
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.simulation import Simulation
from realsim.workload import Workload

__all__ = ["Run", "Result", "execute"]

#: Builds the capability's data plane once the stack exists. It cannot be a
#: plain object: a plane reaches for the clock, the mesh and the ledger, none of
#: which exist before the ``Simulation`` does.
MakePlane = Callable[[Simulation], DataPlane]


@dataclass
class Run:
    """One labelled configuration. Data -- executing it is :func:`execute`'s job.

    Args:
        label: how this run is named in a comparison ("baseline", "cap=1").
        workload: the work to perform. Shared across the runs of a comparison,
            which is what makes the comparison mean something.
        policy: installed in the real controller's ``locate_volumes``. ``None``
            leaves the directory answering for itself.
        plane: builds the capability's :class:`~proposed.plane.DataPlane` onto
            the assembled stack. ``None`` -> no plane, the plain path.
        profile / trace / ledger: the run's target machine, event trace and
            outcome ledger. A capability with a richer outcome row passes its own
            ``Ledger`` subclass. A policy that records into the same trace the
            run reports needs it built here, before the stack.
    """

    label: str
    workload: Workload
    policy: Optional[Policy] = None
    plane: Optional[MakePlane] = None
    profile: Optional[MachineProfile] = None
    trace: Optional[Trace] = None
    ledger: Optional[Ledger] = None


@dataclass
class Result:
    """What a run produced. One type, for every capability.

    ``trace`` and ``ledger`` are the stack's own, exposed here so a report never
    has to reach through ``.sim``; ``workload`` is the run's, so scenario facts
    (how many readers, how big the payload) are read where they are defined
    rather than copied into a result subclass.
    """

    results: Dict[str, Any] = field(default_factory=dict)
    sim: Simulation = None  # type: ignore[assignment]
    run: Optional[Run] = None

    @property
    def label(self) -> str:
        """The configuration's label ("baseline", "cap=1")."""
        return self.run.label if self.run is not None else ""

    @property
    def workload(self) -> Optional[Workload]:
        """The workload this run performed."""
        return self.run.workload if self.run is not None else None

    @property
    def trace(self) -> Trace:
        """The run's event trace (and its fingerprint)."""
        return self.sim.trace

    @property
    def ledger(self) -> Ledger:
        """The run's outcome rows and transfer accounting."""
        return self.sim.ledger


def execute(
    run: Run,
    *,
    real_directory: Optional[bool] = None,
    quiet: Optional[bool] = None,
    random_seed: Optional[int] = None,
) -> Result:
    """Assemble a stack for ``run``, drive it, and return what it produced."""
    sim = Simulation(
        run.workload.topology,
        policy=run.policy,
        profile=run.profile,
        trace=run.trace,
        ledger=run.ledger,
        real_directory=real_directory,
        quiet=quiet,
        random_seed=random_seed,
    )
    plane = run.plane(sim) if run.plane is not None else None
    results = sim.run(run.workload, plane=plane)
    return Result(results=results, sim=sim, run=run)
