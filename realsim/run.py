"""The run lifecycle, in one place.

Five things, in the order they happen: what the work is (:class:`Workload`), how
one configuration of it is described (:class:`Run`), how it is executed
(:meth:`Run.execute`), what that produced (:class:`Result`), and how that is shown
(:class:`Report`). They were five files; they are one, because a capability
touches all of them for a single scenario and jumping between modules to read one
story is its own cost.

A scenario is almost never a single run -- it is a comparison. Dedup runs the
same burst unrouted and then once per fan-out cap; kvcache runs the same request
stream under two schedulers. What differs between those runs is not the workload,
it is the *control plane* and the *data plane*. A :class:`Run` names both, so
neither is buried inside the other's factory -- kvcache used to build its
scheduler inside the thing that built its serving plane, which hid that the two
halves run in different places.

:class:`Run` is that difference: a label, the workload, and the pieces the
capability installs around it. :meth:`Run.execute` is the only code that turns
one into a :class:`Result`, so three capabilities cannot drift in how they wire a
stack -- which is what a per-capability ``harness.py`` used to allow, each with
its own signature (``run_burst(num_readers, ...)`` vs ``run(topology, requests,
kind, ...)``).

    runs = [Run("baseline", burst),
            Run("cap=1", burst, control=DedupKeySelector(1, trace=t), data=make_plane)]
    results = [r.execute() for r in runs]

:class:`Simulation` deliberately does *not* take a ``Run``. It is constructible
from a topology alone, and a dozen tests do exactly that to get an assembled
stack -- mesh, view, transfer-cost estimate -- with nothing to run on it. A
``Run`` requires a ``Workload``, so demanding one would make those tests invent
work they do not perform. :meth:`Run.execute` is the seam between the two: it is
where a capability's declaration becomes the stack's constructor arguments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from proposed import Endpoint, KeySelector
from sim_common.cost_model import MachineProfile
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.runner import ItemDispatch, WorkItem
from realsim.simulation import Simulation

__all__ = ["Workload", "Report", "MakePlane", "Run", "Result"]


class Workload(ABC):
    """The work a run performs. Assembles nothing, renders nothing.

    Args:
        topology: ``node_id -> Endpoint``. The node id is also its storage-volume
            id in the real directory. Known before the stack exists, because
            :class:`~realsim.simulation.Simulation` is built from it.
    """

    def __init__(self, topology: Dict[str, Endpoint]) -> None:
        self.topology = topology

    @abstractmethod
    def items(self, sim: Simulation) -> Sequence[WorkItem]:
        """The work items to release, given the assembled stack."""

    async def prepare(self, sim: Simulation) -> None:
        """Work that precedes the items on the clock. Default: nothing."""


class Report(ABC):
    """What a run produced, as text.

    One method, so every capability's ``report/`` package exposes the same thing
    and a demo can render any of them without knowing which capability it holds.
    A report owns no run state: it is constructed from the results it describes
    (and whatever scenario facts it needs, which live on the workload those
    results carry) and answers :meth:`render`.

    Not to be confused with :mod:`sim_common.report`, the *measurement* side --
    ``Ledger``, outcome rows, the source->dest tree renderer -- that a report
    reads from.
    """

    @abstractmethod
    def render(self) -> str:
        """The rendered summary, ready to log."""


#: Builds the capability's data plane once the stack exists. It cannot be a
#: plain object: a plane reaches for the clock, the mesh, the ledger and the
#: coordinator handle, none of which exist before the ``Simulation`` does.
MakePlane = Callable[[Simulation], ItemDispatch]



@dataclass
class Run:
    """One labelled configuration, and the one way to carry it out.

    Args:
        label: how this run is named in a comparison ("baseline", "cap=1").
        workload: the work to perform. Shared across the runs of a comparison,
            which is what makes the comparison mean something.
        control: the capability's **control plane** -- the half that decides. It
            always runs in a service and is always reached through the seam in
            front of that service; a data plane never holds it. What this field's
            two forms choose is *which* service, which is a fact about where the
            capability ships rather than a configuration:

            * a :class:`~proposed.selector.KeySelector` runs **in the directory
              service**, installed in the real controller's ``locate_volumes``,
              and is reached through the seam already standing there
              (:class:`~realsim.seams.controller_handle.LocalControllerHandle`) --
              a caller just calls ``client.get`` and is routed. This is dedupe:
              its whole control plane is the selector, and it must exist before the
              mesh is built, which is why an object rather than a factory;
            * a :data:`MakeControl` factory runs **in its own service**, built
              over the assembled stack and fronted by a
              :class:`~realsim.seams.placement_handle.LocalPlacementHandle` as
              ``sim.placement_handle``. This is kvcache: its scheduler holds every
              instance's queue, cache and decode occupancy, so it needs
              ``sim.view`` and cannot be built before the stack.

            A capability whose control plane runs in *both* places passes a
            sequence of the two, which is what kvcache does: the placement
            decides where to prefill, and a
            :class:`~kvcache_sim.control._source.LongestPrefixKeySelector` in the
            directory narrows the pull to the peer the coordinator already priced
            -- without it the client reads from whichever holder the directory
            lists first, and a pull predicted over NVLink can be charged over
            RDMA. ``None`` leaves the directory answering for itself.

            (Neither seam charges its hop by default: ``--coordinator-rtt`` gives
            the coordinator one a cost, and the client-to-controller hop is free
            for every capability, including the baseline.)
        data: builds this run's :class:`~realsim.runner.ItemDispatch` onto the
            assembled stack -- how the executing half is driven, and where a
            capability's :class:`~proposed.plane.DataPlane` is plugged in.
            ``None`` -> the plain path: run each item, nothing around it. It
            reaches that control plane through ``sim.placement_handle``, never by
            being handed the object.
        profile / trace / ledger: the run's target machine, event trace and
            outcome ledger. A capability with a richer outcome row passes its own
            ``Ledger`` subclass. A control plane that records into the same trace
            the run reports needs it built here, before the stack.
    """

    label: str
    workload: Workload
    control: Optional[Any] = None
    data: Optional[MakePlane] = None
    profile: Optional[MachineProfile] = None
    trace: Optional[Trace] = None
    ledger: Optional[Ledger] = None

    def execute(
        self,
        *,
        real_directory: Optional[bool] = None,
        quiet: Optional[bool] = None,
        random_seed: Optional[int] = None,
    ) -> "Result":
        """Assemble a stack for this configuration, drive it, return the result.

        The knobs here are the ones a *caller* varies per invocation rather than
        the scenario declaring them -- the CLI's ``--seed``, a test forcing the
        shim directory. Everything that describes the run itself is a field.
        """
        sim = Simulation(
            self.workload.topology,
            control=self.control,
            profile=self.profile,
            trace=self.trace,
            ledger=self.ledger,
            real_directory=real_directory,
            quiet=quiet,
            random_seed=random_seed,
        )
        plane = self.data(sim) if self.data is not None else None
        results = sim.run(self.workload, dispatch=plane)
        return Result(results=results, sim=sim, run=self)


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
