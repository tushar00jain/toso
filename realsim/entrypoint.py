"""The one way to run anything: :func:`run_simulation`.

:class:`~realsim.simulation.Simulation` assembles the stack; this drives it. Every
run in the repo goes through here -- both capabilities' demos, the realsim fixture,
and any future capability -- so there is one signature to learn and one place a run
can go wrong.

A caller supplies two things, and they are exactly the two things that differ
between capabilities:

* a :class:`Workload` -- the scenario: its topology, how to build its data plane
  and work items onto the assembled stack, and anything that precedes them;
* the run knobs (policy, profile, trace, ledger, ...).

Everything else -- whether rows are written per item, whether there is work
outliving the items -- is declared by the data plane itself rather than passed
here.

    result = run_simulation(MyWorkload(...), policy=DedupPolicy())
    print(result.ledger.origin_bytes, result.trace.fingerprint())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from proposed import DataPlane, Endpoint, Policy
from sim_common.cost_model import MachineProfile
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.runner import WorkItem
from realsim.simulation import Simulation

__all__ = ["Workload", "Result", "run_simulation"]


class Workload:
    """What to simulate: a topology, and how to build the work onto a stack.

    One object per scenario. :attr:`topology` is known up front; :meth:`build` is
    called once the stack exists, because a data plane needs a deployment to call
    and work items need clients to drive. :meth:`setup` is for anything that is
    part of the simulated timeline but is not an item -- seeding a key before a
    read burst -- and runs before the first item is released.

    Subclass it per scenario; a capability's harness then hands the instance to
    :func:`run_simulation` and nothing else.
    """

    #: ``node_id -> Endpoint``. The node id is also its storage-volume id.
    topology: Dict[str, Endpoint]

    def build(
        self, sim: Simulation
    ) -> Tuple[Optional[DataPlane], Sequence[WorkItem]]:
        """Return this workload's data plane and its work items."""
        raise NotImplementedError

    async def setup(self, sim: Simulation) -> None:
        """Work that precedes the items on the clock. Default: nothing."""

    def result(self, result: "Result") -> "Result":
        """Enrich the run's :class:`Result` with facts only this scenario knows.

        Default: the generic result unchanged. A scenario whose summary needs more
        than the ledger carries (what the payload was, which volume was the
        origin) returns a subclass here, so a harness never assembles one itself.
        """
        return result


@dataclass
class Result:
    """What a run produced."""

    #: ``item id -> whatever that item's code returned``.
    results: Dict[str, Any]
    #: The run's event trace (and its fingerprint).
    trace: Trace
    #: The run's outcome rows and transfer accounting.
    ledger: Ledger
    #: The stack it ran on, for a caller that wants to inspect volumes,
    #: the directory, or the clock afterwards.
    sim: Simulation


def run_simulation(
    workload: Workload,
    *,
    policy: Optional[Policy] = None,
    profile: Optional[MachineProfile] = None,
    trace: Optional[Trace] = None,
    ledger: Optional[Ledger] = None,
    real_directory: Optional[bool] = None,
    quiet: Optional[bool] = None,
    random_seed: Optional[int] = None,
) -> Result:
    """Assemble a stack, build the capability onto it, run it, return the result."""
    sim = Simulation(
        workload.topology,
        policy=policy,
        profile=profile,
        trace=trace,
        ledger=ledger,
        real_directory=real_directory,
        quiet=quiet,
        random_seed=random_seed,
    )
    plane, items = workload.build(sim)
    results = sim.run(items, plane=plane, setup=workload.setup)
    return workload.result(
        Result(results=results, trace=sim.trace, ledger=sim.ledger, sim=sim)
    )
