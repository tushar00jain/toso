"""The one way to run anything: :func:`run_simulation`.

:class:`~realsim.simulation.Simulation` assembles the stack; this drives it. Every
run in the repo goes through here -- both capabilities' demos, the realsim fixture,
and any future capability -- so there is one signature to learn and one place a run
can go wrong.

A caller supplies two things, and they are exactly the two things that differ
between capabilities:

* a **topology** plus the usual run knobs (policy, profile, trace, ledger, ...);
* a **build** callback, which is handed the assembled stack and returns the
  capability's :class:`~proposed.plane.DataPlane` and its
  :class:`~realsim.runner.Workload`. It runs after the stack exists because a
  plane needs a deployment to call and a workload needs clients to drive.

Everything else -- whether rows are written per item, whether there is work
outliving the items, what precedes the first item -- is declared by the plane and
the workload themselves rather than passed here.

    result = run_simulation(topology, fixture.build, policy=DedupPolicy())
    print(result.ledger.origin_bytes, result.trace.fingerprint())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from proposed import DataPlane, Endpoint, Policy
from sim_common.cost_model import MachineProfile
from sim_common.report import Ledger
from sim_common.trace import Trace

from realsim.runner import Workload
from realsim.simulation import Simulation

__all__ = ["Build", "Result", "run_simulation"]

#: Handed the assembled stack, returns what to run on it.
Build = Callable[[Simulation], Tuple[Optional[DataPlane], Workload]]


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
    topology: Dict[str, Endpoint],
    build: Build,
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
        topology,
        policy=policy,
        profile=profile,
        trace=trace,
        ledger=ledger,
        real_directory=real_directory,
        quiet=quiet,
        random_seed=random_seed,
    )
    plane, workload = build(sim)
    results = sim.run(workload, plane=plane)
    return Result(results=results, trace=sim.trace, ledger=sim.ledger, sim=sim)
