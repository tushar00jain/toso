"""The dedup scenarios: two comparisons, as :class:`realsim.demo.Scenario` values.

:class:`Dedup` declares its choices as :class:`~realsim.run.Run` values -- how
many readers, which fan-out caps, and what each configuration installs -- and
narrates the results. It wires no clock, no mesh and no plane, and it executes
nothing; :meth:`realsim.demo.Demo.main` does that with
:meth:`realsim.run.Run.execute`.

Every run of :class:`Dedup` shares one
:class:`~putget_sim.workload.put_get.PutGetBurst` -- ordinary user code: seed ``W``,
then a gather of ``client.get(W)``. The baseline adds nothing and gets ``m x``; the
routed runs add :class:`~dedup_sim.control.routing.Dedup` and the read-through
:class:`~dedup_sim.data.read_through.ReadThroughPlane` and get 1x. Same topology,
payload, cost model and client calls -- the *only* difference is what the ``Run``
carries, which is what makes the comparison mean something.

:class:`WeightSync` is the same three-way comparison over a key with **two** holders
(:class:`~dedup_sim.workload._weight_sync.WeightSync`), which is where the chain has an
alternative: one trainer per generator instead of a queue behind one trainer.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from putget_sim.workload.put_get import KEY, PutGetBurst
from realsim.demo import Console, Scenario
from realsim.runner import ItemDispatch
from realsim.run import Result, Run
from sim_common.trace import Trace

from ..control import routing
from ..data.read_through import ReadThroughPlane
from ..report.summary import BaselineReport, DedupReport, WeightSyncReport
from ._weight_sync import WeightSync as WeightSyncWorkload

__all__ = ["NUM_READERS", "FANOUT_CAPS", "Dedup", "WeightSync"]

#: Readers in the burst. Three is enough to show a chain and a shallow tree.
NUM_READERS = 3
#: The routed configurations to compare: cap 1 is a chain, cap 2 a shallow tree.
FANOUT_CAPS = (1, 2)


def _read_through(sim, burst) -> ItemDispatch:
    """This run's data plane, brought up against the assembled stack.

    The plane makes both of a reader's store calls itself, so what the runner drives
    is one member and the item carries only who is reading.
    """
    plane = ReadThroughPlane(KEY, burst.put_value, trace=sim.trace)
    plane.attach(sim)
    return ItemDispatch(lambda item: plane.read_through(item.id))


class Dedup(Scenario):
    """The unrouted baseline, then one routed run per fan-out cap.

    Every run shares one :class:`~putget_sim.workload.put_get.PutGetBurst`, so
    the baseline and each cap cannot differ in what they simulate. Parameterized
    by construction rather than by a flag, so a test can ask for a narrower
    comparison without a command line.
    """

    name = "dedup"

    def __init__(
        self,
        num_readers: int = NUM_READERS,
        caps: Sequence[int] = FANOUT_CAPS,
        *,
        burst: Optional[PutGetBurst] = None,
    ) -> None:
        self.num_readers = num_readers
        self.caps = caps
        self.burst = burst

    def runs(self, args=None) -> List[Run]:
        """The runs in comparison order: ``["baseline", "cap=1", "cap=2"]``."""
        burst = self.burst or PutGetBurst(self.num_readers)
        runs = [Run("baseline", burst, profile=burst.profile)]
        for cap in self.caps:
            # The plane records into the same trace the run reports, and it is
            # attached when the stack is built -- so the trace has to exist before
            # the stack, which is why the Run carries it.
            trace = Trace()
            runs.append(
                Run(
                    f"cap={cap}",
                    burst,
                    control=routing.Dedup(fanout_cap=cap, trace=trace),
                    # The capability is the plane's one member; what the runner
                    # drives is the dispatch that calls it. Wiring, so it is here
                    # and not in ``data/``. The plane is built with its knobs and
                    # handed the deployment, exactly as a control plane is.
                    data=lambda sim, b=burst: _read_through(sim, b),
                    profile=burst.profile,
                    trace=trace,
                )
            )
        return runs

    def show(self, console: Console, results: Sequence[Result]) -> None:
        naive, routed = results[0], results[1:]
        payload = naive.workload.payload_bytes
        num_readers = naive.workload.num_readers
        console.trace(naive.trace, label="naive run")

        console.section(f"DEDUP on the REAL directory  --  {num_readers} readers get W")
        console.info(
            "directory: real torchstore.controller.Controller (real Trie state)"
        )
        console.info(
            "payload(W): %dB   1x-union target (each unique byte once): %dB",
            payload, payload,
        )

        for result in routed:
            cap = int(result.label.split("=")[1])
            topo = "chain" if cap == 1 else "tree"
            console.section(f"dedup selector  --  fanout_cap={cap} ({topo})")
            console.trace(result.trace, label=f"dedup(cap={cap}) run")
            console.summary(DedupReport(result, naive, cap))
            # 1x proven live on the real directory.
            assert result.ledger.origin_bytes == payload
            assert naive.ledger.origin_bytes == num_readers * payload

        console.section("NAIVE baseline  --  every reader pulls from the origin")
        console.trace(naive.trace, label="naive run")
        console.summary(BaselineReport(naive))


class WeightSync(Scenario):
    """One key, two trainer replicas, two generators: chain it or spread it.

    Three runs over the one workload, in the order the trade reads: no routing at all,
    dedup as it stands, and dedup with ``spread`` on. What differs is one flag on the
    control plane -- the data plane is the same read-through in both routed runs, so
    every generator publishes what it read either way.

    Parameterized by construction so a test can widen it (more generators than
    replicas is where the peer chain comes back), not by a flag.
    """

    name = "weight_sync"

    def __init__(self, num_trainers: int = 2, num_generators: int = 2) -> None:
        self.num_trainers = num_trainers
        self.num_generators = num_generators

    def runs(self, args=None) -> List[Run]:
        """``["baseline", "dedup", "dedup+spread"]`` over one shared workload."""
        workload = WeightSyncWorkload(self.num_trainers, self.num_generators)
        runs = [Run("baseline", workload, profile=workload.profile)]
        for label, spread in (("dedup", False), ("dedup+spread", True)):
            # The plane records into the trace the run reports, so the trace exists
            # before the stack the plane is attached to (as in ``Dedup`` above).
            trace = Trace()
            runs.append(
                Run(
                    label,
                    workload,
                    control=routing.Dedup(spread=spread, trace=trace),
                    data=lambda sim, w=workload: _read_through(sim, w),
                    profile=workload.profile,
                    trace=trace,
                )
            )
        return runs

    def show(self, console: Console, results: Sequence[Result]) -> None:
        workload = results[0].workload
        console.section(
            f"WEIGHT SYNC  --  {workload.num_generators} generators get W from "
            f"{workload.num_trainers} trainer replicas"
        )
        console.info(
            "the replicas are equidistant, so locality prices them identically and the"
        )
        console.info(
            "id tie-break sends every generator to t0; a load term is what separates"
        )
        console.info("them, and the read-through is on in both routed runs.")
        for result in results:
            console.trace(result.trace, label=f"{result.label} run")
        console.summary(WeightSyncReport(results))
