"""The dedup scenario: one comparison, as a :class:`realsim.demo.Scenario`.

:class:`Dedup` declares its choices as :class:`~realsim.run.Run` values -- how
many readers, which fan-out caps, and what each configuration installs -- and
narrates the results. It wires no clock, no mesh and no plane, and it executes
nothing; :meth:`realsim.demo.Demo.main` does that with
:meth:`realsim.run.Run.execute`.

Every run shares one :class:`~putget_sim.workload.put_get.PutGetBurst` -- ordinary
user code: seed ``W``, then a gather of ``client.get(W)``. The baseline adds
nothing and gets ``m x``; the routed runs add
:class:`~dedup_sim.control.routing.Dedup` and the read-through
:class:`~dedup_sim.data.read_through.ReadThroughPlane` and get 1x. Same topology,
payload, cost model and client calls -- the *only* difference is what the ``Run``
carries, which is what makes the comparison mean something.
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
from ..report.summary import BaselineReport, DedupReport

__all__ = ["NUM_READERS", "FANOUT_CAPS", "Dedup"]

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
