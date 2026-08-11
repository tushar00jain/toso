"""The dedup scenarios: which configurations each comparison runs.

A scenario is a set of choices, expressed as :class:`~realsim.run.Run` values --
how many readers, which fan-out caps, and what each configuration installs. It
wires no clock, no mesh and no plane; :meth:`realsim.run.Run.execute` does that, the
same way for every capability.

This is the point of the whole exercise. Every run below shares one
:class:`~putget_sim.workload.put_get.PutGetBurst` -- ordinary user code: seed
``W``, then a gather of ``client.get(W)``. The baseline installs nothing and
gets ``m x``. The routed runs add
:class:`~dedup_sim.control.routing.DedupPolicy` and the read-through
:class:`~dedup_sim.data.read_through.ReadThroughPlane` and get 1x. Same topology,
same payload, same cost model, same client calls -- the *only* difference is what
the ``Run`` carries, which is what makes the comparison mean something.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from putget_sim.workload.put_get import KEY, PutGetBurst
from realsim.demo import Console, Scenario
from realsim.run import Result, Run
from sim_common.trace import Trace

from ..control.routing import DedupPolicy
from ..data.read_through import ReadThroughPlane
from ..report.summary import BaselineReport, DedupReport

__all__ = ["NUM_READERS", "FANOUT_CAPS", "dedup_vs_baseline", "Dedup"]

#: Readers in the burst. Three is enough to show a chain and a shallow tree.
NUM_READERS = 3
#: The routed configurations to compare: cap 1 is a chain, cap 2 a shallow tree.
FANOUT_CAPS = (1, 2)


def dedup_vs_baseline(
    num_readers: int = NUM_READERS,
    caps: Sequence[int] = FANOUT_CAPS,
    *,
    burst: Optional[PutGetBurst] = None,
) -> List[Run]:
    """The unrouted baseline, then one routed run per fan-out cap.

    Returns the runs in comparison order: ``["baseline", "cap=1", "cap=2"]``.
    """
    burst = burst if burst is not None else PutGetBurst(num_readers)
    runs = [Run("baseline", burst, profile=burst.profile)]
    for cap in caps:
        # The policy records into the same trace the run reports, and it is
        # installed in the controller when the mesh is built -- so the trace has
        # to exist before the stack, which is why the Run carries it.
        trace = Trace()
        runs.append(
            Run(
                f"cap={cap}",
                burst,
                policy=DedupPolicy(fanout_cap=cap, trace=trace),
                plane=lambda sim, b=burst: ReadThroughPlane(
                    sim.mesh, KEY, b.put_value
                ),
                profile=burst.profile,
                trace=trace,
            )
        )
    return runs


class Dedup(Scenario):
    """The one comparison: the same burst unrouted, then routed at each cap."""

    name = "dedup"

    def runs(self, args) -> List[Run]:
        return dedup_vs_baseline()

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
            console.section(f"dedup policy  --  fanout_cap={cap} ({topo})")
            console.trace(result.trace, label=f"dedup(cap={cap}) run")
            console.summary(DedupReport(result, naive, cap))
            # 1x proven live on the real directory.
            assert result.ledger.origin_bytes == payload
            assert naive.ledger.origin_bytes == num_readers * payload

        console.section("NAIVE baseline  --  every reader pulls from the origin")
        console.trace(naive.trace, label="naive run")
        console.summary(BaselineReport(naive))
