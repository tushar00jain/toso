"""What this sim runs: one unrouted burst, as a :class:`realsim.demo.Scenario`.

The degenerate comparison -- a single :class:`~realsim.run.Run` with no selector
and no data plane. That absence is the content: it is the ``m x`` baseline
``dedup_sim`` measures its 1x against, and ``dedup_sim`` builds its own runs over
this same :class:`~putget_sim.workload.put_get.PutGetBurst`.

Unlike the other two sims, this scenario's shape comes from the command line
(``-m``, ``-n``, ``--mode``), so :meth:`Burst.runs` reads its ``args``.
"""

from __future__ import annotations

from typing import List, Sequence

from realsim.demo import Console, Scenario
from realsim.run import Result, Run
from sim_common.cost_model import DEFAULT_PROFILE

from ..report.summary import BurstReport
from .put_get import PutGetBurst

__all__ = ["NUM_READERS", "PROFILE", "Burst"]

#: Readers in the burst, when the CLI does not say otherwise.
NUM_READERS = 3

# The cost model is driven by a MachineProfile that describes the *target*
# machine being simulated -- never the box this demo runs on. Costs are analytic
# functions of modeled bytes/flops, so the same profile yields the same trace on
# any host. The demo uses the illustrative DEFAULT_PROFILE; a real study would
# swap in a profile measured/spec'd for its target hardware.
PROFILE = DEFAULT_PROFILE



class Burst(Scenario):
    """The one comparison there is not: a single unrouted run."""

    name = "burst"

    def runs(self, args=None) -> List[Run]:
        """The one run: ``args.readers`` readers get W, with nothing installed."""
        workload = PutGetBurst(
            args.readers, n=args.elements, mode=args.mode, profile=PROFILE
        )
        return [Run("unrouted", workload, profile=workload.profile)]

    def show(self, console: Console, results: Sequence[Result]) -> None:
        (result,) = results
        console.section(
            f"READ BURST: {result.workload.num_readers} readers each get W over the "
            f"REAL TorchStore"
        )
        console.info(
            "real client planning + real controller directory + real InMemoryStore, "
            "on the deterministic virtual-clock engine."
        )
        console.trace(result.trace)
        console.summary(BurstReport(result))
        console.info(
            "no routing selector => %dx fabric (every reader pulls the origin). Installing "
            "a KeySelector (proposed/selector.py) in the controller's locate_volumes -- as "
            "dedup_sim does -- routes later readers to read-through peers and cuts this "
            "toward 1x, with the scenario code above unchanged.",
            result.workload.num_readers,
        )
