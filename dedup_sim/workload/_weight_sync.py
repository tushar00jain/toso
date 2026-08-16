"""Weight sync with a replica to choose between: :class:`WeightSync`.

``n`` trainers each hold the same key -- data-parallel replicas of one set of weights,
so either can serve it -- and ``m`` generators each want it. Every trainer is cross-node
from every generator and no two are nearer than each other, so locality prices the
replicas **identically** and a ranking over distance alone sends every generator to
whichever id sorts first. That is the tie a load term breaks
(:class:`~proposed.selector.Balance`), and this is the smallest workload where breaking
it changes who serves whom.

The fixture is ordinary user code, as :class:`~putget_sim.workload.put_get.PutGetBurst`
is: a put per trainer, then a gather of ``client.get``. It differs from that one in
exactly one thing, which is the thing the scenario is about -- the key has more than one
pre-existing holder. What is *not* modeled: the trainers' step. It would be the same
constant in every run, and what is being compared is the fetch.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import torch

from putget_sim.workload.put_get import KEY, DEFAULT_N
from realsim.runner import WorkItem
from realsim.seams.transport import Endpoint
from realsim.simulation import Simulation
from realsim.run import Workload
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile

__all__ = ["WeightSync"]


def _topology(num_trainers: int, num_generators: int) -> Dict[str, Endpoint]:
    """Trainers on distinct hosts of node ``T``, generators on distinct hosts of ``G``.

    Two nodes and no colocation, so every trainer->generator pair is one cross-node
    hop: the replicas are interchangeable on distance, which is the premise of the
    comparison. A generator is nearer to its fellow generators than to any trainer,
    which is what makes a generator that has read the key through the better source
    for the next one -- by price, with no load term needed.
    """
    topo: Dict[str, Endpoint] = {}
    for i in range(num_trainers):
        topo[f"t{i}"] = Endpoint(id=f"t{i}", host=f"hT{i}", node="T")
    for j in range(num_generators):
        topo[f"g{j}"] = Endpoint(id=f"g{j}", host=f"hG{j}", node="G")
    return topo


class WeightSync(Workload):
    """``m`` generators get one key that ``n`` trainer replicas already hold.

    Args:
        num_trainers: replicas holding the key before the run. Each is an origin for
            fabric accounting, so a read served by any of them is a byte that had to
            cross from a trainer.
        num_generators: readers released together, each simply getting the key.
        n: elements in the payload; the carrier is a ``device="meta"`` tensor, so the
            modeled size is free of any real allocation.
        dtype: element type of the payload.
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`.
    """

    def __init__(
        self,
        num_trainers: int = 2,
        num_generators: int = 2,
        *,
        n: int = DEFAULT_N,
        dtype: torch.dtype = torch.float32,
        profile: Optional[MachineProfile] = None,
    ) -> None:
        self.num_trainers = num_trainers
        self.num_generators = num_generators
        self.profile = profile if profile is not None else DEFAULT_PROFILE
        super().__init__(_topology(num_trainers, num_generators))
        self.trainer_ids = [f"t{i}" for i in range(num_trainers)]
        self.generator_ids = [f"g{j}" for j in range(num_generators)]
        # Real tensor, zero storage: the payload's size is modeled, its bytes never
        # move (``docs/realsim_design.md`` s7).
        self.expected: Any = torch.empty(n, dtype=dtype, device="meta")

    @property
    def payload_bytes(self) -> int:
        """Bytes of one payload -- the 1x union a routed run drives fabric toward."""
        return self.expected.numel() * self.expected.element_size()

    def items(self, sim: Simulation) -> List[WorkItem]:
        """One work item per generator: bind who I am, then get the key."""
        mesh, trace = sim.mesh, sim.trace
        # Every trainer holds the key before the run, so a read served by one of them
        # is an origin byte however the routing spread it.
        sim.origins(*(self.topology[t].id for t in self.trainer_ids))

        def _get(generator_id: str) -> Callable[[], Any]:
            async def call() -> Any:
                mesh.bind_source(generator_id)
                result = await mesh.client(generator_id).get(KEY)
                trace.record(
                    asyncio.get_running_loop().time(),
                    "burst",
                    f"reader {generator_id} done",
                )
                return result

            return call

        return [
            WorkItem(id=gid, release_time=0.0, run=_get(gid))
            for gid in self.generator_ids
        ]

    async def prepare(self, sim: Simulation) -> None:
        """Every trainer publishes the key, in id order, before the generators run."""
        mesh, trace = sim.mesh, sim.trace
        for trainer_id in self.trainer_ids:
            trainer = mesh.adapter(trainer_id)
            with trainer.installed():
                await trainer.client.put(KEY, self.expected)
        trace.record(
            asyncio.get_running_loop().time(),
            "burst",
            f"{self.num_generators} generators get {KEY!r} from "
            f"{self.num_trainers} trainer replicas",
        )
