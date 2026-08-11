"""The kvcache workload, and the wiring a run installs around it.

Two separate things, deliberately:

* :class:`KVWorkload` is *the work* -- a request stream, one
  :class:`~realsim.runner.WorkItem` per request at its arrival time. It builds no
  store, no scheduler and no plane;
* :func:`coordinator` and :func:`serving_plane` are the *capability wiring*, one
  per plane: the scheduler over the view, and the store plus the
  :class:`~kvcache_sim.data.serving.ServingPlane` over it. Both are factories
  because they reach for the view, the mesh and the ledger, none of which exists
  before the stack does.

They are two functions because they are two services. The plane factory does not
build the scheduler; it takes ``sim.coordinator_handle``, the handle
:meth:`realsim.run.Run.execute` put in front of whatever :func:`coordinator`
returned. A scenario names both on a :class:`~realsim.run.Run`: same workload,
different wiring, which is exactly what "cache-aware vs load-balance" means.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from domain import DEFAULT_MODEL, DEFAULT_PROFILE, Model
from proposed import Endpoint
from realsim.runner import WorkItem
from realsim.seams.transport import TensorDescriptor
from realsim.simulation import Simulation
from realsim.run import Workload

from ..control.scheduler import CacheAwareScheduler, LoadBalanceScheduler
from ..data.serving import ServingPlane
from ..data.store import KVStore

#: Tokens per KV block. Fixed for every scenario so runs stay comparable.
BLOCK_TOKENS = 512

__all__ = ["BLOCK_TOKENS", "coordinator", "KVWorkload", "serving_plane"]


def _sim_block_carrier(
    block_tokens: int = BLOCK_TOKENS, model: Model = DEFAULT_MODEL
):
    """What one KV block is stored as **under simulation**.

    A metadata-only carrier: a uint8 descriptor whose length *is* the block's
    modeled byte size, so the bytes the transport charges cannot drift from the
    bytes the scheduler predicted. Zero real storage.

    This is the one piece a real deployment chooses differently -- it stores the
    KV tensors -- which is why it lives with the run rather than in
    :mod:`kvcache_sim.data.store`.
    """
    return TensorDescriptor(
        shape=(model.block_bytes(1, block_tokens),), dtype=torch.uint8
    )


class KVWorkload(Workload):
    """A request stream over a set of serving instances.

    One work item per request, released at its arrival time. Which scheduler
    serves them is not this object's business -- see :func:`serving_plane`.
    """

    def __init__(self, topology: Dict[str, Endpoint], requests) -> None:
        super().__init__(topology)
        self.requests = requests

    def items(self, sim: Simulation) -> List[WorkItem]:
        """One item per request; the serving plane runs each one's lifecycle."""
        return [
            WorkItem(id=r.id, release_time=r.arrival, payload=r)
            for r in self.requests
        ]


def coordinator(
    kind: str,
    *,
    balance_threshold: float = 1.5,
    replicate: bool = True,
    capacity: Optional[int] = None,
    slo_ttft: float = float("inf"),
    slo_tbt: float = float("inf"),
    simulate_decode: bool = False,
    max_batch: int = 8,
    prefill_pool: Optional[List[str]] = None,
    decode_pool: Optional[List[str]] = None,
    early_rejection: str = "off",
) -> object:
    """This run's **control plane**, as an object a scenario can just declare.

    ``kind`` is ``"cache_aware"`` (the coordinator under test) or
    ``"load_balance"`` (the baseline). Knobs only: the stack's ports arrive later
    through :meth:`~kvcache_sim.control.scheduler._Base.attach`, which is what
    lets this be a value rather than a factory the harness must call at the right
    moment. The :class:`~realsim.run.Run` installs it in the directory (it answers
    the store's routing question) *and* fronts it with a
    :class:`~realsim.seams.coordinator_handle.CoordinatorHandle` (it decides compute
    placement) -- one object, reached through the seam of whichever service is
    asking.
    """
    if kind not in ("cache_aware", "load_balance"):
        raise ValueError(f"unknown scheduler kind {kind!r}")
    knobs = dict(
        block_tokens=BLOCK_TOKENS,
        capacity=capacity,
        profile=DEFAULT_PROFILE,
        slo_ttft=slo_ttft,
        slo_tbt=slo_tbt,
        simulate_decode=simulate_decode,
        max_batch=max_batch,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        early_rejection=early_rejection,
    )
    if kind == "cache_aware":
        return CacheAwareScheduler(
            balance_threshold=balance_threshold, replicate=replicate, **knobs
        )
    return LoadBalanceScheduler(**knobs)


def serving_plane(
    *,
    coupled: bool = False,
    simulate_decode: bool = False,
    max_batch: int = 8,
    decode_pool: Optional[List[str]] = None,
) -> Callable[[Simulation], ServingPlane]:
    """Build the factory for this run's **data plane**.

    ``coupled`` says whether prefill shares the decode instances' compute -- a
    deployment fact, so it belongs to the serving plane, not to the scheduler.
    The decode settings are passed here *and* to :func:`coordinator` from the one
    scenario that declares them, rather than one reading them off the other.
    """

    def build(sim: Simulation) -> ServingPlane:
        # The simulation *is* the deployment: it vends the client for an instance
        # and holds the directory. All the run adds is the block carrier.
        store = KVStore.for_deployment(
            sim.mesh, block_tokens=BLOCK_TOKENS, carrier=_sim_block_carrier()
        )
        return ServingPlane(
            store, sim.coordinator_handle, trace=sim.trace, metrics=sim.ledger,
            coupled=coupled,
            simulate_decode=simulate_decode,
            decode_ids=sorted(decode_pool) if decode_pool else sim.ids,
            max_batch=max_batch,
            profile=DEFAULT_PROFILE,
            model=DEFAULT_MODEL,
        )

    return build
