"""Wiring a KV deployment for a simulated run.

:class:`~kvcache_sim.data.store.KVStore` is application code: it takes a
:class:`~proposed.deployment.Deployment` and calls ordinary torchstore APIs on it.
Something still has to *be* that deployment, and under simulation that is a
:class:`~realsim.mesh.Mesh`. Building it is a run concern, not a data-plane one,
so it happens here -- which is also why the block carrier is chosen here: a
simulated run stores an allocation-free descriptor where a real one would store
the KV tensors.
"""

from __future__ import annotations

import torch

from domain import DEFAULT_MODEL, Model
from realsim.seams.transport import TensorDescriptor
from realsim.simulation import Simulation

from ..data.store import KVStore

__all__ = ["make_store"]


def make_store(
    sim: Simulation,
    *,
    block_tokens: int,
    model: Model = DEFAULT_MODEL,
) -> KVStore:
    """Build the KV store over an assembled :class:`~realsim.simulation.Simulation`.

    The simulation *is* the deployment: it vends the client for an instance and
    holds the directory. All this adds is the block carrier, which is the one
    piece a real run would choose differently.
    """
    # A metadata-only carrier for one KV block: a uint8 descriptor whose length IS
    # the block's modeled byte size, so the bytes the transport charges cannot
    # drift from the bytes the scheduler predicted. Zero real storage.
    carrier = TensorDescriptor(
        shape=(model.block_bytes(1, block_tokens),), dtype=torch.uint8
    )
    return KVStore(
        sim.mesh, block_tokens=block_tokens, carrier=carrier, model=model
    )
