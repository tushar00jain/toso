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

from typing import Dict, Optional, Tuple

import torch

from domain.llm import DEFAULT_MODEL, DEFAULT_PROFILE, MachineProfile, Model
from proposed.topology import Endpoint
from realsim.mesh import Mesh
from realsim.seams.transport import TensorDescriptor
from sim_common.trace import Trace

from ..data.store import KVStore

__all__ = ["make_store"]


def make_store(
    topology: Dict[str, Endpoint],
    *,
    block_tokens: int,
    profile: MachineProfile = DEFAULT_PROFILE,
    model: Model = DEFAULT_MODEL,
    trace: Optional[Trace] = None,
    real_directory: Optional[bool] = None,
) -> Tuple[Mesh, KVStore]:
    """Build the simulated deployment and the KV store over it."""
    mesh = Mesh(
        topology, profile=profile, trace=trace, real_directory=real_directory
    )
    # A metadata-only carrier for one KV block: a uint8 descriptor whose length IS
    # the block's modeled byte size, so the bytes the transport charges cannot
    # drift from the bytes the scheduler predicted. Zero real storage.
    carrier = TensorDescriptor(
        shape=(model.block_bytes(1, block_tokens),), dtype=torch.uint8
    )
    store = KVStore(
        mesh, block_tokens=block_tokens, carrier=carrier, model=model
    )
    return mesh, store
