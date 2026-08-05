"""KV-cache verbs over the *real* TorchStore directory + clients.

A **serving instance** is a real storage volume plus a co-located real
``LocalClient``. All of that wiring -- the real ``Controller`` directory behind
:class:`~realsim.seams.controller_handle.FakeControllerHandle`, one
:class:`~realsim.seams.volume_handle.FakeVolumeHandle` and one
:class:`~realsim.adapters.real_client.RealClientAdapter` per instance, the shared
:class:`~sim_common.resources.ResourceRegistry`, and the single shared
``create_transport_buffer`` substitution -- is generic multi-client machinery, so
it comes from :class:`realsim.mesh.Mesh`. This module holds only what is specific
to KV caching: the four directory verbs the scheduler speaks in.

Mapping (real directory + real types throughout):

* a **KV block** is a plain directory **key** (the prefix-hash chain string);
* **"instance X holds block K"** is the directory entry
  ``keys_to_storage_volumes[K][volume_X]`` -- created by a real, metadata-only
  ``put`` from X's client and read back by the real ``locate_volumes``;
* **publishing** a prefix after prefill (:meth:`Cluster.publish`) is a real
  ``client.put_batch`` of metadata-only carriers (a ``(shape, dtype)``
  ``TensorDescriptor`` per block -- zero real tensor storage), which both writes
  the carrier into X's real store and registers ``K -> volume_X`` via the real
  ``notify_put_batch``;
* a **remote prefix pull** (:meth:`Cluster.fetch`) is a real ``client.get_batch``
  driven through ``realsim``'s transport seam, so the fabric/storage/RAM cost is
  charged by the real cost model;
* **eviction** (:meth:`Cluster.evict`) removes ``K -> volume_X`` from the real
  directory via the real ``notify_delete_batch`` endpoint;
* the scheduler's core query (:meth:`Cluster.prefix_lengths`) is derived from a
  real ``locate_volumes``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

import torch

from realsim.mesh import Mesh
from realsim.seams.transport import Endpoint, TensorDescriptor
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile
from sim_common.resources import ResourceRegistry
from sim_common.trace import Trace


class Cluster:
    """Serving instances over the real directory + real per-instance clients.

    Args:
        topology: ``instance_id -> Endpoint`` (instance id == its volume id).
        block_tokens: tokens per KV block (the modeled byte size of one block is
            ``block_tokens * BYTES_PER_TOKEN``).
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`.
            Also supplies each volume's byte capacity.
        trace: shared :class:`~sim_common.trace.Trace` for transfer events.
        real_directory: controller directory backing (``None`` -> the ambient
            :data:`sim_common.config.SimConfig.real_directory`, default real
            ``Trie``; ``False`` -> the lightweight dict shim). Changes no metric.

    The network/storage contention model is read ambiently from
    :data:`sim_common.config.SimConfig.contention` (default ``"none"``). The mesh
    builds one shared :class:`~sim_common.resources.ResourceRegistry` and injects
    it into every instance's transport, so concurrent cross-instance pulls share a
    hot peer's egress / a volume's read channel. Unlike ``real_directory`` a
    non-``"none"`` mode DOES change timing.
    """

    def __init__(
        self,
        topology: Dict[str, Endpoint],
        *,
        block_tokens: int,
        profile: MachineProfile = DEFAULT_PROFILE,
        trace: Trace | None = None,
        real_directory: Optional[bool] = None,
    ) -> None:
        self.mesh = Mesh(
            topology,
            profile=profile,
            trace=trace,
            real_directory=real_directory,
        )
        self.block_tokens = block_tokens
        # A metadata-only carrier for one KV block (uint8 -> 1 byte/token). Zero
        # real storage; the transport seam reads its nbytes for the cost model.
        self._block_carrier = TensorDescriptor(
            shape=(block_tokens,), dtype=torch.uint8
        )

    # -- the mesh's shared pieces, surfaced for the scheduler/driver -------- #
    @property
    def topology(self) -> Dict[str, Endpoint]:
        """``instance_id -> Endpoint`` for transfer-cost locality."""
        return self.mesh.topology

    @property
    def ids(self) -> List[str]:
        """Instance ids, sorted."""
        return self.mesh.ids

    @property
    def handle(self):
        """The real ``Controller`` directory behind the actor surface."""
        return self.mesh.handle

    @property
    def trace(self) -> Trace:
        """The run's shared :class:`~sim_common.trace.Trace`."""
        return self.mesh.trace

    @property
    def profile(self) -> MachineProfile:
        """The target-machine :class:`~sim_common.cost_model.MachineProfile`."""
        return self.mesh.profile

    @property
    def registry(self) -> ResourceRegistry:
        """The run's shared :class:`~sim_common.resources.ResourceRegistry`."""
        return self.mesh.registry

    @contextmanager
    def installed(self) -> Iterator["Cluster"]:
        """Install the mesh's shared transport factory for the whole run."""
        with self.mesh.installed():
            yield self

    # -- real directory reads --------------------------------------------- #
    async def prefix_lengths(self, block_keys: List[str]) -> Dict[str, int]:
        """Per-instance leading-prefix length held, read from the REAL directory.

        The cache-aware scheduler's core query: ``instance_id -> how many leading
        blocks of block_keys it holds contiguously``. Computed from the real
        ``locate_volumes`` result (``{key -> {volume_id -> StorageInfo}}``);
        instances holding none of the first block are omitted.
        """
        if not block_keys:
            return {}
        located = await self.handle.locate_volumes.call_one(
            list(block_keys), missing_ok=True
        )
        counts: Dict[str, int] = {}
        for inst in sorted(located.get(block_keys[0], {})):
            n = 0
            for k in block_keys:
                if inst in located.get(k, {}):
                    n += 1
                else:
                    break
            counts[inst] = n
        return counts

    # -- real data-plane ops (presence + fabric cost) --------------------- #
    async def publish(self, inst: str, keys: List[str]) -> None:
        """Publish ``keys`` on ``inst`` via a real metadata-only ``put_batch``.

        Writes the ``(shape, dtype)`` carrier into ``inst``'s real store (co-located
        -> zero fabric) and registers ``key -> volume`` in the real directory. No
        real tensor is allocated.
        """
        if not keys:
            return
        self.mesh.bind_source(inst)
        entries = {k: self._block_carrier for k in keys}
        await self.mesh.client(inst).put_batch(entries)

    async def fetch(self, inst: str, keys: List[str]) -> None:
        """Pull ``keys`` into ``inst`` via a real ``get_batch`` (charges fabric).

        Drives the real client planning core + transport seam, so the peer that
        the real directory reports serves the blocks and the storage/RAM/network
        cost is charged by the real cost model.

        A routing decision is made when the request arrives, but the pull runs
        later (after the prefill queue), by which time a peer may have *evicted*
        some of the planned blocks. Like a real read-through, we fetch only the
        blocks still present in the directory; any that vanished are simply
        recomputed by prefill. The presence re-check and the ``get_batch`` locate
        run back-to-back without yielding the loop, so they observe the same state.
        """
        if not keys:
            return
        located = await self.handle.locate_volumes.call_one(
            list(keys), missing_ok=True
        )
        present = [k for k in keys if k in located]
        if not present:
            return
        self.mesh.bind_source(inst)
        await self.mesh.client(inst).get_batch(present)

    async def evict(self, inst: str, keys: List[str]) -> None:
        """Drop ``key -> volume`` for ``inst`` from the REAL directory (eviction)."""
        if not keys:
            return
        await self.handle.notify_delete_batch.call({inst: list(keys)})
