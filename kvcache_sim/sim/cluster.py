"""The serving cluster wired onto the *real* TorchStore directory + client.

A **serving instance** is a real storage volume plus a co-located real
``LocalClient`` (built via ``realsim``'s :class:`~realsim.adapters.real_client.RealClientAdapter`).
Its KV-cache pool is the volume's real ``InMemoryStore``; its presence in the
cluster directory is the real ``Controller`` directory
(``keys_to_storage_volumes``) reached through ``realsim``'s
:class:`~realsim.adapters.real_controller.RealControllerAdapter` /
``FakeControllerHandle``.

Mapping (real directory + real types throughout):

* a **KV block** is a plain directory **key** (the prefix-hash chain string);
* **"instance X holds block K"** is the directory entry
  ``keys_to_storage_volumes[K][volume_X]`` -- created by a real, metadata-only
  ``put`` from X's client and read back by the real ``locate_volumes``;
* **publishing** a prefix after prefill is a real ``client.put_batch`` of
  metadata-only carriers (a ``(shape, dtype)`` ``TensorDescriptor`` per block --
  zero real tensor storage), which both writes the carrier into X's real store and
  registers ``K -> volume_X`` via the real ``notify_put_batch``;
* a **remote prefix pull** is a real ``client.get_batch`` driven through
  ``realsim``'s transport seam, so the fabric/storage/RAM cost is charged by the
  real cost model;
* **eviction** removes ``K -> volume_X`` from the real directory via the real
  ``notify_delete_batch`` endpoint.

Multi-client transport seam
---------------------------
``torchstore.client.create_transport_buffer`` is a process-wide module global, so
concurrent instances cannot each install their own factory. Mirroring
``realsim``'s read coordinator, this module installs **one** shared factory for
the whole run that resolves the calling instance's source endpoint from a
:class:`contextvars.ContextVar` set per client operation; ``asyncio`` copies the
context into each task, so the lookup is task-local and deterministic.
"""

from __future__ import annotations

import contextvars
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch

from realsim.adapters.real_client import RealClientAdapter
from realsim.adapters.real_controller import RealControllerAdapter
from realsim.seams.transport import Endpoint, InMemoryTransport, TensorDescriptor
from realsim.seams.volume_handle import FakeVolumeHandle
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile
from sim_common.trace import Trace

from .cost import BYTES_PER_TOKEN

# The real torchstore.client submodule (shadowed on the package by a `client`
# function, so it must be fetched from sys.modules, not attribute access).
_CLIENT_MODULE = sys.modules["torchstore.client"]

# Per-operation source endpoint, read by the shared transport factory so each
# instance charges the right locality for its puts/gets.
_current_src: "contextvars.ContextVar[Endpoint]" = contextvars.ContextVar(
    "kvcache_current_src_endpoint"
)


class Cluster:
    """Serving instances over the real directory + real per-instance clients.

    Args:
        topology: ``instance_id -> Endpoint`` (instance id == its volume id).
        block_tokens: tokens per KV block (the modeled byte size of one block is
            ``block_tokens * BYTES_PER_TOKEN``).
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`.
        trace: shared :class:`~sim_common.trace.Trace` for transfer events.
    """

    def __init__(
        self,
        topology: Dict[str, Endpoint],
        *,
        block_tokens: int,
        profile: MachineProfile = DEFAULT_PROFILE,
        trace: Trace | None = None,
    ) -> None:
        self.topology = topology
        self.ids: List[str] = sorted(topology)
        self.block_tokens = block_tokens
        self.profile = profile
        self.trace = trace if trace is not None else Trace()
        # Structured fabric-byte accounting filled by the transport factory.
        self.fabric_bytes: int = 0

        self.controller = RealControllerAdapter()
        self.handle = self.controller.handle
        self._volumes = {vid: FakeVolumeHandle() for vid in self.ids}
        # One real LocalClient per instance (co-located with its own volume).
        self._adapters: Dict[str, RealClientAdapter] = {
            vid: RealClientAdapter(
                self.handle,
                volume_handles=self._volumes,
                client_volume_id=vid,
                topology=topology,
                profile=profile,
                trace=self.trace,
            )
            for vid in self.ids
        }
        # A metadata-only carrier for one KV block (uint8 -> 1 byte/token). Zero
        # real storage; the transport seam reads its nbytes for the cost model.
        self._block_carrier = TensorDescriptor(
            shape=(block_tokens,), dtype=torch.uint8
        )

    # -- shared, contextvar-aware transport factory ----------------------- #
    def _on_transfer(self, kind, src_id, dst_id, nbytes, cost) -> None:
        """Count cross-instance KV bytes served by a remote peer on a get."""
        if kind == "get" and src_id != dst_id:
            self.fabric_bytes += nbytes

    @contextmanager
    def installed(self) -> Iterator["Cluster"]:
        """Install one shared ``create_transport_buffer`` for the whole run."""
        topo = self.topology
        profile = self.profile
        trace = self.trace
        on_transfer = self._on_transfer

        def factory(storage_volume_ref) -> InMemoryTransport:
            return InMemoryTransport(
                storage_volume_ref,
                src=_current_src.get(),
                dst=topo[storage_volume_ref.volume_id],
                profile=profile,
                trace=trace,
                on_transfer=on_transfer,
            )

        original = _CLIENT_MODULE.create_transport_buffer
        _CLIENT_MODULE.create_transport_buffer = factory
        try:
            yield self
        finally:
            _CLIENT_MODULE.create_transport_buffer = original

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
        _current_src.set(self.topology[inst])
        entries = {k: self._block_carrier for k in keys}
        await self._adapters[inst].client.put_batch(entries)

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
        _current_src.set(self.topology[inst])
        await self._adapters[inst].client.get_batch(present)

    async def evict(self, inst: str, keys: List[str]) -> None:
        """Drop ``key -> volume`` for ``inst`` from the REAL directory (eviction)."""
        if not keys:
            return
        await self.handle.notify_delete_batch.call({inst: list(keys)})
