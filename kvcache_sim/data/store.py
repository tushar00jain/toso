"""KV directory verbs over the *real* TorchStore clients: :class:`KVStore`.

A **serving instance** is a real storage volume plus a co-located real
``LocalClient``. All of that wiring -- the real ``Controller`` directory behind
:class:`~realsim.seams.controller_handle.FakeControllerHandle`, one
:class:`~realsim.seams.volume_handle.FakeVolumeHandle` and one
:class:`~realsim.adapters.real_client.RealClientAdapter` per instance, the shared
:class:`~sim_common.resources.ResourceRegistry`, and the single shared
``create_transport_buffer`` substitution -- is generic multi-client machinery, so
it comes from :class:`realsim.mesh.Mesh` via :class:`realsim.mesh.MeshView`. This
module holds only the three KV verbs that move bytes or change the directory.

Mapping (real directory + real types throughout):

* a **KV block** is a plain directory **key** (the prefix-hash chain string);
* **"instance X holds block K"** is the directory entry
  ``keys_to_storage_volumes[K][volume_X]`` -- created by a real, metadata-only
  ``put`` from X's client and read back by the real ``locate_volumes``;
* **publishing** a prefix after prefill (:meth:`KVStore.publish`) is a real
  ``client.put_batch`` of metadata-only carriers (a ``(shape, dtype)``
  ``TensorDescriptor`` per block -- zero real tensor storage), which both writes
  the carrier into X's real store and registers ``K -> volume_X`` via the real
  ``notify_put_batch``;
* a **remote prefix pull** (:meth:`KVStore.fetch`) is a real ``client.get_batch``
  driven through ``realsim``'s transport seam, so the fabric/storage/RAM cost is
  charged by the real cost model;
* **eviction** (:meth:`KVStore.evict`) removes ``K -> volume_X`` from the real
  directory via the real ``notify_delete_batch`` endpoint.

Reading the directory is *not* here: a ``locate`` decides nothing and moves
nothing, so per-instance prefix presence is a control-plane view
(:class:`kvcache_sim.control.view.KVView`).
"""

from __future__ import annotations

from typing import List

from proposed.deployment import Deployment

from domain.llm import DEFAULT_MODEL, Model


class KVStore:
    """The KV data plane's three verbs over real per-instance clients.

    Args:
        topology: ``instance_id -> Endpoint`` (instance id == its volume id).
        block_tokens: tokens per KV block.
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`.
            Also supplies each volume's byte capacity.
        model: served-model :class:`~domain.llm.Model`, which sets how many bytes
            one KV block occupies. The block carrier is sized from it (see
            :attr:`block_nbytes`) so the bytes the transport charges are always
            the bytes :meth:`~domain.llm.Model.block_bytes` predicts.
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
        deployment: Deployment,
        *,
        block_tokens: int,
        carrier,
        model: Model = DEFAULT_MODEL,
    ) -> None:
        self.deployment = deployment
        self.block_tokens = block_tokens
        self.model = model
        # What one KV block is, as far as the store is concerned. Supplied by the
        # workload rather than built here: *what* to store is not a data-plane
        # decision, and it is the piece that differs between a simulated run
        # (an allocation-free carrier) and a real one (the KV tensors).
        self._block_carrier = carrier

    @property
    def block_nbytes(self) -> int:
        """Bytes the data plane actually moves for one KV block.

        The authoritative byte count: this is the carrier's size, i.e. what the
        transport seam charges. It must equal ``model.block_bytes(1,
        block_tokens)`` -- the value the scheduler predicts a fetch against --
        which ``kvcache_sim/tests/test_cost_premises.py`` asserts.
        """
        return self._block_carrier.nbytes

    # -- real data-plane ops (presence + fabric cost) --------------------- #
    async def publish(self, inst: str, keys: List[str]) -> None:
        """Publish ``keys`` on ``inst`` via a real metadata-only ``put_batch``.

        Writes the ``(shape, dtype)`` carrier into ``inst``'s real store (co-located
        -> zero fabric) and registers ``key -> volume`` in the real directory. No
        real tensor is allocated.
        """
        if not keys:
            return
        entries = {k: self._block_carrier for k in keys}
        await self.deployment.client_for(inst).put_batch(entries)

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
        located = await self.deployment.controller_handle.locate_volumes.call_one(
            list(keys), missing_ok=True
        )
        present = [k for k in keys if k in located]
        if not present:
            return
        await self.deployment.client_for(inst).get_batch(present)

    async def evict(self, inst: str, keys: List[str]) -> None:
        """Drop ``key -> volume`` for ``inst`` from the REAL directory (eviction)."""
        if not keys:
            return
        await self.deployment.controller_handle.notify_delete_batch.call(
            {inst: list(keys)}
        )
