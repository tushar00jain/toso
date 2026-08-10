"""KV directory verbs over real TorchStore clients: :class:`KVStore`.

A **serving instance** is a storage volume plus a co-located ``LocalClient``.
Obtaining that client is not this module's business: it asks a
:class:`~proposed.deployment.Deployment` for the one belonging to an instance and
drives ordinary torchstore APIs on it. Under simulation the deployment resolves an
instance id to one of many in-process clients; a real one has a single client and
ignores the id. Either way what follows is the same code -- which is the point:
nothing here imports the simulator.

This module holds only the three KV verbs that move bytes or change the directory.

Mapping (real directory + real types throughout):

* a **KV block** is a plain directory **key** (the prefix-hash chain string);
* **"instance X holds block K"** is the directory entry
  ``keys_to_storage_volumes[K][volume_X]`` -- created by a real, metadata-only
  ``put`` from X's client and read back by the real ``locate_volumes``;
* **publishing** a prefix after prefill (:meth:`KVStore.publish`) is a real
  ``client.put_batch`` of one carrier per block, which both writes the carrier
  into X's store and registers ``K -> volume_X`` via the real
  ``notify_put_batch``. *What* a block is stored as is chosen by the run, not
  here: a simulated run supplies an allocation-free ``(shape, dtype)`` descriptor
  where a deployment would supply the KV tensors;
* a **remote prefix pull** (:meth:`KVStore.fetch`) is a real ``client.get_batch``,
  so a simulated run charges fabric/storage/RAM for it through the same cost model
  the scheduler predicted against;
* **eviction** (:meth:`KVStore.evict`) removes ``K -> volume_X`` from the real
  directory via the real ``notify_delete_batch`` endpoint.

Reading the directory is *not* here: a ``locate`` decides nothing and moves
nothing, so per-instance prefix presence is a control-plane view
(:class:`kvcache_sim.control.view.KVView`).
"""

from __future__ import annotations

from typing import List

from proposed import Deployment

from domain import DEFAULT_MODEL, Model


class KVStore:
    """The KV data plane's three verbs over real per-instance clients.

    Args:
        deployment: the :class:`~proposed.deployment.Deployment` these instances
            run against; it vends the client for an instance id.
        block_tokens: tokens per KV block.
        carrier: what one block is stored as. Supplied by the run (see
            ``kvcache_sim/workload/deploy.py``) because it is the piece that
            differs between a simulated run and a real one.
        model: served-model :class:`~domain.llm.Model`, which sets how many bytes
            one KV block occupies. The carrier must be sized from it (see
            :attr:`block_nbytes`) so the bytes moved are the bytes
            :meth:`~domain.llm.Model.block_bytes` predicts.
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
