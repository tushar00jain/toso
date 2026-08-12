"""KV directory verbs over real TorchStore clients: :class:`KVStore`.

A **serving instance** is a storage volume plus a co-located ``LocalClient``.
Obtaining that client is not this module's business: it asks a
:class:`~proposed.deployment.Deployment` for the one belonging to an instance and
drives ordinary torchstore APIs on it. Under simulation the deployment resolves an
instance id to one of many in-process clients; a real one has a single client and
ignores the id. Either way what follows is the same code -- which is the point:
nothing here imports the simulator.

This module holds only the KV verbs that move bytes or change the directory.

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
  the scheduler predicted against.

**Eviction is not here.** A volume drops its own coldest keys when a put does not
fit and tells the directory itself, so a verb here that deregistered a key would
be half an eviction: the entry would go and the bytes would stay. Which key to
drop is the volume's (``realsim.seams._retention``), and saying so is the
volume's too.

Reading the directory is *not* here either: a ``locate`` decides nothing and moves
nothing, so per-instance prefix presence is a control-plane view
(:class:`kvcache_sim.control._view.KVView`). That includes re-reading it to see
whether a planned pull is still available -- :meth:`KVStore.fetch` asks for what it
was told to and lets the store answer.
"""

from __future__ import annotations

from typing import List, Optional

from proposed import Deployment, StorageFull

from domain import DEFAULT_MODEL, Model

__all__ = ["KVStore"]


class KVStore:
    """What a KV block is stored as, and the store calls that follow from it.

    The verbs are thin because the mapping is thin -- a block is a key, publishing
    is a ``put_batch``. What is not thin, and is why this is an object rather than
    three calls in the serving loop, is the premise it enforces at construction:
    whatever a block is stored as must occupy the bytes
    :meth:`~domain.llm.Model.block_bytes` predicts, because that is the number the
    scheduler prices every fetch against. A carrier that disagrees would make the
    run charge for one size and route on another, with nothing else noticing.

    Args:
        deployment: the :class:`~proposed.deployment.Deployment` these instances
            run against; it vends the client for an instance id. A real one vends
            its single client and ignores the id.
        block_tokens: tokens per KV block.
        carrier: what one block is stored as, and the piece that differs between a
            simulated run (an allocation-free descriptor) and a real one (the KV
            tensors). Supplied by the run: *what* to store is not a data-plane
            decision.
        model: served-model :class:`~domain.llm.Model`, which sets how many bytes
            one KV block occupies and therefore what ``carrier`` must measure.

    Raises:
        ValueError: if the carrier is not the size the model predicts.
    """

    def __init__(
        self,
        deployment: Deployment,
        *,
        block_tokens: int,
        carrier,
        model: Model = DEFAULT_MODEL,
    ) -> None:
        want = model.block_bytes(1, block_tokens)
        if carrier.nbytes != want:
            raise ValueError(
                f"block carrier is {carrier.nbytes}B but the model predicts "
                f"{want}B for {block_tokens} tokens: every fetch would be priced "
                f"against the wrong byte count"
            )
        self.deployment = deployment
        self.block_tokens = block_tokens
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
    async def publish(self, inst: str, keys: List[str]) -> bool:
        """Publish ``keys`` on ``inst`` via a real metadata-only ``put_batch``.

        Writes the ``(shape, dtype)`` carrier into ``inst``'s real store (co-located
        -> zero fabric) and registers ``key -> volume`` in the real directory. No
        real tensor is allocated.

        A **cache fill**, so it is allowed to fail: ``False`` when the instance has
        no room for these blocks even after evicting what it could. A KV cache that
        cannot fit something does not cache it -- the request has already been
        served, and the only loss is that nobody reuses this prefix. Letting the
        store refuse and treating that as fatal would make a bounded cache unusable
        the moment one request's working set exceeded a volume.
        """
        if not keys:
            return True
        entries = {k: self._block_carrier for k in keys}
        try:
            await self.deployment.client_for(inst).put_batch(entries)
        except StorageFull:
            return False
        return True

    async def reuse(self, inst: str, keys: List[str]) -> None:
        """Tell ``inst``'s volume it just served ``keys`` from what it already had.

        A local prefix hit never reaches the store -- the instance has the blocks, so
        nothing is fetched and nothing is charged. The volume is the one deciding
        what to drop when it fills up, and on its own evidence those blocks look
        untouched, so it would drop the hottest prefix in the run. This is the read
        it could not see.
        """
        if not keys:
            return
        await self.deployment.volume_handle(inst).touch.call_one(list(keys))

    async def fetch(self, inst: str, keys: List[str]) -> None:
        """Pull ``keys`` into ``inst`` via a real ``get_batch`` (charges fabric).

        Drives the real client planning core + transport seam, so the storage /
        RAM / network cost is charged by the real cost model against the peer that
        actually serves the blocks.

        The peer that serves it is the one the control plane priced: the installed
        policy *is* that control plane, so it narrows the directory answer to the
        decision it already made. Nothing has to be threaded through this call.

        A routing decision is made when the request arrives, but the pull runs
        later (after the prefill queue), by which time the peer may have dropped
        some of the planned blocks. ``get_batch`` is all-or-nothing, so that
        surfaces here as a ``KeyError`` and the caller decides -- which is the
        honest place for it: whether a half-usable prefix is worth pulling is a
        question about the request, not about the store. Filtering the batch down
        to what survived would answer it here, silently, and charge the caller for
        a reuse it did not get.
        """
        if not keys:
            return
        await self.deployment.client_for(inst).get_batch(list(keys))
