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
  ``keys_to_storage_volumes[K][volume_X]`` -- created by a real ``put`` from X's
  client and read back by the real ``locate_volumes``;
* **publishing** a prefix after prefill (:meth:`KVStore.publish`) is a real
  ``client.put_batch`` of the KV tensors the caller hands over, which both writes
  them into X's store and registers ``K -> volume_X`` via the real
  ``notify_put_batch``;
* a **remote prefix pull** (:meth:`KVStore.fetch`) is a real ``client.get_batch``,
  so a simulated run charges fabric/storage/RAM for it through the same cost model
  the scheduler predicted against, and the caller gets the KV back.

Three verbs over whatever it is handed
--------------------------------------
The verbs take the blocks; what produces them is the accelerator
(:meth:`kvcache_sim.data._compute.Accelerator.prefill`) and what they *are* is that
implementation's answer -- zero-storage ``device="meta"`` tensors under simulation,
attention output in a deployment. This module is indifferent, which is what lets it
be three calls and no premises. In particular the byte count every transfer is
priced against belongs with the thing that computes the KV, not with the thing that
moves it.

**Eviction is not here.** A volume drops its own coldest keys when a put does not
fit and tells the directory itself, so a verb here that deregistered a key would be
half an eviction: the entry would go and the bytes would stay. Which key to drop is
the volume's (``realsim.seams._retention``), and saying so is the volume's too.

**Reading the directory is not here either**: a ``locate`` decides nothing and moves
nothing, so per-instance prefix presence is a control-plane view
(:class:`kvcache_sim.control._view.KVView`). That includes re-reading it to see
whether a planned pull is still available -- :meth:`KVStore.fetch` asks for what it
was told to and lets the store answer.

What *is* here is the asking. A fetch is the one verb whose source matters, so it
asks the control plane which peers should serve it and passes the answer to the
client as a preference. Two ordinary calls in one order, with nothing installed
anywhere: the plane is reached over a port like any other service, and the store
applies a value rather than consulting anybody.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from proposed import ControlPlane, Deployment, StorageFull

__all__ = ["KVStore"]


class KVStore:
    """Publish / reuse / fetch, over a deployment's real clients.

    Three verbs and one field. The verbs are thin because the mapping is thin -- a
    block is a key, publishing is a ``put_batch`` -- and they are an object rather
    than three functions only so that the deployment is named once instead of at
    every call site.

    Deliberately absent: what a KV block *is*, and how big one is (see the module
    docstring). The one thing this class insists on is that a caller publishing
    ``n`` keys hands it ``n`` blocks -- an arity check, not a size premise.

    Args:
        deployment: the :class:`~proposed.deployment.Deployment` these instances
            run against; it vends the client for an instance id. A real one vends
            its single client and ignores the id.
    """

    def __init__(self, deployment: Deployment) -> None:
        self.deployment = deployment

    # -- real data-plane ops (presence + fabric cost) --------------------- #
    async def publish(
        self, inst: str, keys: List[str], blocks: Sequence[torch.Tensor]
    ) -> bool:
        """Publish ``keys`` on ``inst`` via a real ``put_batch`` of ``blocks``.

        Writes the KV into ``inst``'s real store (co-located -> zero fabric) and
        registers ``key -> volume`` in the real directory. ``blocks[i]`` is the KV
        of ``keys[i]``; the caller is the host that holds both.

        A **cache fill**, so it is allowed to fail: ``False`` when the instance has
        no room for these blocks even after evicting what it could. The request has
        already been served, and the only loss is that nobody reuses this prefix.

        Raises:
            ValueError: if there is not exactly one block per key. Loud rather than
                zipped-to-the-shorter: too few blocks registers a prefix the
                directory routes reads to and the volume does not hold, too many
                drops KV this host computed and paid for.
        """
        if not keys:
            return True
        if len(keys) != len(blocks):
            raise ValueError(
                f"publishing {len(keys)} keys with {len(blocks)} KV blocks: a "
                f"block is a key, so the two are the same list seen twice and a "
                f"mismatch means the caller lost track of which is which"
            )
        entries = dict(zip(keys, blocks))
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

    async def _sources(self, inst: str, keys: List[str]) -> Optional[Tuple[str, ...]]:
        """Which peers should serve ``keys`` for ``inst``, best first.

        One call to the control plane, which answers from what it already priced for
        this caller and does not answer until those peers are usable. ``None`` -- the
        directory's own order -- when the run stands up no control plane.
        """
        control: Optional[ControlPlane] = self.deployment.control_plane_handle
        if control is None:
            return None
        selection = await control.sources.call_one(list(keys), inst)
        return selection.sources

    async def fetch(self, inst: str, keys: List[str]) -> List[torch.Tensor]:
        """Pull ``keys`` into ``inst`` via a real ``get_batch`` (charges fabric).

        Drives the real client planning core + transport seam, so the storage / RAM
        / network cost is charged by the real cost model against the peer that
        actually serves the blocks. That peer is the one the control plane priced:
        this asks it for the ranking (:meth:`_sources`) and hands that to the client as
        a preference, so the read itself is an ordinary ``get_batch`` and the store
        decides nothing.

        Answers with the KV, one block per key in the order asked for
        (``get_batch`` answers with a dict, so the prefix order is re-imposed here).
        A caller that wants the bytes can sum them off the tensors, which is the
        same number the transport charged.

        The pull runs after the prefill queue, by which time the peer may have
        dropped some of the planned blocks. ``get_batch`` is all-or-nothing, so that
        surfaces as a ``KeyError`` and the caller decides: whether a half-usable
        prefix is worth pulling is a question about the request. Filtering the batch
        down to what survived would answer it here, silently, and charge the caller
        for a reuse it did not get.
        """
        if not keys:
            return []
        prefer = await self._sources(inst, keys)
        got = await self.deployment.client_for(
            inst, prefer=prefer
        ).get_batch(list(keys))
        return [got[k] for k in keys]
