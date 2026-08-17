"""KV directory verbs over real TorchStore clients: :class:`KVStore`.

A **serving instance** is a storage volume plus a co-located ``LocalClient``. This
module asks a :class:`~proposed.deployment.Deployment` for the client belonging to an
instance and drives ordinary torchstore APIs on it. Under simulation the deployment
resolves an instance id to one of many in-process clients; a real one has a single
client and ignores the id. Either way the code below is the same, and nothing here
imports the simulator.

Only the KV verbs that move bytes or change the directory are here.

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
The verbs take the blocks; the accelerator produces them
(:meth:`kvcache_sim.data._compute.Accelerator.prefill`) and what they *are* is that
implementation's answer -- ``device="meta"`` tensors under simulation, attention output
in a deployment. This module makes no assumption about either, including the byte count
a transfer is priced against.

**Eviction is not here.** A volume drops its own coldest keys when a put does not fit
and tells the directory itself, so a verb here that deregistered a key would be half an
eviction: the entry would go and the bytes would stay
(``realsim.seams._retention``).

**Reading the directory is not here either.** A ``locate`` decides nothing and moves
nothing, so per-instance prefix presence is a control-plane sensor
(:class:`proposed.sensors.DirectorySensor`) -- including re-reading it to see whether a
planned pull is still available. :meth:`KVStore.fetch` asks for what it was told to and
lets the store answer.

A fetch is the one verb whose source matters, so it asks the control plane which peers
should serve it and passes the answer to the client as a preference: the store applies
a value rather than consulting anybody.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from proposed import ControlPlane, Deployment, StorageFull

__all__ = ["KVStore"]


class KVStore:
    """Publish / reuse / fetch, over a deployment's real clients.

    Deliberately absent: what a KV block *is*, and how big one is. The one thing this
    class insists on is that a caller publishing ``n`` keys hands it ``n`` blocks -- an
    arity check, not a size premise.

    Args:
        deployment: the deployment these instances run against.
    """

    def __init__(self, deployment: Deployment) -> None:
        self.deployment = deployment

    # -- real data-plane ops (presence + fabric cost) --------------------- #
    async def publish(
        self, inst: str, keys: List[str], blocks: Sequence[torch.Tensor]
    ) -> bool:
        """Publish ``keys`` on ``inst`` via a real ``put_batch`` of ``blocks``.

        Writes the KV into ``inst``'s real store (co-located -> zero fabric) and
        registers ``key -> volume`` in the real directory. ``blocks[i]`` is the KV of
        ``keys[i]``.

        A **cache fill**, so it is allowed to fail: ``False`` when the instance has no
        room even after evicting what it could. The request has already been served,
        and the only loss is that nobody reuses this prefix.

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

        A local prefix hit never reaches the store, so nothing is fetched and nothing
        is charged. On its own evidence the volume would see those blocks as untouched
        and drop the hottest prefix in the run; this is the read it could not see.
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

        Drives the real client planning core + transport seam, so storage / RAM /
        network cost is charged by the real cost model against the peer that serves the
        blocks -- the one the control plane priced (:meth:`_sources`), handed to the
        client as a preference.

        Answers with the KV, one block per key in the order asked for (``get_batch``
        answers with a dict, so the prefix order is re-imposed here). Summing the
        tensors gives the same byte count the transport charged.

        The pull runs after the prefill queue, by which time the peer may have dropped
        some of the planned blocks. ``get_batch`` is all-or-nothing, so a half-usable
        prefix surfaces as a ``KeyError`` and the caller decides. Filtering the batch
        down to what survived would answer that here, silently, and charge the caller
        for a reuse it did not get.
        """
        if not keys:
            return []
        prefer = await self._sources(inst, keys)
        got = await self.deployment.client_for(
            inst, prefer=prefer
        ).get_batch(list(keys))
        return [got[k] for k in keys]
