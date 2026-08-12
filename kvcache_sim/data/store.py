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
This object used to own a fourth thing: *what a KV block is*. It was constructed
with a "carrier" -- one stand-in value it wrote under every key -- plus the tokens
per block and the served model, so that it could check the carrier was the size the
model predicts, and it answered ``block_nbytes`` for anyone pricing a transfer.

None of that is a store's business, and holding it had two costs. It made the store
the authority on a byte count it never computes, one object away from the thing that
does (a forward pass produces KV; a ``put`` moves it). And because one carrier stood
in for every block, a run could only ever store the same value everywhere -- which
is how the simulated data plane ended up on torchstore's *object* path
(``put_batch`` types its values: a ``Tensor`` takes ``Request.from_any``, anything
else takes ``Request.from_objects``), exercising a code path no KV deployment uses.

So the verbs take the blocks. What produces them is the accelerator
(:meth:`kvcache_sim.data._compute.Accelerator.prefill`), what they *are* is that
implementation's answer -- zero-storage ``device="meta"`` tensors under simulation,
attention output in a deployment -- and this module is indifferent, which is what
lets it be three calls and no premises.

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

from typing import List, Sequence

import torch

from proposed import Deployment, StorageFull

__all__ = ["KVStore"]


class KVStore:
    """Publish / reuse / fetch, over a deployment's real clients.

    Three verbs and one field. The verbs are thin because the mapping is thin -- a
    block is a key, publishing is a ``put_batch`` -- and they are an object rather
    than three functions only so that the deployment is named once instead of at
    every call site.

    Deliberately absent: what a KV block *is*, and how big one is. See the module
    docstring for what used to be here and why it moved to the accelerator; the
    short version is that this object moves KV and computes none, so the byte count
    every fetch is priced against belongs with the thing that produces it. The one
    thing this class still insists on is that a caller publishing ``n`` keys hands
    it ``n`` blocks -- an arity check, not a size premise (see :meth:`publish`).

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
        of ``keys[i]``: the caller is the host that just held both -- it pulled part
        of that prefix and computed the rest -- and pairing them anywhere else would
        mean re-deriving here which key each block belongs to.

        A **cache fill**, so it is allowed to fail: ``False`` when the instance has
        no room for these blocks even after evicting what it could. A KV cache that
        cannot fit something does not cache it -- the request has already been
        served, and the only loss is that nobody reuses this prefix. Letting the
        store refuse and treating that as fatal would make a bounded cache unusable
        the moment one request's working set exceeded a volume.

        Raises:
            ValueError: if there is not exactly one block per key. Loud rather than
                zipped-to-the-shorter, because both ways of being wrong are silent
                and expensive: publishing fewer blocks than keys registers a prefix
                the directory will route reads to and the volume does not hold, and
                publishing more drops KV this host computed and paid for.
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

    async def fetch(self, inst: str, keys: List[str]) -> List[torch.Tensor]:
        """Pull ``keys`` into ``inst`` via a real ``get_batch`` (charges fabric).

        Drives the real client planning core + transport seam, so the storage /
        RAM / network cost is charged by the real cost model against the peer that
        actually serves the blocks.

        Answers with the KV, one block per key in the order asked for -- what the
        client returned, not a count derived from a size this object was told once.
        A caller that wants the bytes it just paid for can add them up off the
        tensors, which is the same number the transport charged and cannot drift
        from it. ``get_batch`` answers with a dict, so the ordering is re-imposed
        here: the caller asked for a prefix and a prefix has an order.

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
            return []
        got = await self.deployment.client_for(inst).get_batch(list(keys))
        return [got[k] for k in keys]
