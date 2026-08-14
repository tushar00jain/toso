"""The storage service, off-actor: :class:`VolumeService`.

The **server** side of a storage volume, and the third of the same pair this package
builds twice already (:mod:`realsim.seams.controller_service`,
:mod:`realsim.seams.control_plane_service`). In a deployment this is a Monarch
actor:
a process holding an ``InMemoryStore``, receiving messages, and answering them. Here
it is a plain object holding the same real store in this process, with the same
methods, receiving ordinary calls instead of messages.

It implements :class:`proposed.deployment.StorageVolume`, and every body **mirrors
the real ``StorageVolume`` endpoint verbatim** -- upstream each one simply delegates
to ``self.store`` (``torchstore/storage_volume.py``)::

    @endpoint
    async def put(self, transport_buffer, requests):
        await self.store.put(transport_buffer, requests)

So the *real* ``InMemoryStore`` put/get/handshake logic is what executes; only the
Monarch ``@endpoint`` shell is replaced, and what a caller holds instead is
:class:`realsim.seams.volume_handle.LocalVolumeHandle`.

What the seam adds around the real bodies
-----------------------------------------
Two things, both behind the same signatures, so neither is a surface a deployment
would have to gain:

* **Residency.** Every volume tracks its aggregate resident bytes -- added on put,
  subtracted on delete or reset -- because the real store is unbounded and a
  simulation of a bounded one has to know. The volume is handed ``meta_only``
  requests (``tensor_val`` stripped, see ``TransportBuffer._put_requests``), so a
  put's modeled bytes are read off the *landed* value with the same
  :func:`realsim.seams.transport._nbytes` helper the transport charges its fabric
  with; residency and fabric accounting therefore agree.
* **Capacity, and making room.** When the backing
  ``MachineProfile.storage_capacity_bytes`` is finite, a put that would exceed it
  evicts this volume's coldest keys (:mod:`realsim.seams._retention`) and refuses
  only if that still does not fit. Local, because recency of *this* volume's data is
  the one thing this object cannot be wrong about -- asking a cluster-wide service
  would be a round trip to be told what is already here. What it does need the
  directory for is afterwards: the bytes are gone, so it says so, over the handle it
  holds.

With the default ``math.inf`` capacity neither the check nor the ask ever fires and
the residency bookkeeping changes no behaviour, so the historical path stays
byte-identical.

The same resident-bytes mechanism could later back GPU/DRAM device-memory capacity
(track bytes resident on a device, reject an over-commit); only storage volumes are
modeled here.
"""

from __future__ import annotations

from typing import Any

import torch

from realsim.seams.transport import _nbytes
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile
from proposed import StorageFull
from realsim.seams._retention import LeastRecentlyUsed
from torchstore.storage_volume import InMemoryStore

__all__ = ["StorageCapacityExceeded", "VolumeService"]


class StorageCapacityExceeded(StorageFull):
    """Raised when a put would push a volume's resident bytes past its capacity.

    A :class:`proposed.deployment.StorageFull` with the numbers attached: the
    contract says "no room", this says how much of it there was.


    The volume's byte capacity comes from the run's
    :class:`~sim_common.cost_model.MachineProfile`
    (``storage_capacity_bytes``); the check is active only when that capacity is
    finite. The message reports the volume id, its capacity, the bytes currently
    resident, and the bytes the rejected put attempted to add, so an over-commit
    fails loudly instead of silently fitting infinite data.
    """

    def __init__(
        self,
        volume_id: str,
        capacity: float,
        resident: int,
        attempted: int,
    ) -> None:
        self.volume_id = volume_id
        self.capacity = capacity
        self.resident = resident
        self.attempted = attempted
        super().__init__(
            f"storage volume {volume_id!r} over capacity: "
            f"resident {resident}B + put {attempted}B = {resident + attempted}B "
            f"exceeds capacity {capacity}B"
        )


class VolumeService:
    """A real ``InMemoryStore``, its residency, and its endpoints' bodies.

    Args:
        store: an existing ``InMemoryStore`` to back this volume; a fresh one is
            created if omitted.
        volume_id: the volume's **directory** identity -- the id its co-located
            client registers puts under, which is therefore the id a dropped key
            has to be reported against (and the one named in a
            :class:`StorageCapacityExceeded` message). Not the endpoint id the
            transport prices locality with: telling the directory that one would
            name a volume it has never heard of, and the real
            ``Controller._notify_delete`` swallows that silently.
        profile: the run's target-machine
            :class:`~sim_common.cost_model.MachineProfile`; its
            ``storage_capacity_bytes`` is this volume's byte capacity (defaults
            to :data:`~sim_common.cost_model.DEFAULT_PROFILE`, i.e. unbounded).
        controller: a handle to the directory service, which is who this volume
            tells about what it dropped when a put does not fit. ``None`` -- the
            default -- means there is nobody to tell, which is the historical
            behaviour: refuse the put.
    """

    def __init__(
        self,
        store: InMemoryStore | None = None,
        *,
        volume_id: str = "",
        profile: MachineProfile | None = None,
        controller: Any | None = None,
        retention: Any | None = None,
    ) -> None:
        self.store: InMemoryStore = store if store is not None else InMemoryStore()
        self.volume_id = volume_id
        self._profile = profile if profile is not None else DEFAULT_PROFILE
        self._controller = controller
        # Per-key modeled bytes, so an overwrite replaces (not doubles) and a
        # delete subtracts exactly. ``resident_bytes`` is their sum.
        self._resident_by_key: dict[str, int] = {}
        self.resident_bytes: int = 0
        self.peak_resident_bytes: int = 0
        # How this volume picks its own victims. Swappable, because which key
        # should go is a selector and holding the bytes is not.
        self._retention = retention if retention is not None else LeastRecentlyUsed()

    @property
    def capacity_bytes(self) -> float:
        """This volume's byte capacity (``math.inf`` == unbounded)."""
        return self._profile.storage_capacity_bytes

    def _forget(self, key: str) -> None:
        """Drop a key's resident bytes (a real delete removed it from the store)."""
        self.resident_bytes -= self._resident_by_key.pop(key, 0)
        self._retention.forget(key)

    def _use(self, keys) -> None:
        """Mark ``keys`` as just accessed, with the bytes they now occupy."""
        for key in keys:
            self._retention.note(key, self._resident_by_key.get(key))

    def _stored_nbytes(self, value: Any) -> int:
        """Modeled bytes of a value as ``InMemoryStore`` stores it.

        Unwraps the three store representations (see ``InMemoryStore._store``) and
        defers each carrier to :func:`realsim.seams.transport._nbytes`, so the
        resident total is byte-consistent with the transport's fabric accounting:

        * a full tensor -- stored as the ``torch.Tensor`` itself;
        * an object (our metadata-only carrier) -- stored as ``{"obj": value}``;
        * DTensor shards -- ``{coordinates: {"slice", "tensor"}}``, summed.
        """
        if isinstance(value, torch.Tensor):
            return _nbytes(value)
        if isinstance(value, dict):
            if "obj" in value:
                return _nbytes(value["obj"])
            return sum(_nbytes(shard.get("tensor")) for shard in value.values())
        return 0

    # -- proposed.deployment.StorageVolume ---------------------------------- #
    # Bodies mirror the real StorageVolume @endpoint bodies verbatim, with the
    # resident-bytes accounting wrapped around the real store lifecycle.
    async def put(self, transport_buffer, requests) -> None:
        # The volume sees ``meta_only`` requests (no tensor_val), so we size the
        # put from what actually LANDS in the store, not the request. Let the real
        # put run, then recompute the resident bytes of each written key from the
        # stored value (an overwrite replaces the key's prior bytes) and enforce
        # the volume's capacity. With the default infinite capacity the check is
        # never tripped and this is pure, free bookkeeping (byte-identical run).
        await self.store.put(transport_buffer, requests)
        keys = {r.key for r in requests}
        new_by_key = {k: self._stored_nbytes(self.store.kv.get(k)) for k in keys}
        projected = self.resident_bytes + sum(
            nbytes - self._resident_by_key.get(key, 0)
            for key, nbytes in new_by_key.items()
        )
        if projected > self.capacity_bytes:
            # Out of room. Ask what to drop before refusing: this is the one moment
            # the store *knows* it is full, and the one thing it cannot answer is
            # which copies are still worth keeping. Dropping happens through the
            # ordinary delete path, so resident bytes stay exact.
            projected = await self._ask_for_room(projected, keys)
        if projected > self.capacity_bytes:
            # Nobody to ask, or not enough freed: roll the just-landed data back out
            # (so it does not silently fit) and fail loudly. Only newly-added keys
            # are removed; the resident bookkeeping is left uncommitted.
            attempted = projected - self.resident_bytes
            for key in keys:
                if key not in self._resident_by_key:
                    await self.store.delete(key)
            raise StorageCapacityExceeded(
                self.volume_id, self.capacity_bytes, self.resident_bytes, attempted
            )
        self._resident_by_key.update(new_by_key)
        self._use(keys)
        self.resident_bytes = projected
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    async def get(self, transport_buffer, requests):
        self._use(r.key for r in requests)
        return await self.store.get(transport_buffer, requests)

    async def touch(self, keys: list) -> None:
        """Report a read of ``keys`` that did not come through this volume.

        Nothing moves and nothing is charged: the caller already had the bytes. What
        it buys is recency -- a cache whose hits bypass the store would otherwise
        look untouched to the one object deciding what to drop, and it would drop
        exactly the blocks being reused.
        """
        self._use(k for k in keys if k in self._resident_by_key)

    async def handshake(self, transport_buffer, requests):
        return await self.store.handshake(transport_buffer, requests)

    async def delete(self, key: str) -> None:
        await self.store.delete(key)
        self.store.transport_context.delete(key)
        self._forget(key)

    async def delete_batch(self, keys: list[str]) -> None:
        await self.store.delete_batch(keys)
        self.store.transport_context.delete(keys)
        for key in keys:
            self._forget(key)

    async def reset(self) -> None:
        self.store.reset()
        self._resident_by_key.clear()
        self.resident_bytes = 0
        # The ranking is keyed by the keys that just went, so it has to hear about
        # them too. Left alone it goes on ranking a working set that no longer
        # exists, and those ghosts are *colder* than anything real -- so the next
        # full volume picks them as its victims, frees nothing, and refuses a put
        # it had room for.
        for key in list(self._retention.held()):
            self._retention.forget(key)
        # peak_resident_bytes is a run-lifetime high-water mark; reset does not
        # lower it.

    # -- making room, and telling the directory what went ------------------- #
    async def _ask_for_room(self, projected: int, incoming: set) -> int:
        """Make room for the overshoot, drop what the ranking names, return the new total.

        Nobody is *asked*: which of this volume's keys is coldest is the one thing
        this volume cannot be wrong about, so the victims come from its own
        ``_retention`` ranking. The directory is only *told*, afterwards -- which is
        why its absence means this cannot proceed at all rather than evicting
        silently: bytes dropped without a deregistration would leave the directory
        routing later reads here for data that is gone.

        Never asks for more than the overshoot, and never drops a key this very put
        is writing -- freeing bytes the caller is about to re-add would be a wasted
        round trip at best and, on an overwrite, would drop the new value. That
        exclusion goes *into* the ranking rather than filtering its answer, so the
        bytes of a skipped key are never counted toward the need.
        """
        if self._controller is None:
            return projected
        need = int(projected - self.capacity_bytes)
        victims = [
            key for key in self._retention.victims(need, exclude=incoming)
            if key in self._resident_by_key
        ]
        for key in victims:
            freed = self._resident_by_key.get(key, 0)
            # The volume's own delete endpoint, not an open-coded copy of part of
            # it: an evicted key has to release everything a deleted key releases,
            # and the per-key transport state is easy to forget. Dropping the
            # value while keeping that entry leaks the resource it names -- a
            # shared-memory segment, a process group -- for a key nobody can ask
            # for any more.
            await self.delete(key)
            projected -= freed
        if victims:
            # The bytes are gone, so the directory must stop saying this volume has
            # them -- otherwise a later read is routed here for something that was
            # dropped. Told over the same handle the room was asked for.
            await self._controller.notify_delete_batch.call_one(
                {self.volume_id: victims}
            )
        return projected
