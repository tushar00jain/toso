"""Fake storage-volume actor handle backed by a real ``InMemoryStore``.

The client / transport reach the storage volume through a Monarch actor handle
whose endpoints are invoked as ``.put.call(...)`` / ``.get.call_one(...)`` /
``.handshake.call_one(...)``. :class:`FakeVolumeHandle` mimics exactly that
surface, and each method body **mirrors the real ``StorageVolume`` endpoint
verbatim** -- every real endpoint simply delegates to ``self.store`` (see
``torchstore/storage_volume.py``):

    @endpoint
    async def put(self, transport_buffer, requests):
        await self.store.put(transport_buffer, requests)

    @endpoint
    async def get(self, transport_buffer, requests):
        return await self.store.get(transport_buffer, requests)

    @endpoint
    async def handshake(self, transport_buffer, requests):
        return await self.store.handshake(transport_buffer, requests)

So the *real* ``InMemoryStore`` put/get/handshake logic is what executes; only
the Monarch ``@endpoint`` + ``.call``/``.call_one`` dispatch shell is replaced.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from realsim.seams.transport import _nbytes
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile
from torchstore.storage_volume import InMemoryStore


class StorageCapacityExceeded(Exception):
    """Raised when a put would push a volume's resident bytes past its capacity.

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


class _VolumeEndpoint:
    """Mimics a Monarch endpoint's ``.call`` / ``.call_one`` awaitable surface.

    In this single-process sim both ``call`` (broadcast) and ``call_one`` (single
    actor) resolve to the same in-process coroutine; callers that ignore the
    ValueMesh return of a real ``.call`` are unaffected.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)

    async def call_one(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)


class FakeVolumeHandle:
    """In-process stand-in for a ``StorageVolume`` actor handle.

    On top of mirroring the real endpoint bodies, the seam tracks this volume's
    **resident bytes** -- the aggregate working set of everything currently
    stored (added on put, subtracted on delete/reset) -- around the real
    ``InMemoryStore`` put/get/delete lifecycle. The real store is left untouched
    (it is unbounded); the accounting lives here in the seam. The volume is
    handed ``meta_only`` requests (``tensor_val`` stripped -- see
    ``TransportBuffer._put_requests``), so a put's modeled bytes are read off the
    *landed* value in the store with the same
    :func:`realsim.seams.transport._nbytes` helper the transport uses for its
    fabric charges; resident accounting and fabric accounting therefore agree.

    When the backing ``MachineProfile.storage_capacity_bytes`` is finite, a put
    that would push resident bytes past that capacity raises
    :class:`StorageCapacityExceeded` before the data lands. With the default
    ``math.inf`` capacity the check never fires and the resident bookkeeping adds
    no behavioral change, so the historical path stays byte-identical.

    The same resident-bytes mechanism could later back GPU/DRAM device-memory
    capacity (track bytes resident on a device, reject an over-commit); only
    storage volumes are modeled here.

    Args:
        store: an existing ``InMemoryStore`` to back this volume; a fresh one is
            created if omitted.
        volume_id: the volume's id, used in a :class:`StorageCapacityExceeded`
            message. Purely descriptive.
        profile: the run's target-machine
            :class:`~sim_common.cost_model.MachineProfile`; its
            ``storage_capacity_bytes`` is this volume's byte capacity (defaults
            to :data:`~sim_common.cost_model.DEFAULT_PROFILE`, i.e. unbounded).
    """

    def __init__(
        self,
        store: InMemoryStore | None = None,
        *,
        volume_id: str = "",
        profile: MachineProfile | None = None,
    ) -> None:
        self.store: InMemoryStore = store if store is not None else InMemoryStore()
        self.volume_id = volume_id
        self._profile = profile if profile is not None else DEFAULT_PROFILE
        # Per-key modeled bytes, so an overwrite replaces (not doubles) and a
        # delete subtracts exactly. ``resident_bytes`` is their sum.
        self._resident_by_key: dict[str, int] = {}
        self.resident_bytes: int = 0
        self.peak_resident_bytes: int = 0
        self.put = _VolumeEndpoint(self._put)
        self.get = _VolumeEndpoint(self._get)
        self.handshake = _VolumeEndpoint(self._handshake)
        self.delete = _VolumeEndpoint(self._delete)
        self.delete_batch = _VolumeEndpoint(self._delete_batch)
        self.reset = _VolumeEndpoint(self._reset)

    @property
    def capacity_bytes(self) -> float:
        """This volume's byte capacity (``math.inf`` == unbounded)."""
        return self._profile.storage_capacity_bytes

    def _forget(self, key: str) -> None:
        """Drop a key's resident bytes (a real delete removed it from the store)."""
        self.resident_bytes -= self._resident_by_key.pop(key, 0)

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

    # Bodies below mirror the real StorageVolume @endpoint bodies verbatim, with
    # the resident-bytes accounting wrapped around the real store lifecycle.
    async def _put(self, transport_buffer, requests) -> None:
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
            # Over capacity: roll the just-landed data back out (so it does not
            # silently fit) and fail loudly. Only newly-added keys are removed;
            # the resident bookkeeping is left uncommitted.
            attempted = projected - self.resident_bytes
            for key in keys:
                if key not in self._resident_by_key:
                    await self.store.delete(key)
            raise StorageCapacityExceeded(
                self.volume_id, self.capacity_bytes, self.resident_bytes, attempted
            )
        self._resident_by_key.update(new_by_key)
        self.resident_bytes = projected
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    async def _get(self, transport_buffer, requests):
        return await self.store.get(transport_buffer, requests)

    async def _handshake(self, transport_buffer, requests):
        return await self.store.handshake(transport_buffer, requests)

    async def _delete(self, key: str) -> None:
        await self.store.delete(key)
        self.store.transport_context.delete(key)
        self._forget(key)

    async def _delete_batch(self, keys: list[str]) -> None:
        await self.store.delete_batch(keys)
        self.store.transport_context.delete(keys)
        for key in keys:
            self._forget(key)

    async def _reset(self) -> None:
        self.store.reset()
        self._resident_by_key.clear()
        self.resident_bytes = 0
        # peak_resident_bytes is a run-lifetime high-water mark; reset does not
        # lower it.
