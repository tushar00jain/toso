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

from torchstore.storage_volume import InMemoryStore


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

    Args:
        store: an existing ``InMemoryStore`` to back this volume; a fresh one is
            created if omitted.
    """

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store: InMemoryStore = store if store is not None else InMemoryStore()
        self.put = _VolumeEndpoint(self._put)
        self.get = _VolumeEndpoint(self._get)
        self.handshake = _VolumeEndpoint(self._handshake)
        self.delete = _VolumeEndpoint(self._delete)
        self.delete_batch = _VolumeEndpoint(self._delete_batch)
        self.reset = _VolumeEndpoint(self._reset)

    # Bodies below mirror the real StorageVolume @endpoint bodies verbatim.
    async def _put(self, transport_buffer, requests) -> None:
        await self.store.put(transport_buffer, requests)

    async def _get(self, transport_buffer, requests):
        return await self.store.get(transport_buffer, requests)

    async def _handshake(self, transport_buffer, requests):
        return await self.store.handshake(transport_buffer, requests)

    async def _delete(self, key: str) -> None:
        await self.store.delete(key)
        self.store.transport_context.delete(key)

    async def _delete_batch(self, keys: list[str]) -> None:
        await self.store.delete_batch(keys)
        self.store.transport_context.delete(keys)

    async def _reset(self) -> None:
        self.store.reset()
