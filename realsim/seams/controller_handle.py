"""Fake controller actor handle dispatching to a real ``Controller``.

The client reaches the controller through a Monarch actor handle
(``locate_volumes.call_one(...)``, ``notify_put_batch.call(...)``,
``keys.call_one(...)``). :class:`FakeControllerHandle` mimics that surface and
dispatches to a **real** ``Controller`` instance constructed off-actor.

Why the read endpoints are re-stated here rather than called: ``Controller``'s
``@endpoint`` methods are ``EndpointProperty`` descriptors, not plain
coroutines, so they cannot be invoked directly off-actor. The mutation path is
already factored into a plain sync helper (``Controller._notify_put``), so
``notify_put_batch`` calls the real helper. The read endpoints
(``locate_volumes`` / ``keys``) are only ~5-line ``Trie`` reads that torchstore
has not extracted into sync helpers, so their bodies are **mirrored verbatim**
below (each mirrored block quotes the real endpoint it reproduces). All state
touched (``controller.keys_to_storage_volumes``, ``_is_dtensor_fully_committed``,
``_notify_put``, ``assert_initialized``) is the real object's.
"""

from __future__ import annotations

from typing import Any, Callable


class _ControllerEndpoint:
    """Mimics a Monarch endpoint's ``.call`` / ``.call_one`` awaitable surface."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)

    async def call_one(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)


class FakeControllerHandle:
    """In-process stand-in for a ``Controller`` actor handle.

    Args:
        controller: a real ``Controller`` instance (constructed off-actor and
            marked initialized by :class:`realsim.adapters.real_controller.RealControllerAdapter`).
    """

    def __init__(self, controller) -> None:
        self.controller = controller
        self.locate_volumes = _ControllerEndpoint(self._locate_volumes)
        self.notify_put_batch = _ControllerEndpoint(self._notify_put_batch)
        self.keys = _ControllerEndpoint(self._keys)
        self.notify_delete = _ControllerEndpoint(self._notify_delete)
        self.notify_delete_batch = _ControllerEndpoint(self._notify_delete_batch)

    async def _locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, Any]]:
        # Mirrors Controller.locate_volumes @endpoint body verbatim
        # (torchstore/controller.py, the body after the docstring):
        c = self.controller
        c.assert_initialized()
        result = {}
        for key in keys:
            if key not in c.keys_to_storage_volumes:
                if missing_ok:
                    continue
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            volume_map = c.keys_to_storage_volumes[key]
            if require_fully_committed and not c._is_dtensor_fully_committed(
                key, volume_map
            ):
                raise KeyError(
                    f"DTensor '{key}' is only partially committed. "
                    f"Not all shards have been stored yet. "
                    f"Please ensure all ranks complete their put() operations."
                )
            result[key] = volume_map
        return result

    async def _notify_put_batch(
        self,
        requests: list[Any],
        storage_volume_id: str,
    ) -> None:
        # Mirrors Controller.notify_put_batch @endpoint body verbatim; the loop
        # calls the real sync helper Controller._notify_put.
        c = self.controller
        c.assert_initialized()
        for request in requests:
            c._notify_put(request, storage_volume_id)

    async def _keys(self, prefix: str | None = None) -> list[str]:
        # Mirrors Controller.keys @endpoint body verbatim:
        c = self.controller
        if prefix is None:
            return list(c.keys_to_storage_volumes.keys())
        return c.keys_to_storage_volumes.keys().filter_by_prefix(prefix)

    async def _notify_delete(self, key: str, storage_volume_id: str) -> None:
        # Mirrors Controller.notify_delete @endpoint body verbatim.
        c = self.controller
        c.assert_initialized()
        c._notify_delete(key, storage_volume_id)

    async def _notify_delete_batch(
        self, volume_to_keys: dict[str, list[str]]
    ) -> None:
        # Mirrors Controller.notify_delete_batch @endpoint body verbatim.
        c = self.controller
        c.assert_initialized()
        for storage_volume_id, keys in volume_to_keys.items():
            for key in keys:
                c._notify_delete(key, storage_volume_id, missing_ok=True)
