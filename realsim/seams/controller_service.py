"""The real torchstore controller behind local endpoint-shaped handles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["ControllerService"]


class ControllerService:
    """Delegate directory operations to one real controller."""

    def __init__(self, controller) -> None:
        self.controller = controller

    def _locate(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        *,
        prefer: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self.controller._locate(
            keys,
            missing_ok,
            require_fully_committed,
            prefer=prefer,
        )

    def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        prefer: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self._locate(
            keys,
            missing_ok,
            require_fully_committed,
            prefer=prefer,
        )

    def notify_put_batch(
        self,
        requests: Sequence[Any],
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        return self.controller._notify_put_batch(
            requests, storage_volume_id, pending=pending
        )

    def _notify_put(
        self, request: Any, storage_volume_id: str, *, pending: bool = True
    ) -> int:
        return self.controller._notify_put(
            request, storage_volume_id, pending=pending
        )

    def notify_delete(self, key: str, storage_volume_id: str) -> None:
        self.controller.assert_initialized()
        self.controller._notify_delete(key, storage_volume_id)

    def notify_delete_batch(
        self,
        volume_to_keys: dict[str, list[str]] | None = None,
        *,
        pub: int | None = None,
    ) -> None:
        self.controller.assert_initialized()
        self.controller._notify_delete_batch(volume_to_keys, pub=pub)

    def keys(self, prefix: str | None = None) -> list[str]:
        return self.controller._keys(prefix)

    def serving_union(
        self, requests: Sequence[Any]
    ) -> tuple[dict[str, set[str]], dict[str, set[int]]]:
        return self.controller.serving_union(requests)
