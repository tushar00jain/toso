"""The directory service, off-actor: :class:`ControllerService`.

This is the **server** side of the directory. In a deployment it is a Monarch
actor: a process holding the real ``Controller`` and receiving messages. Here it is
a plain object holding the same real ``Controller`` in this process, with the same
methods, receiving ordinary calls instead of messages.

It implements :class:`proposed.deployment.Controller` -- the surface declared
there, method for method -- which is the point of the split: the thing that *is* a
controller is a controller, and the thing a caller *holds* is
:class:`realsim.seams.controller_handle.LocalControllerHandle`, which is a
different shape (endpoints) for a different reason (it stands in for the process
boundary).

Why the bodies are re-stated rather than called
-----------------------------------------------
``Controller``'s ``@endpoint`` methods are ``EndpointProperty`` descriptors, not
plain coroutines, so they cannot be invoked off-actor. The mutation path is
already factored into real sync helpers (``Controller._notify_put`` /
``_notify_delete``), so those are called. The read endpoints (``locate_volumes`` /
``keys``) are ~5-line ``Trie`` reads torchstore has not extracted, so their bodies
are **mirrored verbatim** below, each quoting the endpoint it reproduces. Every
piece of state touched is the real object's, and
``realsim/tests/test_upstream_parity.py`` fails the build if an original changes.

No control plane runs here
--------------------------
The one thing this service does that torchstore's does not is **apply a preference
its caller handed it**: ``locate_volumes`` reads the directory and then puts the
answer in the order the caller asked for (:func:`proposed.selector.prefer`), so the
client's "first volume listed per key" is the source the caller chose. It consults
nobody to do that and holds nothing that decides. Whoever wanted a say asked a
control plane *before* calling, and what arrives here is the ranking that came back.

The preference reaches the body through a coroutine binding
(:func:`realsim.seams.factory.current_prefer`) rather than an argument, because the
real ``LocalClient.get`` has no parameter to carry it: upstream it is one optional
argument on the read path, applied to the located map before the client picks a
volume per key. That is the only reason this is ambient, and the only thing about it
that is simulation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from proposed.selector import prefer

from realsim.seams import factory

__all__ = ["ControllerService"]


class ControllerService:
    """A real ``Controller`` and its endpoints' bodies.

    Args:
        controller: a real ``Controller`` instance (constructed off-actor and
            marked initialized by
            :class:`realsim.adapters.real_controller.RealControllerAdapter`).
    """

    def __init__(self, controller) -> None:
        self.controller = controller

    # -- proposed.deployment.Controller ------------------------------------- #
    async def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """The real ``locate_volumes`` body, then the caller's source preference."""
        located = self.locate_raw(keys, missing_ok, require_fully_committed)
        return prefer(located, factory.current_prefer())

    async def notify_put_batch(
        self,
        requests: Sequence[Any],
        storage_volume_id: str,
    ) -> None:
        # Mirrors Controller.notify_put_batch @endpoint body verbatim; the loop
        # calls the real sync helper Controller._notify_put.
        c = self.controller
        c.assert_initialized()
        for request in requests:
            c._notify_put(request, storage_volume_id)

    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
        # Mirrors Controller.notify_delete @endpoint body verbatim.
        c = self.controller
        c.assert_initialized()
        c._notify_delete(key, storage_volume_id)

    async def notify_delete_batch(
        self, volume_to_keys: dict[str, list[str]]
    ) -> None:
        # Mirrors Controller.notify_delete_batch @endpoint body verbatim.
        c = self.controller
        c.assert_initialized()
        for storage_volume_id, keys in volume_to_keys.items():
            for key in keys:
                c._notify_delete(key, storage_volume_id, missing_ok=True)

    async def keys(self, prefix: Optional[str] = None) -> list[str]:
        # Mirrors Controller.keys @endpoint body verbatim:
        c = self.controller
        if prefix is None:
            return list(c.keys_to_storage_volumes.keys())
        return c.keys_to_storage_volumes.keys().filter_by_prefix(prefix)

    def locate_raw(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """The real ``locate_volumes`` body, with no preference applied.

        :meth:`proposed.deployment.Controller.locate_raw` -- the one member of that
        surface torchstore does not have, and what a ``View`` reads: not the
        preferred order, so a control plane ranks the directory rather than an
        answer somebody has already ranked, and not a coroutine, so forming one
        answer against it cannot be interleaved with forming another.
        """
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
