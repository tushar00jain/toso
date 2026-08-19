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

Inside a mirrored body, every line this repo added rather than copied is marked
``OURS``. There are four, all one thing: the directory also holds entries a volume
has *promised* (:mod:`realsim.seams.projection`), and each mirrored body has to keep
them away from a reader that asked for holders.

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
from realsim.seams.projection import Projecting

__all__ = ["ControllerService"]


class ControllerService(Projecting):
    """A real ``Controller`` and its endpoints' bodies.

    Args:
        controller: a real ``Controller`` instance (constructed off-actor and
            marked initialized by
            :class:`realsim.adapters.real_controller.RealControllerAdapter`).
    """

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        #: Directory mutations so far. Every registration and deregistration in a run
        #: passes through the three ``notify_*`` members below, so a reader that saw
        #: this number unchanged saw the same directory, at ``O(1)`` against a
        #: signature over the keys it read
        #: (:meth:`proposed.sensors.DirectorySensor._directory_stamp`). Advanced by
        #: whole calls rather than by entry, so it counts batches, not keys.
        self.revision = 0

    @property
    def entries(self):
        """The real ``Controller``'s own directory, which promises are written into."""
        return self.controller.keys_to_storage_volumes

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
            # OURS: a promise that landed stops being one, and nothing suspends
            # between the two calls, so no read sees the slot empty.
            self.unpromise(request.key, storage_volume_id)
            c._notify_put(request, storage_volume_id)
        self.revision += 1

    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
        # Mirrors Controller.notify_delete @endpoint body verbatim.
        c = self.controller
        c.assert_initialized()
        # OURS: a promise is not a stored copy, so deleting one is deleting nothing;
        # dropping it first is what makes the real helper raise as it would upstream.
        self.unpromise(key, storage_volume_id)
        c._notify_delete(key, storage_volume_id)
        self.revision += 1

    async def notify_delete_batch(
        self, volume_to_keys: dict[str, list[str]]
    ) -> None:
        # Mirrors Controller.notify_delete_batch @endpoint body verbatim.
        c = self.controller
        c.assert_initialized()
        for storage_volume_id, keys in volume_to_keys.items():
            for key in keys:
                self.unpromise(key, storage_volume_id)  # OURS, as in notify_delete
                c._notify_delete(key, storage_volume_id, missing_ok=True)
        self.revision += 1

    async def keys(self, prefix: Optional[str] = None) -> list[str]:
        # Mirrors Controller.keys @endpoint body verbatim:
        c = self.controller
        if prefix is None:
            listed = list(c.keys_to_storage_volumes.keys())
        else:
            listed = c.keys_to_storage_volumes.keys().filter_by_prefix(prefix)
        # OURS: a key only promised is a key nobody holds, so it is not registered.
        if not self.projected_owners():
            return listed
        return [
            key
            for key in listed
            if self.live_map(key, c.keys_to_storage_volumes[key])
        ]

    def locate_raw(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        *,
        projected: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """The real ``locate_volumes`` body, with no preference applied.

        :meth:`proposed.deployment.Controller.locate_raw` -- one of the members of
        that surface torchstore does not have, and what a directory sensor reads: not
        the preferred order, so a control plane ranks the directory rather than an
        answer somebody has already ranked, and not a coroutine, so forming one
        answer against it cannot be interleaved with forming another.

        Args:
            projected: include entries a volume has only promised
                (:class:`realsim.seams.projection.Promised`). Off by default, so one
                plane's promises are invisible to every other reader of this
                directory, ordinary reads included.
        """
        # Mirrors Controller.locate_volumes @endpoint body verbatim
        # (torchstore/controller.py, the body after the docstring), with the one
        # insertion marked OURS below.
        c = self.controller
        c.assert_initialized()
        result = {}
        for key in keys:
            volume_map = (
                c.keys_to_storage_volumes[key]
                if key in c.keys_to_storage_volumes
                else {}
            )
            # OURS: subtract the promises here, above the commit check, or a promised
            # shard makes a half-written DTensor look complete. A key left with no
            # live holder is a key nobody holds, which is the branch upstream spells
            # as `key not in keys_to_storage_volumes`.
            live_map = self.live_map(key, volume_map) if volume_map else volume_map
            answer = volume_map if projected else live_map
            if not answer:
                if missing_ok:
                    continue
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            if (
                require_fully_committed
                and live_map
                and not c._is_dtensor_fully_committed(key, live_map)
            ):
                raise KeyError(
                    f"DTensor '{key}' is only partially committed. "
                    f"Not all shards have been stored yet. "
                    f"Please ensure all ranks complete their put() operations."
                )
            result[key] = answer
        return result
