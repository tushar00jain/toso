"""The directory service, off-actor: :class:`ControllerService`.

This is the **server** side of the directory. In a deployment it is a Monarch
actor: a process holding the real ``Controller``, receiving messages, and
consulting whatever selector was installed in it. Here it is a plain object holding
the same real ``Controller`` in this process, with the same methods, receiving
ordinary calls instead of messages.

It implements :class:`proposed.deployment.Controller` -- the surface declared
there, method for method -- which is the point of the split: the thing that *is* a
controller is a controller, and the thing a caller *holds* is
:class:`realsim.seams.controller_handle.LocalControllerHandle`, which is a
different shape (endpoints) for a different reason (it stands in for the process
boundary). Fusing the two into one object was what made "is this client side or
server side?" unanswerable.

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

Where a control plane runs
--------------------------
Here. A :class:`~proposed.selector.KeySelector` installed in this service is a
capability's control plane running *inside the directory service* -- all of
dedupe's, and the source half of kvcache's. :meth:`ControllerService._route` is
that call site, kept as its own method so it is findable: the mirrored real body
runs first (:meth:`locate_raw`), then the selector is asked which of the directory's
volumes should serve this requester, and the answer is *withheld* until the chosen
source is usable. With no selector installed -- the default -- ``locate_volumes`` is
exactly the mirrored real body.

The selector is asked for a decision and handed no sensor: it runs on this side, so
it senses through the :class:`~proposed.view.View` the run attached it to
(:meth:`proposed.selector.Selector.attach`), which reads :meth:`locate_raw` and
therefore cannot re-enter the hook it is being called from.

That is the only thing here that knows what a selector is. Hearing that the
directory changed goes the other way: anything at all may :meth:`subscribe`, and
this service calls plain callables back without knowing whose they are.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from proposed.deployment import Registered

__all__ = ["ControllerService"]


class ControllerService:
    """A real ``Controller``, its selector, and its endpoints' bodies.

    Args:
        controller: a real ``Controller`` instance (constructed off-actor and
            marked initialized by
            :class:`realsim.adapters.real_controller.RealControllerAdapter`).

    Built with no selector and none installed, the directory answers for itself --
    which is what :class:`~proposed.selector.NaiveKeySelector` says anyway.
    """

    def __init__(self, controller) -> None:
        self.controller = controller
        self._selector: Optional[Any] = None
        self._subscribers: List[Registered] = []

    def install_selector(self, selector: Any) -> None:
        """Install a control plane in this service after construction.

        Two-phase because a selector senses through a
        :class:`~proposed.view.View` built over this service, so it cannot exist
        before the service does.
        """
        self._selector = selector

    # -- proposed.deployment.Controller ------------------------------------- #
    def subscribe(self, on_register: Registered) -> None:
        """Call ``on_register(volume_id, keys)`` after every registration."""
        self._subscribers.append(on_register)

    async def locate_volumes(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """The real ``locate_volumes`` body, then the routing hook."""
        selection = await self._route(keys)
        if selection is None:
            return self.locate_raw(keys, missing_ok, require_fully_committed)
        # Withhold the answer until the chosen source is usable. The directory is
        # re-read afterwards because waiting is exactly what lets it change: the
        # source the selector picked registers while we are blocked here.
        await selection.wait()
        located = self.locate_raw(keys, missing_ok, require_fully_committed)
        return selection.narrow(located)

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
        # The directory just changed; whoever asked to hear about it is told, in
        # subscription order. Synchronous and inside this body on purpose: a
        # subscriber that could suspend would let a second registration interleave
        # with this one (:meth:`proposed.deployment.Controller.subscribe`).
        keys = [r.key for r in requests]
        for on_register in self._subscribers:
            on_register(storage_volume_id, keys)

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

    # -- the control-plane call site, and the surface's other read ---------- #
    # ``_route`` is this service's own; ``locate_raw`` is the surface's other
    # read, kept next to the hook it exists to avoid.
    async def _route(self, keys: Sequence[str]) -> Optional[Any]:
        """Ask the installed control plane who should serve ``keys``.

        ``None`` means there is nobody to ask -- no selector installed, or no
        requester bound (a caller the seam cannot identify, GAP 2) -- and the
        directory answers for itself.
        """
        if self._selector is None:
            return None
        # Import locally: the seam is otherwise dependency-free, and only a
        # routed run needs to know who is calling.
        from realsim.seams import factory

        requester = factory.current_requester()
        if requester is None:
            return None
        return await self._selector.select(list(keys), requester)

    def locate_raw(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """The real ``locate_volumes`` body, unrouted (the sensor read).

        :meth:`proposed.deployment.Controller.locate_raw` -- the one member of that
        surface torchstore does not have yet, and what a ``View`` reads: not the
        routed read, so a selector sensing the directory cannot re-enter the hook it is
        being called from, and not a coroutine, so forming an answer against it
        cannot be interleaved with forming another.
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
