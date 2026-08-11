"""What a caller holds to reach the directory: :class:`LocalControllerHandle`.

The **client** side, and only that. A caller does not hold a controller; it holds a
reference to one, and Monarch's reference is not the same shape as the actor:
``@endpoint`` turns each method into an ``EndpointProperty``, so the attribute
resolves to an endpoint object and the call reads

    await handle.locate_volumes.call_one(keys, missing_ok=True)

which is what real ``LocalClient`` code writes (``torchstore/client.py``). This
class is that reference, for a service living in this process instead of another
one: an endpoint per method of
:class:`realsim.seams.controller_service.ControllerService`, and the one place the
distance to it is charged.

In a deployment there is nothing here to keep. ``get_or_spawn_controller`` returns
Monarch's own handle, whose endpoints do the same thing over the wire; this class
is the `[S]` that disappears. Which is also why it does not implement
:class:`proposed.deployment.Controller`: the *service* implements that, and a
reference to a service is a different shape. Fusing the two into one object is what
used to make "is this the client side or the server side?" unanswerable about this
file -- the mirrored endpoint bodies and the policy hook now live in the service,
where they always belonged.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from realsim.seams.link import ServiceHop

__all__ = ["LocalControllerHandle"]


class _ControllerEndpoint:
    """One method of the service, as a caller reaches it.

    Mimics a Monarch endpoint's ``.call`` / ``.call_one`` awaitable surface, and
    pays the hop -- reaching the directory is a round trip, so every endpoint goes
    through the shared :class:`~realsim.seams.link.ServiceHop`. At the default
    ``rtt`` of 0 that is inline and changes nothing, which is why it went unnoticed
    that this boundary -- crossed by every capability on every directory read, and
    by every consultation of a policy installed there -- was the one seam charging
    nothing.
    """

    def __init__(self, fn: Callable[..., Any], hop: ServiceHop) -> None:
        self._fn = fn
        self._hop = hop

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        return await self._hop.call(lambda: self._fn(*args, **kwargs))

    async def call_one(self, *args: Any, **kwargs: Any) -> Any:
        return await self._hop.call(lambda: self._fn(*args, **kwargs))


class LocalControllerHandle:
    """A reference to a :class:`ControllerService` living in this process.

    Args:
        service: the directory service this refers to.
        hop: what reaching it costs. ``None`` is a free hop, which is what a test
            wanting the directory and nothing else wants; a run builds one from
            :attr:`sim_common.config.SimConfig.controller_rtt`.
    """

    def __init__(self, service, *, hop: Optional[ServiceHop] = None) -> None:
        self.service = service
        # One hop shared by every endpoint: they are all the same boundary.
        self.hop = hop if hop is not None else ServiceHop()
        self.locate_volumes = _ControllerEndpoint(service.locate_volumes, self.hop)
        self.notify_put_batch = _ControllerEndpoint(service.notify_put_batch, self.hop)
        self.keys = _ControllerEndpoint(service.keys, self.hop)
        self.notify_delete = _ControllerEndpoint(service.notify_delete, self.hop)
        self.notify_delete_batch = _ControllerEndpoint(
            service.notify_delete_batch, self.hop
        )

    @property
    def controller(self):
        """The real ``Controller`` behind the service, for tests asserting on it."""
        return self.service.controller

    def install_policy(self, policy: Any, view: Any) -> None:
        """Install a control plane in the service this refers to.

        The policy runs in the *service* and is stored there; this exists for the
        one caller that holds a handle and not a service
        (:class:`realsim.mesh.Mesh`, assembling a run).
        """
        self.service.install_policy(policy, view)
