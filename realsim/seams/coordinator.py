"""Fake coordinator actor handle dispatching to a real control plane.

The sibling of :mod:`realsim.seams.controller_handle`, for the other service a
serving host talks to. A capability whose control plane is a *coordinator* --
kvcache's scheduler holds every instance's queue, cache and decode occupancy, and
serializes routing decisions cluster-wide, so no serving host can hold it -- does
not reach it by holding the object. It reaches it the way it reaches the store:
through a handle, over calls that carry values.

:class:`CoordinatorHandle` is that handle: it wraps the control-plane object,
mirrors whatever surface that object declares, and is the single place a round trip
is charged. In a deployment it becomes a Monarch actor endpoint and nothing on
either side changes shape.

It names no method of its own
----------------------------
The surface is read off the object, exactly as a Monarch handle mirrors an actor's
``@endpoint`` methods. That matters twice over: this module sits below every
capability, so hard-coding one capability's method names would be the harness
knowing what a capability decided; and a *custom* control plane with a different
surface would otherwise be unreachable without editing this file.

Calls and sends
---------------
Which is which is also read off the object rather than listed here:

* an ``async def`` member is a **call** -- awaited, and it pays ``rtt`` twice,
  once out and once back, because the caller is blocked for both legs;
* a plain member is a one-way **send** -- forwarded, and free, because the sender
  does not wait for it. A real bus would still deliver it ``rtt`` later, so control
  would act on a slightly stale picture; that lag is *not* modelled, and it is the
  one piece of coordinator distance this seam leaves out.

So a capability states the difference where it declares its coordinator, by making
a member awaitable or not, and both sides agree by construction.

Cost
----
``rtt`` defaults to ``0.0``, which makes every call inline: awaiting a coroutine
that never suspends does not yield to the loop, so a default run is byte-identical
to holding the object directly -- the seam is structure, not a behaviour change.
Set ``TOSO_COORDINATOR_RTT`` (or ``--coordinator-rtt``) to give the hop a duration,
and it lands where it belongs: in front of every routing decision, and therefore in
TTFT. A control plane reads its own clock, so a decision made over a non-zero hop is
made at the time the request *arrived*, not the time the sender stamped it.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from sim_common import config

from realsim.seams.link import ServiceHop

__all__ = ["CoordinatorHandle"]


class CoordinatorHandle:
    """A control plane reached as a service, not held as an object.

    Args:
        control: the control-plane object this endpoint fronts (kvcache's
            scheduler). Whatever it declares is what this handle offers.
        rtt: one-way latency of the hop. ``None`` reads the ambient
            :attr:`sim_common.config.SimConfig.coordinator_rtt`.
    """

    def __init__(self, control: Any, *, rtt: Optional[float] = None) -> None:
        self.control = control
        self.hop = ServiceHop(
            rtt if rtt is not None else config.current().coordinator_rtt
        )

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Mirror ``name`` off the wrapped control plane, across the boundary."""
        if name.startswith("_"):
            # Never forward dunder or private lookups: this is an endpoint, not a
            # transparent proxy, and forwarding them breaks copy/pickle protocols.
            raise AttributeError(name)
        member = getattr(self.control, name)
        if inspect.iscoroutinefunction(member):

            async def call(*args: Any, **kwargs: Any) -> Any:
                return await self.hop.call(lambda: member(*args, **kwargs))

            return call

        def send(*args: Any, **kwargs: Any) -> Any:
            return member(*args, **kwargs)

        return send
