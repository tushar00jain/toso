"""Fake coordinator actor handle dispatching to a real control plane.

The sibling of :mod:`realsim.seams.controller_handle`, for the other service a
serving host talks to. A capability whose control plane is a *coordinator* --
kvcache's scheduler holds every instance's queue, cache and decode occupancy, and
serializes routing decisions cluster-wide, so no serving host can hold it -- does
not reach it by holding the object. It reaches it the way it reaches the store:
through a handle, over calls that carry values.

:class:`CoordinatorHandle` is that handle: it refers to a
:class:`realsim.seams.coordinator_service.CoordinatorService` -- the server side,
which holds the deciding object -- and is the single place a round trip is charged. In a deployment it becomes a Monarch actor endpoint and nothing on
either side changes shape.

One endpoint per member of the surface -- and there are two
-----------------------------------------------------------
The members are :class:`proposed.deployment.Coordinator`'s: ``decide`` and
``observe``. Naming them here is not the harness knowing what a capability decided --
it is the harness knowing the port, which lives in ``proposed`` exactly so both sides
can be written without either knowing the other. Same as
:class:`realsim.seams.controller_handle.LocalControllerHandle` and its five.

It used to name six, after one application's questions. That put a KV-cache
vocabulary in a generic harness file and meant a second application would have to
edit this one. The questions are payload now, so this file is done: any application,
two endpoints.

Calls and sends
---------------
Every member comes back as a :class:`~realsim.seams.link.LocalEndpoint`, so the
*caller* chooses how to send, exactly as it would with Monarch:

* ``decide.call_one(demand)`` -- a call, awaited, paying the hop twice because
  the caller is blocked out and back;
* ``observe.broadcast(fact)`` -- one-way, free, because the
  sender does not wait. A real bus would still deliver it ``rtt`` later, so control
  would act on a slightly stale picture; that lag is *not* modelled, and it is the
  one piece of coordinator distance this seam leaves out.

That is why the surface is endpoints and not methods. A handle offering methods
would be a shape Monarch does not have, so swapping in a real actor handle would
mean editing every caller -- and "the `[S]` piece disappears and nothing changes
shape" is the claim this package is making.

Cost
----
``rtt`` defaults to ``0.0``, which makes every call inline: awaiting a coroutine
that never suspends does not yield to the loop, so a default run is byte-identical
to calling the object directly -- the seam is structure, not a behaviour change.
Set ``TOSO_COORDINATOR_RTT`` (or ``--coordinator-rtt``) to give the hop a duration,
and it lands where it belongs: in front of every routing decision, and therefore in
TTFT. A control plane reads its own clock, so a decision made over a non-zero hop is
made at the time the request *arrived*, not the time the sender stamped it.
"""

from __future__ import annotations

from typing import Any, Optional

from sim_common import config

from realsim.seams.link import LocalEndpoint, ServiceHop

__all__ = ["CoordinatorHandle"]


class CoordinatorHandle:
    """A reference to a :class:`CoordinatorService` living in this process.

    Args:
        service: the coordinator service this refers to.
        rtt: one-way latency of the hop. ``None`` reads the ambient
            :attr:`sim_common.config.SimConfig.coordinator_rtt`.
    """

    def __init__(self, service: Any, *, rtt: Optional[float] = None) -> None:
        self.service = service
        self.hop = ServiceHop(
            rtt if rtt is not None else config.current().coordinator_rtt
        )
        # One hop shared by both endpoints: they are the same boundary.
        self.decide = LocalEndpoint(service.decide, self.hop)
        self.observe = LocalEndpoint(service.observe, self.hop)

    @property
    def control(self) -> Any:
        """The deciding object behind the service, for tests asserting on it."""
        return self.service.control
