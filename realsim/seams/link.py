"""One service boundary, and what crossing it costs: :class:`ServiceHop`.

Every `[S]` seam in this package stands in for something that is an actor
endpoint in a deployment -- :class:`~realsim.seams.controller_handle.FakeControllerHandle`
for the directory, :class:`~realsim.seams.coordinator.CoordinatorHandle` for a
capability's coordinator. Standing in for an endpoint means two things: dispatch
to a real object in this process, and *be the place the distance is charged*.
The first was written twice; the second only once, so the coordinator hop could
be priced and the controller hop was silently free for every capability, the
baseline included. This is the second half, factored out.

A hop is deliberately not a transport. The transport seam charges *bytes* against
network, storage and RAM (:mod:`realsim.seams.transport`); this charges the
*round trip* of a request whose payload is a decision, which is latency and
nothing else. Two boundaries, two costs, and a request pays both.

Free by default
---------------
``rtt`` is ``0.0`` unless a run asks for one, and at zero :meth:`ServiceHop.call`
never suspends: awaiting a coroutine that does not itself await runs inline, so
the ready queue is untouched and the run is byte-identical to calling the object
directly. That is what makes the seam structure rather than a behaviour change --
a non-zero hop is an opt-in fidelity model, like ``contention``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

__all__ = ["ServiceHop"]


class ServiceHop:
    """The latency of reaching a service, and the way a call pays it.

    Args:
        rtt: one-way latency. A :meth:`call` pays it **twice** -- out and back --
            because the caller is blocked for both legs. A one-way notification
            pays :meth:`leg` once, or nothing at all if the sender does not wait
            for delivery, which is the sender's choice to state.
    """

    def __init__(self, rtt: float = 0.0) -> None:
        self.rtt = rtt

    async def leg(self) -> None:
        """Cross the boundary once. Free -- and inline -- when ``rtt`` is 0."""
        if self.rtt:
            await asyncio.sleep(self.rtt)

    async def call(self, thunk: Callable[[], Any]) -> Any:
        """Run ``thunk`` on the far side of the boundary and bring the answer back.

        A thunk rather than ``(fn, *args)`` so that anything the receiver should
        evaluate *on arrival* -- its own clock, most of all -- is read after the
        outbound leg rather than at the sender's instant.
        """
        await self.leg()
        out = thunk()
        if inspect.isawaitable(out):
            out = await out
        await self.leg()
        return out
