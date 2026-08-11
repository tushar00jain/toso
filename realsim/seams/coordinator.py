"""Fake coordinator actor handle dispatching to a real control plane.

The sibling of :mod:`realsim.seams.controller_handle`, for the other service a
serving host talks to. A capability whose control plane is a *coordinator* --
kvcache's scheduler holds every instance's queue, cache and decode occupancy, and
serializes routing decisions cluster-wide, so no serving host can hold it -- does
not reach it by holding the object. It reaches it the way it reaches the store:
through a handle, over calls that carry values.

:class:`CoordinatorHandle` is that handle. It wraps the control-plane object,
presents the capability's port (kvcache's
:class:`~kvcache_sim.control.scheduler.Coordinator`), and is the single place a
round trip is charged. In a deployment it becomes a Monarch actor endpoint and
nothing on either side changes shape; under simulation it is the `[S]` piece that
was missing from the stack, which is why the hop used to cost nothing.

Calls and sends
---------------
The four request/response members are awaitable and pay ``rtt`` **twice** -- once
out, once back -- because the caller is blocked for both legs. The two
observations are one-way sends: the sender does not wait, so they cost it
nothing. A real bus would still deliver them ``rtt`` later, so control would act
on a slightly stale decode picture; that lag is *not* modelled (see
:mod:`sim_common.config`), and it is the one piece of coordinator distance this
seam leaves out.

Cost
----
``rtt`` defaults to ``0.0``, which makes every call inline: awaiting a coroutine
that never suspends does not yield to the loop, so a default run is
byte-identical to holding the object directly -- the seam is structure, not a
behaviour change. Set ``TOSO_COORDINATOR_RTT`` (or ``--coordinator-rtt``) to give
the hop a duration, and it lands where it belongs: in front of every ``schedule``,
and therefore in TTFT.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from sim_common import config

from realsim.seams.link import ServiceHop

__all__ = ["CoordinatorHandle"]


class CoordinatorHandle:
    """A control plane reached as a service, not held as an object.

    Args:
        control: the control-plane object this endpoint fronts (kvcache's
            scheduler). Its methods are ordinary sync ones; making them look like
            a wire is this class's job, not theirs.
        rtt: one-way latency of the hop. ``None`` reads the ambient
            :attr:`sim_common.config.SimConfig.coordinator_rtt`.
    """

    def __init__(self, control: Any, *, rtt: Optional[float] = None) -> None:
        self.control = control
        self.hop = ServiceHop(
            rtt if rtt is not None else config.current().coordinator_rtt
        )

    # -- calls: the caller waits for a reply, so it pays both legs --------- #
    async def schedule(self, request: Any, now: float) -> Any:
        # The thunk runs on the far side, so the coordinator decides against
        # *its own* clock at the moment the message lands -- not the stamp the
        # sender took on the way out, which by then is a hop old. Routing that
        # compares every instance's queue would otherwise read a past cluster.
        return await self.hop.call(
            lambda: self.control.schedule(request, self._clock(now))
        )

    def _clock(self, sent: float) -> float:
        """The receiver's clock: the sender's stamp when the hop is free."""
        return sent if not self.hop.rtt else asyncio.get_running_loop().time()

    async def complete(self, plan: Any) -> Any:
        return await self.hop.call(lambda: self.control.complete(plan))

    async def decode_admission(self, plan: Any) -> bool:
        return await self.hop.call(lambda: self.control.decode_admission(plan))

    async def observe_prefill_done(self, inst: str, now: float) -> float:
        return await self.hop.call(
            lambda: self.control.observe_prefill_done(inst, now)
        )

    # -- sends: one-way, so the sender does not block --------------------- #
    def observe_compute_busy(self, inst: str, until: float) -> None:
        self.control.observe_compute_busy(inst, until)

    def observe_decode_state(self, inst: str, finishes: Sequence[float]) -> None:
        self.control.observe_decode_state(inst, finishes)
