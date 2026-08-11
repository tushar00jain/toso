"""A capability's control plane, off-actor: :class:`CoordinatorService`.

The **server** side of a coordinator, and the exact counterpart of
:class:`realsim.seams.controller_service.ControllerService`. In a deployment this is
a Monarch actor: a process holding the deciding object, receiving messages, and
answering them. Here it is a plain object holding that same deciding object in this
process, with the same methods, receiving ordinary calls instead of messages.

It implements :class:`proposed.deployment.Coordinator` -- the service surface
declared there -- by storing the capability's control plane and calling it. Which is
all a service is: the surface is generic (it is in ``proposed``), the decisions are
the capability's, and neither has to know about the other's process.

Nothing about a capability is named here. That is what makes the split possible at
this layer: the member list comes from ``proposed``, not from ``kvcache_sim``, so the
harness can spell out six forwarders without knowing what any of them decide.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

__all__ = ["CoordinatorService"]


class CoordinatorService:
    """The deciding object, as a service reachable from another process.

    Args:
        control: the capability's control plane -- kvcache's scheduler, which
            declares the same members on its own abstract base. This service holds
            it and forwards; it decides nothing.
    """

    def __init__(self, control: Any) -> None:
        self.control = control

    # -- proposed.deployment.Coordinator ------------------------------------ #
    async def schedule(self, request: Any) -> Optional[Any]:
        return await self.control.schedule(request)

    async def complete(self, plan: Any) -> Any:
        return await self.control.complete(plan)

    async def decode_admission(self, plan: Any) -> bool:
        return await self.control.decode_admission(plan)

    async def observe_prefill_done(self, inst: str, now: float) -> float:
        return await self.control.observe_prefill_done(inst, now)

    async def observe_compute_busy(self, inst: str, until: float) -> None:
        self.control.observe_compute_busy(inst, until)

    async def observe_decode_state(
        self, inst: str, finishes: Sequence[float]
    ) -> None:
        self.control.observe_decode_state(inst, finishes)
