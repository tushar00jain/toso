"""A capability's control plane, off-actor: :class:`PlacementService`.

The **server** side of the selector an application's own hosts ask, and the exact
counterpart of :class:`realsim.seams.controller_service.ControllerService`. In a
deployment this is a Monarch actor: a process holding the deciding object,
receiving messages, and answering them. Here it is a plain object holding that same
deciding object in this process, with the same method, receiving ordinary calls
instead of messages.

It implements :class:`proposed.policy.AnySelector`'s one member by storing the
capability's control plane and calling it. Which is all a service is: the surface
is generic (it is in ``proposed``), the decisions are the capability's, and neither
has to know about the other's process.

One forwarder, and that is the whole file
-----------------------------------------
The question lives in the *subject*: what a host asks about is a value the
application defines, and this service neither names nor inspects one, so a second
application needs no change here at all. What a host *reports* is not a question
and does not come through here -- a fact goes to
:mod:`realsim.seams.cluster_model_service`, the service in front of the model it
corrects.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PlacementService"]


class PlacementService:
    """The deciding object, as a service reachable from another process.

    Args:
        control: the capability's control plane -- a
            :class:`proposed.policy.AnySelector`. This service holds it and forwards;
            it decides nothing.
    """

    def __init__(self, control: Any) -> None:
        self.control = control

    # -- proposed.policy.AnySelector ------------------------------------------ #
    async def select(self, subject: Any, requester: str) -> Any:
        return await self.control.select(subject, requester)
