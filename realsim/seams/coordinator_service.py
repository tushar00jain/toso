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

Two forwarders, and that is the whole file
------------------------------------------
It used to spell out six, named after one application's questions -- ``schedule``,
``decode_admission``, three ``observe_*`` -- which meant this generic harness file
carried a KV-cache vocabulary, and a second application would have had to come here
and add its own. Now the questions live in the *payload*: a demand is a value the
application defines, and this service neither names nor inspects one. A second
application needs no change here at all.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["CoordinatorService"]


class CoordinatorService:
    """The deciding object, as a service reachable from another process.

    Args:
        control: the capability's control plane -- a
            :class:`proposed.coordinator.Coordinator`, which declares these same
            members for the side that *implements* them. This service holds it and
            forwards; it decides nothing.
    """

    def __init__(self, control: Any) -> None:
        self.control = control

    # -- proposed.deployment.Coordinator ------------------------------------ #
    async def decide(self, demand: Any) -> Optional[Any]:
        return await self.control.decide(demand)

    async def observe(self, fact: Any) -> None:
        # Awaited from outside (a message crosses a boundary either way), handled
        # without suspending: a body that cannot yield cannot lose a fact halfway
        # through learning it.
        self.control.observe(fact)
