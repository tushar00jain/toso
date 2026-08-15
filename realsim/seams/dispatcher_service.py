"""Where an application's hosts report, off-actor: :class:`DispatcherService`.

The **server** side of where a control plane has facts arrive, and the fourth of
the pair this package builds for each service
(:mod:`realsim.seams.controller_service`,
:mod:`realsim.seams.control_plane_service`,
:mod:`realsim.seams.volume_service`). In a deployment this is a Monarch actor: a
process holding that object, receiving the actions its hosts report, and folding them.
Here it is a plain object holding the same one in this process, with the same member,
receiving ordinary calls instead of messages.

It fronts the application's :class:`proposed.dispatch.Dispatcher`
(:attr:`proposed.plane.ControlPlane.dispatcher`) by holding it and forwarding the one
member a reporter reaches, which is the whole of what crosses: an action goes in and the
reply carries nothing. Named for that surface, as each pair here is. The split is the
same one the directory makes: the thing that *is* the receiver is the receiver, and the
thing a host *holds* is
:class:`realsim.seams.dispatcher_handle.LocalDispatcherHandle`, a different shape
(endpoints) for a different reason (it stands in for the process boundary).

What crosses here and what does not
-----------------------------------
This endpoint carries what a *host* reports. What a control plane dispatches to itself
takes the synchronous half instead
(:meth:`proposed.dispatch.Dispatcher.dispatch_sync`), by plain local call -- the same
co-location a :class:`~proposed.selector.KeySelector` has with the directory it senses
through ``locate_raw``, and for the same reason: a decision formed against a read that
could suspend is a decision formed against a picture that changed halfway through.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DispatcherService"]


class DispatcherService:
    """Where an application reports its facts, as a service another process reaches.

    Args:
        dispatcher: the application's :class:`proposed.dispatch.Dispatcher` -- where the
            action is folded. This service holds it and forwards; it holds nothing
            itself.
    """

    def __init__(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher

    # -- proposed.dispatch.Dispatcher --------------------------------------- #
    async def dispatch(self, action: Any) -> None:
        await self.dispatcher.dispatch(action)
