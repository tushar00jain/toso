"""A capability's data plane, off-actor: :class:`DataPlaneService`.

The **server** side of one host, as somebody off the box reaches it -- the exact
counterpart of :class:`realsim.seams.control_plane_service.ControlPlaneService` for
the executing half. In a deployment this is a Monarch actor: a process holding the
host, receiving messages, and answering them. Here it is a plain object holding
that same host in this process, receiving ordinary calls instead of messages.

Its surface is the plane's own, read off it (:func:`_asked_of`) for the reason the
control plane's is: what a capability's hosts answer is the capability's to name.

A plane fronted here may not hold its peers (:func:`proposed.routed.peerless`): a
host that answers with an address and could also call it is a forwarder, and this is
where a host becomes reachable, so this is where that is refused.
"""

from __future__ import annotations

import inspect
from typing import Any, Tuple

from proposed import DataPlane
from proposed.routed import peerless

__all__ = ["DataPlaneService"]


def _asked_of(plane: Any) -> Tuple[str, ...]:
    """The members of ``plane`` a caller may reach: its public coroutines, sorted.

    :func:`realsim.seams.control_plane_service._asked_of`'s rule, applied to the
    other half: a call is awaited, anything underscored is the plane's own working,
    and nothing :class:`~proposed.plane.DataPlane` declares is a member -- ``attach``
    is the lifecycle a *run* drives, and ``routes`` is what a caller walks *with*.
    """
    lifecycle = set(vars(DataPlane))
    return tuple(sorted(
        name for name in dir(type(plane))
        if not name.startswith("_")
        and name not in lifecycle
        and inspect.iscoroutinefunction(getattr(plane, name, None))
    ))


class DataPlaneService:
    """One host, as a service reachable from another process.

    Args:
        plane: the capability's data plane for one node. This service holds it and
            forwards whatever :attr:`asked` names; it executes nothing.
    """

    def __init__(self, plane: Any) -> None:
        peerless(plane)
        self.plane = plane
        #: What a caller may reach, and what a handle builds its endpoints from.
        self.asked = _asked_of(plane)
        for name in self.asked:
            # The bound method itself, as on the control side: a forwarder that
            # wrapped the call would be a place for behaviour to accumulate on the
            # way to the host that is supposed to own all of it.
            setattr(self, name, getattr(plane, name))
