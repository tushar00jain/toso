"""A capability's data plane, off-actor: :class:`DataPlaneService`.

The **server** side of one host, as somebody off the box reaches it -- the exact
counterpart of :class:`realsim.seams.control_plane_service.ControlPlaneService` for
the executing half. In a deployment this is a Monarch actor: a process holding the
host, receiving messages, and answering them. Here it is a plain object holding
that same host in this process, receiving ordinary calls instead of messages.

Its surface is the plane's explicit :func:`proposed.endpoint` declarations. What a
capability's hosts answer is the capability's to name.

A plane fronted here may not hold its peers (:func:`proposed.routed.peerless`): a
host that answers with an address and could also call it is a forwarder, and this is
where a host becomes reachable, so this is where that is refused.
"""

from __future__ import annotations

from typing import Any, Tuple

from proposed.routed import peerless
from realsim.seams._plane import endpoint_names, mount_endpoints

__all__ = ["DataPlaneService"]


def _asked_of(plane: Any) -> Tuple[str, ...]:
    """The endpoints ``plane`` explicitly offers to callers."""
    return endpoint_names(plane)


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
        self.asked = mount_endpoints(self, plane)
