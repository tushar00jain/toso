"""A capability's control plane, off-actor: :class:`ControlPlaneService`.

The **server** side of the control plane an application's own hosts ask, and the
exact counterpart of :class:`realsim.seams.controller_service.ControllerService`. In
a deployment this is a Monarch actor: a process holding the deciding object,
receiving messages, and answering them. Here it is a plain object holding that same
deciding object in this process, receiving ordinary calls instead of messages.

Its surface is the plane's own
------------------------------
:class:`~proposed.plane.ControlPlane` declares a lifecycle and no questions. A
capability marks its own questions with :func:`proposed.endpoint`; this service
mounts those declarations and names none itself.

A capability adding a second question therefore changes nothing here, and nothing in
:mod:`realsim.seams.control_plane_handle` either.

What a host *reports* is not a question and does not come through here -- a fact goes
to :mod:`realsim.seams.dispatcher_service`, the service in front of the dispatcher that
folds it.
"""

from __future__ import annotations

from typing import Any, Tuple

from realsim.seams._plane import endpoint_names, mount_endpoints

__all__ = ["ControlPlaneService"]


def _asked_of(control: Any) -> Tuple[str, ...]:
    """The endpoints ``control`` explicitly offers to hosts."""
    return endpoint_names(control)


class ControlPlaneService:
    """The deciding object, as a service reachable from another process.

    Args:
        control: the capability's control plane. This service holds it and forwards
            whatever :attr:`asked` names; it decides nothing.
    """

    def __init__(self, control: Any) -> None:
        self.control = control
        #: What a caller may ask, and what a handle builds its endpoints from.
        self.asked = mount_endpoints(self, control)
