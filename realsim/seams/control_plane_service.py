"""A capability's control plane, off-actor: :class:`ControlPlaneService`.

The **server** side of the control plane an application's own hosts ask, and the
exact counterpart of :class:`realsim.seams.controller_service.ControllerService`. In
a deployment this is a Monarch actor: a process holding the deciding object,
receiving messages, and answering them. Here it is a plain object holding that same
deciding object in this process, receiving ordinary calls instead of messages.

Its surface is the plane's own
------------------------------
:class:`~proposed.plane.ControlPlane` declares a lifecycle and no questions: what a
capability's hosts may *ask* is the capability's to name -- ``decide`` for
``kvcache_sim``'s scheduler, ``select`` for a plane whose answer is a
:class:`~proposed.selector.Selection`, whatever a capability written next needs. So
this service does not name them either: it reads them off the plane it is handed
(:func:`_asked_of`) and forwards each one. Which is all a service is -- the surface is
the capability's, the decisions are the capability's, and neither side has to know
about the other's process.

A capability adding a second question therefore changes nothing here, and nothing in
:mod:`realsim.seams.control_plane_handle` either.

What a host *reports* is not a question and does not come through here -- a fact goes
to :mod:`realsim.seams.dispatcher_service`, the service in front of the dispatcher that
folds it.
"""

from __future__ import annotations

import inspect
from typing import Any, Tuple

from proposed import ControlPlane

__all__ = ["ControlPlaneService"]


def _asked_of(control: Any) -> Tuple[str, ...]:
    """The members of ``control`` a host may ask: its public coroutines, sorted.

    A question is answered, so it is awaited -- which is what makes "is a coroutine"
    the test rather than a list of names this file would have to keep current. Sorted,
    so a handle's endpoints are built in one order whatever ``dir`` reports.

    Two kinds of member are deliberately not questions. Anything underscored is the
    plane's own working, so a coroutine a capability does not want reached says so the
    ordinary way. And nothing :class:`~proposed.plane.ControlPlane` declares is one:
    ``attach`` and ``cluster`` are the lifecycle a *run* drives, and a host reaching
    those would be holding the plane rather than asking it.
    """
    lifecycle = set(vars(ControlPlane))
    return tuple(sorted(
        name for name in dir(type(control))
        if not name.startswith("_")
        and name not in lifecycle
        and inspect.iscoroutinefunction(getattr(control, name, None))
    ))


class ControlPlaneService:
    """The deciding object, as a service reachable from another process.

    Args:
        control: the capability's control plane. This service holds it and forwards
            whatever :attr:`asked` names; it decides nothing.
    """

    def __init__(self, control: Any) -> None:
        self.control = control
        #: What a caller may ask, and what a handle builds its endpoints from.
        self.asked = _asked_of(control)
        for name in self.asked:
            # The bound method itself. A forwarder that wrapped the call would be a
            # place for behaviour to accumulate on the way to the plane that is
            # supposed to own all of it.
            setattr(self, name, getattr(control, name))
