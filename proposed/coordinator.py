"""The coordinator a capability *writes*: :class:`Coordinator`.

Some control planes are consulted inside the store -- a
:class:`~proposed.policy.Policy` runs in the controller's ``locate_volumes``. Others
have to run as their own service, because the decision needs the cluster-wide
picture no single host holds: every instance's queue, cache and load. This is the
abstract base for the second kind: subclass it and a capability *is* a coordinator,
the same way subclassing ``Policy`` makes it a source policy.

Two shapes, and this is the author's one
----------------------------------------
:class:`proposed.deployment.Coordinator` declares the same decisions for the
opposite side of the boundary: what a *caller* reaches, through an endpoint per
member (``schedule.call_one(request)``), because a caller holds a reference to a
running service rather than the object. This module is what the author of that
service inherits from. The same split :class:`proposed.deployment.Controller` has
against the handle a client holds, for the same reason -- and the reason both can be
declared without either importing the other is that neither is the implementation.

Values in, values out
---------------------
Every argument and every return is a value; nothing here takes a handle to a
data-plane object, and nothing here is a field the data plane reads. That is what
lets the boundary be inserted: the object and the reference to it have the same
shape, so a serving host cannot tell whether the coordinator it is talking to is in
this process or another one.

The payloads are therefore ``Any``, for the reason given in
:mod:`proposed.deployment`: a request, a plan and a completion are the
*application's* types, and this package cannot import an application. What is
declared is the surface -- which decisions exist, and which way each one travels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from proposed.plane import ControlPlane

__all__ = ["Coordinator"]


class Coordinator(ControlPlane, ABC):
    """The deciding half a data plane may use: values in, values out.

    This is the port between the two planes, and it is deliberately the *whole*
    port -- a serving host that held anything else would be reaching into another
    service. ``check_structure.py`` rule 6 enforces that: a ``data/`` module
    annotating a field with this class may name only these members on it (no other
    attribute, no subscript, no ``getattr``).

    Abstract, and a base class rather than a ``Protocol``, so it is the same kind of
    thing as :class:`~proposed.policy.Policy`: a control plane declares which
    surfaces it implements in its bases, and both answers are read the same way. A
    capability that plays both roles says so once --
    ``class Scheduler(Policy, Coordinator)`` -- which is what
    ``kvcache_sim``'s schedulers do, because the peer they price a pull against is
    the peer they later tell the directory to serve.

    The four request/response members are awaitable and the two observations are
    not, which is the *author's* side of the actor distinction: an implementation
    that must answer has to be able to suspend, while one that is only told
    something cannot. Whether the **sender** waits is the sender's choice, not this
    surface's -- it picks ``call_one`` or ``broadcast`` on the endpoint it holds (see
    :class:`proposed.deployment.Coordinator`). The two sends are the ones the decode
    engine drives, and a per-step notification that blocked the stepping loop would
    be modelling something no deployment would build.

    Under simulation :class:`realsim.seams.coordinator_service.CoordinatorService`
    holds an implementation of this and answers the caller's surface in front of it,
    which is what a Monarch actor would do in a deployment.
    """

    @abstractmethod
    async def schedule(self, request: Any) -> Optional[Any]:
        """Route one request, or ``None`` to reject it (SLO / overload).

        No clock is passed in. A coordinator reads its own -- ``view.now()`` -- because
        a request arrives when it arrives: over a non-zero hop the sender's stamp is
        already stale, and routing that compares it against every instance's queue
        would be reading the cluster in the past.
        """

    @abstractmethod
    async def complete(self, plan: Any) -> Any:
        """What to publish and what to evict, once the planned work has finished."""

    @abstractmethod
    async def decode_admission(self, plan: Any) -> bool:
        """May this accepted request enter the stage it is queued for now?"""

    @abstractmethod
    async def observe_prefill_done(self, inst: str, now: float) -> float:
        """Report the clock the real ops reached; return the corrected queue tail."""

    @abstractmethod
    def observe_compute_busy(self, inst: str, until: float) -> None:
        """Report work occupying a **coupled** instance's compute until ``until``."""

    @abstractmethod
    def observe_decode_state(self, inst: str, finishes: Sequence[float]) -> None:
        """Report ``inst``'s live batch as estimated finish times.

        One entry per request currently running or queued there, so its length is
        the occupancy and its values answer "still running at ``t``?". Sent whenever
        the batch changes, which is the only time either answer moves.
        """
