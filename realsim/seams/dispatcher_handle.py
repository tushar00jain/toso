"""What a host holds to report a fact: :class:`LocalDispatcherHandle`.

The **client** side, and only that -- the fourth of the pair this package builds for
each service (see :mod:`realsim.seams.controller_handle`,
:mod:`realsim.seams.control_plane_handle`, :mod:`realsim.seams.volume_handle`). A
serving host does not hold what it reports into; it holds a reference to it, and
Monarch's reference is not the same shape as the actor: ``@endpoint`` turns each
method into an ``EndpointProperty``, so the attribute resolves to an endpoint object
and the call reads

    await dispatcher.dispatch.call_one(DecodeState(me, finishes))

This class is that reference, for a service living in this process instead of
another one: an endpoint per member of
:class:`realsim.seams.dispatcher_service.DispatcherService` -- one, because reporting an
action is the whole of what a host does over this seam. In a deployment it is Monarch's
own handle and nothing on either side changes shape.

Facts are called, not broadcast
-------------------------------
:meth:`~realsim.seams.link.LocalEndpoint.broadcast` refuses a member that suspends,
and a member that pays a non-zero hop suspends -- firing it off would need a task,
and a task reorders the run. That refusal is the right answer rather than an
obstacle: a host reports a completion and immediately asks the question that has to
be decided against it, and only the reply orders the two at the *receiver*.

Cost
----
The hop is handed in and defaults to free, as
:class:`~realsim.seams.control_plane_handle.LocalControlPlaneHandle`'s does. A run
builds it from the same :attr:`sim_common.config.SimConfig.control_rtt`: the
dispatcher is held by the control plane that reads what it folds, so reaching it is that
same boundary at that same distance. At ``0.0`` -- the default -- awaiting an endpoint
never suspends, so a run is byte-identical to calling the dispatcher directly; a
non-zero one lands in front of every fact a host reports, and therefore inside the
decode cadence that reports them.
"""

from __future__ import annotations

from typing import Any, Optional

from realsim.seams.link import LocalEndpoint, ServiceHop

__all__ = ["LocalDispatcherHandle"]


class LocalDispatcherHandle:
    """A reference to a :class:`DispatcherService` living in this process.

    Args:
        service: the dispatcher service this refers to.
        hop: what reaching it costs. ``None`` is a free hop, which is what a test
            wanting a dispatcher and nothing else wants; a run builds one from
            :attr:`sim_common.config.SimConfig.control_rtt`.
    """

    def __init__(self, service: Any, *, hop: Optional[ServiceHop] = None) -> None:
        self.service = service
        self.hop = hop if hop is not None else ServiceHop()
        self.dispatch = LocalEndpoint(service.dispatch, self.hop)

    @property
    def dispatcher(self) -> Any:
        """What is behind the service, for tests asserting on it."""
        return self.service.dispatcher
