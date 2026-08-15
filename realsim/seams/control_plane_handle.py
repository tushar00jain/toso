"""What a caller holds to reach a control plane: :class:`LocalControlPlaneHandle`.

The **client** side, and only that -- the third of the pair this package builds for
each service (see :mod:`realsim.seams.controller_handle`,
:mod:`realsim.seams.volume_handle`), for the other service a serving host talks to.
A capability whose control plane ranks over cluster-wide state -- kvcache's
scheduler holds every instance's queue, cache and decode occupancy, and serializes
routing decisions cluster-wide, so no serving host can hold it -- does not reach it
by holding the object. It reaches it the way it reaches the store: through a
handle, over calls that carry values.

This class is that reference, for a service living in this process instead of
another one: an endpoint per member of
:class:`realsim.seams.control_plane_service.ControlPlaneService` -- the server side,
which holds the deciding object -- and the single place a round trip is charged. In a
deployment it is Monarch's own handle and nothing on either side changes shape,
which is what the ``Local`` in the name is for: it names the three siblings that
disappear, not a different kind of thing.

One endpoint per member of the surface, and the surface is the plane's
---------------------------------------------------------------------
The members are not named here. They are read off the plane the service holds
(:func:`~realsim.seams.control_plane_service.asked_of`) and turned into one endpoint
each, because *which* questions a capability answers is the capability's to say --
:class:`~realsim.seams.controller_handle.LocalControllerHandle` can name its five
because the directory's surface is declared in ``proposed``, and a control plane's is
not. So a capability written next needs no change here, however many questions it
answers.

What a host reports is not a question and reaches its own service
(:mod:`realsim.seams.dispatcher_handle`).

Calls and sends
---------------
The member comes back as a :class:`~realsim.seams.link.LocalEndpoint`, so the
*caller* chooses how to send, exactly as it would with Monarch:
``decide.call_one(subject, me)`` is a call, awaited, paying the hop twice because
the caller is blocked out and back.

That is why the surface is endpoints and not methods. A handle offering methods
would be a shape Monarch does not have, so swapping in a real actor handle would
mean editing every caller -- and "the `[S]` piece disappears and nothing changes
shape" is the claim this package is making.

Cost
----
The hop is handed in, as it is to
:class:`~realsim.seams.controller_handle.LocalControllerHandle`, and defaults to
free: awaiting an endpoint whose hop is ``0.0`` never suspends, so a default run is
byte-identical to calling the object directly -- the seam is structure, not a
behaviour change. Resolving what it costs is the job of whoever builds a run's
control plane (:class:`realsim.simulation.Simulation`), for the reason the
directory resolves its own in ``make_controller_adapter``: the ambient
``TOSO_CONTROL_RTT`` (or ``--control-rtt``) is read once, where the object is
built, rather than by each object that might want it.

A non-zero hop lands where it belongs: in front of every routing decision, and
therefore in TTFT. A control plane reads its own clock, so a decision made over a
non-zero hop is made at the time the request *arrived*, not the time the sender
stamped it.
"""

from __future__ import annotations

from typing import Any, Optional

from realsim.seams.link import LocalEndpoint, ServiceHop

__all__ = ["LocalControlPlaneHandle"]


class LocalControlPlaneHandle:
    """A reference to a :class:`ControlPlaneService` living in this process.

    Args:
        service: the control-plane service this refers to. Its ``asked`` names the
            members to build endpoints for.
        hop: what reaching it costs. ``None`` is a free hop, which is what a test
            wanting a control plane and nothing else wants; a run builds one from
            :attr:`sim_common.config.SimConfig.control_rtt`.
    """

    def __init__(self, service: Any, *, hop: Optional[ServiceHop] = None) -> None:
        self.service = service
        self.hop = hop if hop is not None else ServiceHop()
        #: The members this handle offers, in the order the endpoints were built.
        self.asked = tuple(service.asked)
        for name in self.asked:
            setattr(self, name, LocalEndpoint(getattr(service, name), self.hop))

    @property
    def control(self) -> Any:
        """The deciding object behind the service, for tests asserting on it."""
        return self.service.control
