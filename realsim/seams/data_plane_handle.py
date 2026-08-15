"""What a caller holds to reach one host: :class:`LocalDataPlaneHandle`.

The **client** side, and only that -- the fourth of the pair this package builds
per service (see :mod:`realsim.seams.control_plane_handle`), for the half that
executes. A caller is off the box, so reaching a host is a boundary like reaching
the directory or the control plane, and it is charged like one: one
:class:`~realsim.seams.link.LocalEndpoint` per member of
:class:`~realsim.seams.data_plane_service.DataPlaneService`, over a hop shared by
every host in the run.

The members are not named here. They are read off the plane the service holds, so a
capability whose hosts answer three questions or thirty needs no change here -- the
same argument as on the control side, and the same reason the surface is endpoints
rather than methods: swapping in a real actor handle must not edit any caller.

The hop is handed in and defaults to free, resolved once where the planes are fronted
(:meth:`realsim.simulation.Simulation.front_plane`) from
:attr:`sim_common.config.SimConfig.client_rtt`. It is paid out and back per leg, so a
request redirected twice pays it three times.
"""

from __future__ import annotations

from typing import Any, Optional

from realsim.seams.link import LocalEndpoint, ServiceHop

__all__ = ["LocalDataPlaneHandle"]


class LocalDataPlaneHandle:
    """A reference to a :class:`DataPlaneService` living in this process.

    Args:
        service: the data-plane service this refers to. Its ``asked`` names the
            members to build endpoints for.
        hop: what reaching it costs. ``None`` is a free hop, which is what a test
            wanting one host and nothing else wants.
    """

    def __init__(self, service: Any, *, hop: Optional[ServiceHop] = None) -> None:
        self.service = service
        self.hop = hop if hop is not None else ServiceHop()
        #: The members this handle offers, in the order the endpoints were built.
        self.asked = tuple(service.asked)
        for name in self.asked:
            setattr(self, name, LocalEndpoint(getattr(service, name), self.hop))

    @property
    def plane(self) -> Any:
        """The host behind the service, for tests asserting on it."""
        return self.service.plane
