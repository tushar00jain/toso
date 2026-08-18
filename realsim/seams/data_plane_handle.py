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

from realsim.seams._plane import LocalPlaneHandle
from realsim.seams.link import ServiceHop

__all__ = ["LocalDataPlaneHandle"]


class LocalDataPlaneHandle(LocalPlaneHandle):
    """A reference to a :class:`DataPlaneService` living in this process.

    Args:
        service: the data-plane service this refers to. Its ``asked`` names the
            members to build endpoints for.
        hop: what reaching it costs. ``None`` is a free hop, which is what a test
            wanting one host and nothing else wants.
    """

    def __init__(self, service: Any, *, hop: Optional[ServiceHop] = None) -> None:
        super().__init__(service, hop=hop)

    @property
    def plane(self) -> Any:
        """The host behind the service, for tests asserting on it."""
        return self.service.plane
