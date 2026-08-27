"""Local handle and actor-mesh stand-ins for routing services."""

from __future__ import annotations

from typing import Any, Optional

from realsim.seams.link import LocalEndpoint, ServiceHop
from realsim.seams.routing_service import RoutingService

__all__ = ["LocalRoutingServiceHandle"]


class LocalRoutingServiceHandle:
    """Endpoint-shaped reference to one local routing service."""

    def __init__(
        self,
        service: RoutingService,
        *,
        hop: Optional[ServiceHop] = None,
    ) -> None:
        self.service = service
        self.hop = hop if hop is not None else ServiceHop()
        self.wait_ready = LocalEndpoint(service.wait_ready, self.hop)
        self.notify_ready = LocalEndpoint(service.notify_ready, self.hop)

    @property
    def routing(self) -> Any:
        return self.service.routing
