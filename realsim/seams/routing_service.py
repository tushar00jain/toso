"""A production routing-readiness actor behind local endpoint-shaped handles."""

from __future__ import annotations

from typing import Any

from realsim.seams._plane import mount_endpoints

__all__ = ["RoutingService"]


class RoutingService:
    """Expose the production readiness actor through the local simulator."""

    def __init__(self, routing: Any) -> None:
        self.routing = routing
        self.asked = mount_endpoints(self, routing)
