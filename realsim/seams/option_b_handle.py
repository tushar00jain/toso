"""Local handle and actor-mesh stand-ins for Option B services."""

from __future__ import annotations

from typing import Any, Optional

from realsim.seams.link import LocalEndpoint, ServiceHop
from realsim.seams.option_b_service import OptionBService

__all__ = ["LocalOptionBServiceHandle"]


class LocalOptionBServiceHandle:
    """Endpoint-shaped reference to one local Option B service."""

    def __init__(
        self,
        service: OptionBService,
        *,
        hop: Optional[ServiceHop] = None,
    ) -> None:
        self.service = service
        self.hop = hop if hop is not None else ServiceHop()
        self.put = LocalEndpoint(service.put, self.hop)
        self.get = LocalEndpoint(service.get, self.hop)
        self.wait_ready = LocalEndpoint(service.wait_ready, self.hop)
        self.notify_ready = LocalEndpoint(service.notify_ready, self.hop)

    @property
    def option_b(self) -> Any:
        return self.service.option_b
