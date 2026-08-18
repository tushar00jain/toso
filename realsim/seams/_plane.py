"""Shared local mounting for control-plane and data-plane endpoints."""

from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple

from monarch._src.actor.endpoint import EndpointProperty

from realsim.seams.link import LocalEndpoint, ServiceHop

__all__ = ["LocalPlaneHandle", "endpoint_names", "mount_endpoints"]


def endpoint_names(target: Any) -> Tuple[str, ...]:
    """The explicitly declared endpoints on ``target``, sorted."""
    return tuple(sorted(
        name for name in dir(type(target))
        if not name.startswith("_")
        and isinstance(
            inspect.getattr_static(type(target), name, None), EndpointProperty
        )
    ))


def mount_endpoints(service: Any, target: Any) -> Tuple[str, ...]:
    """Bind ``target``'s endpoint bodies onto a local service."""
    names = endpoint_names(target)
    for name in names:
        bound = getattr(target, name)
        if not callable(bound):
            declared = inspect.getattr_static(type(target), name)
            bound = declared._method.__get__(target, type(target))
        setattr(service, name, bound)
    return names


class LocalPlaneHandle:
    """An endpoint-shaped reference to a local plane service."""

    def __init__(self, service: Any, *, hop: Optional[ServiceHop] = None) -> None:
        self.service = service
        self.hop = hop if hop is not None else ServiceHop()
        self.asked = tuple(service.asked)
        for name in self.asked:
            setattr(self, name, LocalEndpoint(getattr(service, name), self.hop))
