"""Monarch endpoints that remain callable on an unspawned actor.

The descriptor stays an ``EndpointProperty`` for actor discovery while binding its
method for local simulations and unit tests.
"""

from __future__ import annotations

from typing import Any, Callable, overload, TypeVar

from monarch._src.actor.endpoint import EndpointProperty  # type: ignore[import-untyped]

__all__ = ["endpoint"]

_M = TypeVar("_M")


class _BoundEndpoint(EndpointProperty):
    def __init__(self, declared: EndpointProperty, instance: Any) -> None:
        super().__init__(
            declared._method,
            declared._propagator,
            declared._explicit_response_port,
            declared._instrument,
        )
        self._instance = instance

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._method(self._instance, *args, **kwargs)


class _LocalEndpoint(EndpointProperty):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._method(*args, **kwargs)

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return _BoundEndpoint(self, instance)


@overload
def endpoint(method: _M) -> _M: ...


@overload
def endpoint(
    method: None = None,
    *,
    propagate: Any = None,
    explicit_response_port: bool = False,
    instrument: bool = True,
) -> Callable[[_M], _M]: ...


def endpoint(
    method: Any = None,
    *,
    propagate: Any = None,
    explicit_response_port: bool = False,
    instrument: bool = True,
) -> Any:
    """Declare a Monarch endpoint and permit direct calls before it is spawned."""
    if method is None:
        return lambda member: endpoint(
            member,
            propagate=propagate,
            explicit_response_port=explicit_response_port,
            instrument=instrument,
        )
    return _LocalEndpoint(
        method,
        propagator=propagate,
        explicit_response_port=explicit_response_port,
        instrument=instrument,
    )
