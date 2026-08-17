"""Stable facts and calculations shared by every decision in one run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

from proposed.topology import Endpoint

__all__ = ["Environment"]


@dataclass(frozen=True)
class Environment:
    """Stable facts and calculations shared by every decision in one run.

    Args:
        topology: ``volume_id -> Endpoint`` for the run.
        profile: the run's machine profile. ``None`` when decisions price no reads.
    """

    topology: Mapping[str, Endpoint]
    profile: Optional[Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology", MappingProxyType(dict(self.topology)))

    def read_time(self, src_id: str, dst_id: str, nbytes: int) -> float:
        """Seconds to read ``nbytes`` from ``src_id`` at ``dst_id``."""
        if self.profile is None:
            raise RuntimeError(
                "this run supplied no machine profile, so no read can be priced"
            )
        return self.profile.read_time(
            self.topology[src_id], self.topology[dst_id], nbytes
        )

    def now(self) -> float:
        """The running loop's real or virtual clock, in seconds."""
        return asyncio.get_running_loop().time()
