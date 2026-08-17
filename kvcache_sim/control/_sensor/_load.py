"""How much this plane has sent at each source: :class:`SourceLoad`.

The load signal behind spreading reads over the replicas of a hot prefix. Reuse value
is a property of the prefix, so replicas of one hold identical rank and the id
tie-break sends every read to the same volume -- this is the thing that changes, so
the tie can be broken on it (:data:`proposed.selector.Balance`).

Written by the decision that names a source and read by the ranking that named it, off
the action every accepted decision already dispatches
(:class:`~kvcache_sim.control._sensor.Committed`), so a source's load rises where the
pull it was priced for was decided.

It implements the shared :class:`proposed.sensors.LoadSensor` reading.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, Mapping

from proposed import LoadSensor, VolumeId
from proposed.dispatch import Fold

from ._action import Committed

__all__ = ["SourceLoad"]


class SourceLoad(LoadSensor):
    """Per-volume count of the decisions that named it as a source.

    Read through the selector's declared :class:`proposed.sensors.LoadSensor`.

    Counts only a decision that priced a pull: one that recomputes the gap sends nothing
    to anybody, and counting it would load a volume nobody is going to read.
    """

    def __init__(self) -> None:
        self._named: Dict[VolumeId, int] = {}
        self._folds: Dict[type, Fold] = {Committed: self._committed}

    @property
    def folds(self) -> Mapping[type, Fold]:
        """What this sensor folds, by action type."""
        return MappingProxyType(self._folds)

    def _committed(self, action: Committed) -> None:
        """Count the source this decision priced its pull against, if it priced one."""
        source = action.response.plan.reuse_source
        if source is not None:
            self._named[source] = self._named.get(source, 0) + 1

    def named(self) -> Mapping[VolumeId, int]:
        """``volume -> decisions that named it``. Absent means none."""
        return MappingProxyType(self._named)
