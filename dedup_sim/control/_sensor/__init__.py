"""What this capability's decisions read: the fan-out this plane has planned.

A sensor holds facts between calls (:class:`proposed.Sensor`); a view is how a
decision reaches one (:mod:`dedup_sim.control._view`). One sensor here --
:class:`FanoutSensor`, the tree and the puts it is owed -- beside the gate mechanism
it waits on (:mod:`dedup_sim.control._sensor._readiness`), which holds no facts and
is nobody's sensor.

It derives the **bare** :class:`proposed.Sensor` even though a host does write it. A
reader's data plane reports its own landed put, but that fact arrives over this
plane's surface (:meth:`~dedup_sim.control.routing.Dedup.published`) rather than a
sensor service of its own, which is what orders the publish before the next ask: a
stale fan-out is a wrong readiness gate, and a gate nothing opens hangs the requester
behind it, where a stale queue elsewhere would only misprice a candidate. Fronting it
as a :class:`proposed.NotifiedSensor` would add a second seam for the same fact, and
nothing would order the two.

Folder-private: what a decision may read is the view, not the sensor behind it.
"""

from ._fanout import FanoutSensor

__all__ = ["FanoutSensor"]
