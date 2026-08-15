"""What this capability's decisions read: the fan-out this plane has planned.

A sensor holds facts between calls (:class:`proposed.Sensor`); a view is how a
decision reaches one (:mod:`dedup_sim.control._view`). One sensor here:
:class:`FanoutSensor`, the tree and the puts it is owed. No waiting -- that is the
commit of the action it folds (:meth:`proposed.dispatch.Dispatcher.gate`), and nothing
holds a record of who is parked.

Nothing reaches it from outside this process, though a host does write it. A reader's
data plane reports its own landed put, but what arrives is one
:class:`proposed.dispatch.Stored`, dispatched into this plane's
:class:`proposed.dispatch.Dispatcher`, which folds it by calling this sensor's own
reducer -- so what the run fronts with a service is the dispatcher, not the sensor.
A service of its own in front of the sensor would be a second seam for the same fact,
and nothing would order it against the directory's: a stale fan-out is a wrong readiness
gate, and a gate nothing opens hangs the requester behind it, where a stale queue
elsewhere would only misprice a candidate.

Folder-private: what a decision may read is the view, not the sensor behind it.
"""

from ._fanout import FanoutSensor

__all__ = ["FanoutSensor"]
