"""What this capability's decisions read: one sensor per kind of fact it holds.

A sensor holds facts between calls (:class:`proposed.Sensor`); selectors resolve one
by its declared type. Four of them, and what
separates them is who writes each:

* :class:`ClusterSensor` -- the predicted prefill queues and observed decode batches,
  moved by what the hosts report;
* :class:`ReservationSensor` and :class:`RoutedPullSensor` -- what this plane decided
  and has not yet seen carried out, moved by the decision that took it and by the one
  that says it has happened;
* :class:`SourceLoad` -- how much this plane has sent at each source, moved by the same
  decision and read by a ranking that spreads reads over equally good ones.

None is reached from outside this process. Every write is an action
(:mod:`kvcache_sim.control._sensor._action`) dispatched into this plane's one
dispatcher, so an accepted decision moves every sensor it touches or none of them. What
the run fronts with a service is that dispatcher, never a sensor.

Folder-private, all four: only control-plane selectors read them.
"""

from ._action import (
    Committed, ComputeBusy, DecodeState, FetchAnswered, PrefillFinished,
)
from ._cluster import ClusterSensor
from ._load import SourceLoad
from ._pending import Reservation, ReservationSensor, RoutedPullSensor

__all__ = [
    "ClusterSensor",
    "Committed",
    "ComputeBusy",
    "DecodeState",
    "FetchAnswered",
    "PrefillFinished",
    "Reservation",
    "ReservationSensor",
    "RoutedPullSensor",
    "SourceLoad",
]
