"""What this capability's decisions read: one sensor per kind of fact it holds.

A sensor holds facts between calls (:class:`proposed.Sensor`); a view is how a
decision reaches one (:mod:`kvcache_sim.control._view`). Four of them, and what
separates them is who writes each:

* :class:`ClusterSensor` -- the predicted prefill queues and observed decode batches,
  moved by what the hosts report;
* :class:`ReservationSensor` and :class:`RoutedPullSensor` -- what this plane decided
  and has not yet seen carried out, moved by the decision that took it and by the one
  that says it has happened;
* :class:`SourceLoad` -- how much this plane has sent at each source, moved by the same
  decision and read by a ranking that spreads reads over equally good ones.

None of the three is reached from outside this process. Every write is an action
(:mod:`kvcache_sim.control._sensor._action`) dispatched into this plane's one
:class:`proposed.dispatch.Dispatcher`, which folds it into every sensor it moves and
commits them together -- so an accepted decision moves all three or none, and what the
run fronts with a service is that dispatcher rather than a sensor.

Folder-private, all four: what a decision may read is the view, not the sensor
behind it.
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
