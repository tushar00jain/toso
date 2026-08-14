"""What this capability's decisions read: one sensor per kind of fact it holds.

A sensor holds facts between calls (:class:`proposed.Sensor`); a view is how a
decision reaches one (:mod:`kvcache_sim.control._view`). Three of them, and what
separates them is who writes each:

* :class:`ClusterSensor` -- the predicted prefill queues and observed decode batches,
  written by the hosts over a seam, so it is the one
  :class:`~proposed.deployment.NotifiedSensor` here and the one the run fronts with a
  service. The facts a host reports live with it, and so does the fold;
* :class:`ReservationSensor` and :class:`RoutedPullSensor` -- what this plane decided
  and has not yet seen carried out. Written and read in this process by the plane that
  decided, so they declare no ``notify`` and no service reaches them.

Folder-private, all three: what a decision may read is the view, not the sensor
behind it.
"""

from ._cluster import (
    ClusterSensor, Committed, ComputeBusy, DecodeState, PrefillFinished,
)
from ._pending import Reservation, ReservationSensor, RoutedPullSensor

__all__ = [
    "ClusterSensor",
    "Committed",
    "ComputeBusy",
    "DecodeState",
    "PrefillFinished",
    "Reservation",
    "ReservationSensor",
    "RoutedPullSensor",
]
