"""The cluster as control sees it: :class:`ClusterSensor`, and what moves it.

Nothing here executes. What control knows about what the cluster is *doing* is a
*model* corrected by what the hosts report, never a live read -- two things it keeps,
and a host's report is what keeps each of them true:

* the **prefill queue** (:attr:`ClusterSensor.busy_until`) is predicted -- a
  :class:`~kvcache_sim.control._sensor.Committed` plan holds its instance until the
  TTFT it was priced at -- and corrected by
  :class:`~kvcache_sim.control._sensor.PrefillFinished`. It diverges from the wait the
  data plane measures by construction: a candidate is priced for
  ``queue -> transfer -> prefill``, so a remote pull is charged to a device idle
  while the fabric works. Both are recorded side by side
  (:attr:`kvcache_sim.report.metrics.RequestResult.queue_wait` against
  ``predicted_queue_wait``). On a **coupled** instance prefill and decode share one
  accelerator, so each decode step is mirrored back as
  :class:`~kvcache_sim.control._sensor.ComputeBusy`;
* **decode occupancy** is a per-instance list of estimated finish times, replaced
  wholesale by :class:`~kvcache_sim.control._sensor.DecodeState`.

A prefill this plane has *promised* is neither: no host corrects it, and it expires as
it is read, so it is a sensor of its own
(:class:`kvcache_sim.control._sensor.ReservationSensor`).

Nothing reaches this sensor from outside the process holding it. A host's report is an
action dispatched into the plane's :class:`proposed.dispatch.Dispatcher`, which folds it
by calling the reducer below (:attr:`ClusterSensor.folds`), while the control plane
reads it through the view it is composed into
(:class:`kvcache_sim.control._view.ClusterView`).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Sequence

from proposed import Sensor
from proposed.dispatch import Fold

from ._action import Committed, ComputeBusy, DecodeState, PrefillFinished

__all__ = ["ClusterSensor"]


class ClusterSensor(Sensor):
    """Every instance's predicted prefill queue and observed decode batch.

    One per run, and that is load-bearing: a second one starts empty, and an empty
    sensor would report every host idle -- a run that looks healthy and is wrong.
    Two things keep it to one. It is built in a single place, the control plane's
    :meth:`~kvcache_sim.control.scheduler._Scheduler.attach`, which the run calls
    once when the stack exists; and it is keyed by the instances handed to it
    there, so one built for the wrong cluster (or for none) raises on the
    first read instead of answering "idle".

    Args:
        ids: every instance in the run -- the prefill and decode pools may each
            be a subset, but load is tracked over all of them.
    """

    def __init__(self, ids: Sequence[str]) -> None:
        self._busy_until: Dict[str, float] = {i: 0.0 for i in ids}
        # instance -> one estimated finish time per request decoding or queued
        # there. Empty until the data plane reports.
        self._decode_finishes: Dict[str, List[float]] = {i: [] for i in ids}
        # action type -> the bound method that folds it. A table rather than
        # ``functools.singledispatchmethod``, which captures the function registered on
        # this class and so silently ignores a subclass redefining one.
        self._folds: Dict[type, Fold] = {
            ComputeBusy: self._compute_busy,
            DecodeState: self._decode_state,
            PrefillFinished: self._prefill_finished,
            Committed: self._committed,
        }

    # -- what it folds ------------------------------------------------------- #
    @property
    def folds(self) -> Mapping[type, Fold]:
        """:class:`proposed.dispatch.Reducer` -- what it folds, by action type.

        Read-only, so the one way to move this state is to dispatch something that
        moves it.
        """
        return MappingProxyType(self._folds)

    def _compute_busy(self, action: ComputeBusy) -> None:
        """A decode step on a **coupled** instance occupied its compute.

        Only the data plane knows whether prefill and decode share a timeline; a
        disaggregated host never reports this, so decode never touches its
        predicted prefill queue.
        """
        self._busy_until[action.inst] = action.until

    def _decode_state(self, action: DecodeState) -> None:
        """Replace the batch rather than merge into it: the action is the whole of
        what ``inst`` is decoding, so anything it omits has ended."""
        self._decode_finishes[action.inst] = list(action.finishes)

    def _prefill_finished(self, action: PrefillFinished) -> None:
        """Correct the predicted queue with the clock the real ops reached.

        Raises the tail and never lowers it. An early completion leaves the
        instance looking busier than it is until the next request is routed
        against it, whereas lowering on one report would under-count the prefills
        control has promised and not yet seen finish.
        """
        if action.now > self._busy_until[action.inst]:
            self._busy_until[action.inst] = action.now

    def _committed(self, action: Committed) -> None:
        """Hold the prefill instance an accepted decision spoke for."""
        self._busy_until[action.response.prefill] = action.response.plan.done_time

    # -- what ranking against the load reads -------------------------------- #
    @property
    def busy_until(self) -> Mapping[str, float]:
        """Predicted prefill queue tail per instance, read-only.

        A mapping rather than the dict, so the only way to move a tail is to
        dispatch something that moved it.
        """
        return MappingProxyType(self._busy_until)

    def occupancy(self, inst: str) -> int:
        """Requests currently decoding or queued on ``inst``."""
        return len(self._decode_finishes[inst])

    def predict_occupancy(self, inst: str, at_t: float) -> int:
        """How many of those are estimated to still be decoding at ``at_t``."""
        return sum(1 for f in self._decode_finishes[inst] if f > at_t)
