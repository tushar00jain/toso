"""Control's model of the cluster: :class:`KVClusterModel`, and the facts it folds.

Nothing here executes. What control knows about what the cluster is *doing* is a
*model* corrected by what the hosts report, never a live read -- two records, and a
host's fact is what keeps each of them true:

* the **prefill queue** (:attr:`KVClusterModel.busy_until`) is predicted -- a
  :class:`Committed` plan holds its instance until the TTFT it was priced at --
  and corrected by :class:`PrefillFinished`. It diverges from the wait the data
  plane measures by construction: a candidate is priced for
  ``queue -> transfer -> prefill``, so a remote pull is charged to a device idle
  while the fabric works. Both are recorded side by side
  (:attr:`kvcache_sim.report.metrics.RequestResult.queue_wait` against
  ``predicted_queue_wait``). On a **coupled** instance prefill and decode share one
  accelerator, so each decode step is mirrored back as :class:`ComputeBusy`;
* **decode occupancy** is a per-instance list of estimated finish times, replaced
  wholesale by :class:`DecodeState`.

A prefill this plane has *promised* is neither: the control plane both writes and reads
that, and no host corrects it, so it is a record and a sense of its own
(:class:`kvcache_sim.control._pending.Reservations`,
:class:`kvcache_sim.control._view.ReservedSense`).

Folder-private: the port this answers (:class:`proposed.ClusterModel`) is the
surface, and a host reaches it through the seam in front of it
(:class:`realsim.seams.cluster_model_handle.LocalClusterModelHandle`). The control
plane reads and writes it here, in this process, through the sense it is composed into
(:class:`kvcache_sim.control._view.ClusterSense`).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any, Callable, Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING,
)

from proposed import ClusterModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .scheduler import Response

__all__ = [
    "ComputeBusy",
    "DecodeState",
    "PrefillFinished",
    "Committed",
    "KVClusterModel",
]


# -- what a host reports, and what the scheduler promised --------------------- #
# Frozen values, so they cross a process boundary unchanged and cannot be edited
# after they are handed over.


@dataclass(frozen=True)
class PrefillFinished:
    """*Prefill really finished at this clock -- what is the queue tail now?*

    The only thing that tells this control plane its model of an instance's
    prefill queue was wrong: ``now`` is measured independently (the host's
    accelerator serialises its own passes).

    Reported over a call the host waits for, and not for the reply, which carries
    nothing: the decode admission it asks next must be decided against a model that
    has already folded this completion.
    """

    inst: str
    now: float


@dataclass(frozen=True)
class ComputeBusy:
    """A decode step occupied a **coupled** instance's compute until ``until``."""

    inst: str
    until: float


@dataclass(frozen=True)
class DecodeState:
    """``inst``'s live decode batch, as one estimated finish time per request.

    Its length is the occupancy and its values answer "still decoding at ``t``?".
    Reported whenever the batch changes.
    """

    inst: str
    finishes: Tuple[float, ...]


@dataclass(frozen=True)
class Committed:
    """A decision the scheduler accepted: its prefill instance is now spoken for.

    The one fact control tells itself. Sent at commit rather than while pricing,
    so a candidate that lost -- or a decision a gate refused -- leaves nothing
    behind.
    """

    response: "Response"


class KVClusterModel(ClusterModel):
    """Every instance's predicted prefill queue and observed decode batch.

    One per run, and that is load-bearing: a second one starts empty, and an empty
    model would report every host idle -- a run that looks healthy and is wrong.
    Two things keep it to one. It is built in a single place, the control plane's
    :meth:`~kvcache_sim.control.scheduler._Scheduler.attach`, which the run calls
    once when the stack exists; and it is keyed by the instances handed to it
    there, so a model built for the wrong cluster (or for none) raises on the
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
        # fact type -> the bound method that folds it.
        self._folds: Dict[type, Callable[[Any], None]] = {
            ComputeBusy: self._compute_busy,
            DecodeState: self._decode_state,
            PrefillFinished: self._prefill_finished,
            Committed: self._committed,
        }

    # -- the two ways in, one fold ------------------------------------------ #
    async def notify(self, fact: Any) -> None:
        """:class:`~proposed.deployment.ClusterModel` -- fold ``fact`` into state.

        The endpoint a host reports over, and the whole of what crosses the seam.
        """
        self.notify_sync(fact)

    def notify_sync(self, fact: Any) -> None:
        """Fold ``fact`` in, here and now: what co-located control code calls.

        The same fold as :meth:`notify` over a shorter reach, so what a caller
        chooses between is the transport. Only a caller holding the model itself can
        reach this one -- :class:`~proposed.deployment.ClusterModel` declares
        ``notify`` alone and the service in front of it forwards only that -- so a
        host reports over the seam whatever it is co-located with.

        **Not a coroutine, and that is load-bearing**, the same way
        :meth:`proposed.deployment.Controller.locate_raw` is not one: the scheduler
        writes :class:`Committed` in the middle of a routing decision, and a plain
        call cannot let a second decision interleave with the first -- whereas an
        ``await`` would rest that atomicity on the coroutine happening not to
        suspend. That is the reason to call it, not that it is cheaper.

        Dispatch is on the fact's type, through the table bound in ``__init__``
        and not ``functools.singledispatchmethod``, which captures the function
        registered on this class and so silently ignores a subclass redefining
        one.
        """
        fold = self._folds.get(type(fact))
        if fold is None:
            raise TypeError(
                f"{type(self).__name__} is not told {type(fact).__name__}: this "
                f"application's facts are "
                f"{', '.join(sorted(f.__name__ for f in self._folds))}"
            )
        fold(fact)

    def _compute_busy(self, fact: ComputeBusy) -> None:
        """A decode step on a **coupled** instance occupied its compute.

        Only the data plane knows whether prefill and decode share a timeline; a
        disaggregated host never reports this, so decode never touches its
        predicted prefill queue.
        """
        self._busy_until[fact.inst] = fact.until

    def _decode_state(self, fact: DecodeState) -> None:
        """Replace the batch rather than merge into it: the fact is the whole of
        what ``inst`` is decoding, so anything it omits has ended."""
        self._decode_finishes[fact.inst] = list(fact.finishes)

    def _prefill_finished(self, fact: PrefillFinished) -> None:
        """Correct the predicted queue with the clock the real ops reached.

        Raises the tail and never lowers it. An early completion leaves the
        instance looking busier than it is until the next request is routed
        against it, whereas lowering on one report would under-count the prefills
        control has promised and not yet seen finish.
        """
        if fact.now > self._busy_until[fact.inst]:
            self._busy_until[fact.inst] = fact.now

    def _committed(self, fact: Committed) -> None:
        """Hold the prefill instance an accepted decision spoke for."""
        self._busy_until[fact.response.prefill] = fact.response.plan.done_time

    # -- what ranking against the load reads -------------------------------- #
    @property
    def busy_until(self) -> Mapping[str, float]:
        """Predicted prefill queue tail per instance, read-only.

        A mapping rather than the dict, so the only way to move a tail is to
        report something that moved it.
        """
        return MappingProxyType(self._busy_until)

    def occupancy(self, inst: str) -> int:
        """Requests currently decoding or queued on ``inst``."""
        return len(self._decode_finishes[inst])

    def predict_occupancy(self, inst: str, at_t: float) -> int:
        """How many of those are estimated to still be decoding at ``at_t``."""
        return sum(1 for f in self._decode_finishes[inst] if f > at_t)
