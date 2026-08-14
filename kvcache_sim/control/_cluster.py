"""Control's model of the cluster: :class:`KVClusterModel`, and the facts it folds.

Nothing here executes. What control knows about the running cluster is a *model*
corrected by what the hosts report, never a live read:

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
  wholesale by :class:`DecodeState`;
* what was decided and not yet carried out -- prefills promised, pulls priced
  against a peer -- is :mod:`kvcache_sim.control._pending`, written by
  :class:`Committed` and read back as it expires.

Folder-private: the port this answers (:class:`proposed.ClusterModel`) is the
surface, and a host reaches it through the seam in front of it
(:class:`realsim.seams.cluster_model_handle.LocalClusterModelHandle`). The control
plane reads and writes it here, in this process, through the one sensor it senses
everything through (:class:`kvcache_sim.control._view.KVView`).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING,
)

from proposed import ClusterModel

from ._pending import Reservation, Reservations, RoutedPulls

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
    """A decision the scheduler accepted: its instances are now spoken for.

    The one fact control tells itself. Sent at commit rather than while pricing,
    so a candidate that lost -- or a decision a gate refused -- leaves nothing
    behind. Carries ``output_tokens`` because a decode reservation is as long as the
    generation, and the request itself is not part of the answer.
    """

    response: "Response"
    output_tokens: int


class KVClusterModel(ClusterModel):
    """Every instance's predicted load, and what has been promised against it.

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
        lookahead: predict decode occupancy forward. What makes a promised
            prefill worth remembering: without it nothing reads a reservation,
            because the observed decode state is the whole of the picture.
    """

    def __init__(self, ids: Sequence[str], *, lookahead: bool = False) -> None:
        self._busy_until: Dict[str, float] = {i: 0.0 for i in ids}
        # instance -> one estimated finish time per request decoding or queued
        # there. Empty until the data plane reports.
        self._decode_finishes: Dict[str, List[float]] = {i: [] for i in ids}
        self._lookahead = lookahead
        # Decided but not yet carried out: prefills promised, and pulls priced
        # against a peer. Both self-expire (:mod:`kvcache_sim.control._pending`).
        self._reserved = Reservations()
        self._routed = RoutedPulls()
        # fact type -> the bound method that folds it.
        self._folds: Dict[type, Callable[[Any], None]] = {
            ComputeBusy: self._compute_busy,
            DecodeState: self._decode_state,
            PrefillFinished: self._prefill_finished,
            Committed: self._committed,
        }

    # -- proposed.ClusterModel: the one way in ----------------------------- #
    async def notify(self, fact: Any) -> None:
        """:class:`~proposed.deployment.ClusterModel` -- fold ``fact`` into state.

        The endpoint a host reports over, and the whole of what crosses the seam.
        """
        self._notify_impl(fact)

    def _notify_impl(self, fact: Any) -> None:
        """Fold ``fact`` in, here and now: what co-located control code calls.

        **Not a coroutine, and that is load-bearing**, the same way
        :meth:`proposed.deployment.Controller.locate_raw` is not one: the scheduler
        writes :class:`Committed` in the middle of a routing decision, and a plain
        call cannot let a second decision interleave with the first -- whereas an
        ``await`` would rest that atomicity on the coroutine happening not to
        suspend.

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
        """Hold the prefill instance for an accepted decision, and note its pull."""
        plan = fact.response.plan
        # The peer this pull was priced against, for when the fetch asks
        # (:meth:`claim`).
        if plan.reuse_source is not None and plan.pull_keys:
            self._routed.route(
                fact.response.prefill, plan.pull_keys, plan.reuse_source
            )
        self._busy_until[fact.response.prefill] = plan.done_time
        if self._lookahead:
            self._reserved.reserve(
                plan.done_time, fact.response.decode, fact.output_tokens
            )

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

    def pending(self, now: float) -> Sequence[Reservation]:
        """Prefills promised and not landed as of ``now``, oldest first.

        They stand in for requests the observed decode state cannot show yet
        (:class:`~kvcache_sim.control._pending.Reservations`).
        """
        return self._reserved.pending(now)

    def claim(self, requester: str, keys: Sequence[str]) -> Optional[str]:
        """The peer ``requester``'s pull of ``keys`` was priced against, consumed.

        A read that *writes*: taking a routed pull expires it
        (:meth:`~kvcache_sim.control._pending.RoutedPulls.claim`), which is what
        stops two fetches claiming the same record. And one call, which it has to
        stay: split into a read and a following write, both could read before either
        claimed, and the second would pull from a peer nothing priced -- an
        unplanned transfer, with the predicted cost drifting from the actual one and
        nothing failing.
        """
        return self._routed.claim(requester, keys)
