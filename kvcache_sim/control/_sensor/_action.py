"""What this capability dispatches: the five things that happen to a cluster.

Each is a :class:`proposed.dispatch.Action`, folded by every sensor here that folds
its type and committed once (:class:`proposed.dispatch.Dispatcher`). Three come from a
host over the seam in front of that dispatcher and two the scheduler dispatches to
itself, one per question it answers; nothing here says which sensor folds what, because
an action does not know who folds it.

Frozen values, so they cross a process boundary unchanged and cannot be edited after
they are handed over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from proposed.dispatch import Action

from .._answer import Response

__all__ = [
    "Committed", "ComputeBusy", "DecodeState", "FetchAnswered", "PrefillFinished",
]


@dataclass(frozen=True)
class PrefillFinished(Action):
    """*Prefill really finished at this clock -- what is the queue tail now?*

    The only thing that tells this control plane its picture of an instance's
    prefill queue was wrong: ``now`` is measured independently (the host's
    accelerator serialises its own passes).

    Reported over a call the host waits for, and not for the reply, which carries
    nothing: the decode admission it asks next must be decided against a sensor
    that has already folded this completion.
    """

    inst: str
    now: float


@dataclass(frozen=True)
class ComputeBusy(Action):
    """A decode step occupied a **coupled** instance's compute until ``until``."""

    inst: str
    until: float


@dataclass(frozen=True)
class DecodeState(Action):
    """``inst``'s live decode batch, as one estimated finish time per request.

    Its length is the occupancy and its values answer "still decoding at ``t``?".
    Reported whenever the batch changes.
    """

    inst: str
    finishes: Tuple[float, ...]


@dataclass(frozen=True)
class FetchAnswered(Action):
    """``requester`` has been told which peers serve its fetch of ``keys``.

    Dispatched as the answer is given, whatever answered it: a fetch a decision priced a
    pull for spends the memo that answered it
    (:class:`~kvcache_sim.control._sensor.RoutedPullSensor`), and one nothing priced
    spends nothing, so the plane never has to know which link won.
    """

    requester: str
    keys: Tuple[str, ...]


@dataclass(frozen=True)
class Committed(Action):
    """A decision the scheduler accepted: every sensor it moves, in one action.

    The only action here with more than one fold. Dispatched at commit rather than while
    pricing, so a candidate that lost -- or a decision an SLO refused -- leaves nothing
    behind.

    ``output_tokens`` is the request's, and is here because no sensor can read a
    request: the reservation this holds stands in for a decode that has not started,
    so how long it will run is part of what was decided.
    """

    response: Response
    output_tokens: int
