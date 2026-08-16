"""What a dedup decision senses: :class:`FanoutView`, and :class:`DedupView` over it.

Dedup holds one sensor and reads it two ways: as the tree a decision extends
(:class:`FanoutView`) and as the load it spreads over
(:class:`~proposed.view.LoadView`).
"""

from __future__ import annotations

from proposed import LoadView, Sensed, SensorView

__all__ = ["DedupView", "FanoutView"]


class FanoutView(SensorView):
    """The tree this plane has planned and the puts it is owed: :attr:`fanout`.

    Observed state, as the directory is: this plane's own record of its own decisions.
    """

    fanout = Sensed("fan-out")


class DedupView(FanoutView, LoadView):
    """Both reads of dedup's one sensor: the tree, and the load on it.

    Who is routed to whom is both facts at once, so there is one sensor under two
    names; a second would be a second copy to keep in step.
    """
