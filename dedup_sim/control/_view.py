"""What one dedup decision senses: :class:`FanoutView`.

Dedup senses one thing beside the directory, so there is no view to assemble out of
several: ``view.derived(FanoutView, fanout=...)``
(:meth:`~proposed.view.View.derived`) is the whole of it. The keyword is claimed here
and nowhere else -- one no view claims reaches :class:`~proposed.view.View`, which
takes none, and raises there.
"""

from __future__ import annotations

from proposed import Sensed, SensorView

__all__ = ["FanoutView"]


class FanoutView(SensorView):
    """The tree this plane has planned and the puts it is owed: :attr:`fanout`.

    Observed state as much as the directory is -- this plane's own record of its own
    decisions -- so whatever routes against it senses it here instead of being handed
    the sensor. A landed put reaches the same sensor as a reducer's fold, not through
    this view: what a view offers is the read.
    """

    fanout = Sensed("fan-out")
