"""What a dedup decision senses: :class:`FanoutView`, and :class:`DedupView` over it.

Dedup holds one sensor, and its decisions read it two ways: as the tree they extend
(:class:`FanoutView`) and as the load they spread over
(:class:`~proposed.view.LoadView`). :class:`DedupView` is the pair, which is what
:meth:`dedup_sim.control.routing.Dedup.attach` derives; each link is attached to the
subset its own header declares. A keyword no view claims reaches
:class:`~proposed.view.View`, which takes none, and raises there.
"""

from __future__ import annotations

from proposed import LoadView, Sensed, SensorView

__all__ = ["DedupView", "FanoutView"]


class FanoutView(SensorView):
    """The tree this plane has planned and the puts it is owed: :attr:`fanout`.

    Observed state as much as the directory is -- this plane's own record of its own
    decisions -- so whatever routes against it senses it here instead of being handed
    the sensor. A landed put reaches the same sensor as a reducer's fold, not through
    this view: what a view offers is the read.
    """

    fanout = Sensed("fan-out")


class DedupView(FanoutView, LoadView):
    """Both reads of dedup's one sensor: the tree, and the load on it.

    One sensor composed under two names, because the two reads are the same record:
    who is routed to whom is the tree a link extends and the load a
    :class:`~proposed.selector.Balance` appends
    (:meth:`~dedup_sim.control._sensor.FanoutSensor.named`). A second sensor would be a
    second copy of it to keep in step.
    """
