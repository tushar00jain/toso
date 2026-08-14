"""The dedup control plane: which peer serves a reader, and when.

One plane, one chain. :class:`~dedup_sim.control.routing.Dedup` is a
:class:`proposed.plane.ControlPlane` with two members -- the source question a reader
asks and the report of the put that answers it -- and it answers both from the
:class:`proposed.selector.FirstMatch` chain it builds over
:mod:`dedup_sim.control._selector`, where the routing decision and the waiting live.
Formed against a read-only :class:`~proposed.view.View` carrying the one sensor those
links read (:mod:`dedup_sim.control._sensor`): it holds no client, no volume and no
deployment, and it never executes anything -- the read-through write that makes its
answer come true lives in :mod:`dedup_sim.data`.
"""
