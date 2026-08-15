"""The dedup control plane: which peer serves a reader, and when.

One plane, one chain. :class:`~dedup_sim.control.routing.Dedup` is a
:class:`proposed.plane.ControlPlane` with one member -- the source question a reader
asks -- and it answers it from the :class:`proposed.selector.FirstMatch` chain it
builds over :mod:`dedup_sim.control._selector`, where the ranking lives, plus
:mod:`dedup_sim.control._answer`, where the route it records and the waiting do. What
makes that answer come true is not asked of it: a reader commits one
:class:`proposed.dispatch.Stored`, and this plane's own state
(:mod:`dedup_sim.control._sensor`) is what folds it.

Formed against a read-only :class:`~proposed.view.View` carrying that one sensor: it
holds no client, no volume and no deployment, and it never executes anything -- the
read-through write lives in :mod:`dedup_sim.data`.
"""
