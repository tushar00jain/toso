"""The dedup control plane: which peer serves a reader, and when.

One plane, one ranking. :class:`~dedup_sim.control.routing.Dedup` is a
:class:`proposed.plane.ControlPlane` with two members -- the source question a reader
asks and the report of the put that answers it -- and it answers both from the
:class:`~dedup_sim.control._source.PeerKeySelector` it holds, where the routing
decision and the waiting live. Formed against a read-only
:class:`~proposed.view.View`: it holds no client, no volume and no deployment, and it
never executes anything -- the read-through write that makes its answer come true
lives in :mod:`dedup_sim.data`.
"""
