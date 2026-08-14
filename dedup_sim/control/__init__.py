"""The dedup control plane: which peer serves a reader, and when.

One module, one plane. :class:`~dedup_sim.control.routing.Dedup` is a
:class:`proposed.plane.ControlPlane` with two members -- the source question a reader
asks and the report of the put that answers it -- formed against a read-only
:class:`~proposed.view.View`. It holds no client, no volume and no deployment, and it
never executes anything: the read-through write that makes its answer come true lives
in :mod:`dedup_sim.data`.
"""
