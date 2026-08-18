"""The dedup control plane: which peer serves a reader, and when.

:class:`~dedup_sim.control.routing.Dedup` is a control plane with one member, the
source question a reader asks, answered from a chain over
:mod:`dedup_sim.control._selector` (the ranking). The plane records the route and
withholds the answer until its head is usable.

Nothing here makes that answer come true: a reader commits one
:class:`proposed.dispatch.Stored` and this plane's own state
(:mod:`dedup_sim.control._sensor`) folds it. The plane holds no client, no volume and
no deployment, and executes nothing; the read-through write is :mod:`dedup_sim.data`.
"""
