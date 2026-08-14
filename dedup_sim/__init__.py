"""Dedup read-routing on the real TorchStore directory (a realsim capability).

``dedup_sim`` runs the dedup algorithm on the **real** ``Controller`` directory
and **real** client/transport (via ``realsim``): a synchronized read burst is
routed so each unique byte crosses the fabric exactly once (1x), versus ``m x``
for the unrouted baseline.

Laid out by plane, like ``kvcache_sim``:

* :mod:`dedup_sim.control` -- one :class:`proposed.plane.ControlPlane` over a chain of
  real :class:`proposed.selector.KeySelector` links: the plane answers the two
  questions a reader asks, the chain makes the routing decision behind them. It holds
  no client, no volume and no mesh;
* :mod:`dedup_sim.data` -- the read-through put that turns a finished reader into
  a real directory source, a :class:`proposed.plane.DataPlane` overriding one
  method;
* :mod:`dedup_sim.workload` -- the configurations to compare, as
  :class:`realsim.run.Run` values. Every one shares ``putget_sim``'s ordinary
  put/get fixture; the baseline installs nothing and each routed run adds the two
  planes, so nothing else changes between them;
* :mod:`dedup_sim.report` -- the dedup-vs-baseline fabric summary.

There is no harness, no runtime package and no cost layer of its own: ``realsim``'s
``Runner`` releases the readers and its transport seam charges the cost model.
See ``dedup_sim/README.md`` for the folder-by-folder comparison.
"""
