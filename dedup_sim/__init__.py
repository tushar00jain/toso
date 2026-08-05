"""Dedup read-routing on the real TorchStore directory (a realsim capability).

``dedup_sim`` runs the dedup algorithm on the **real** ``Controller`` directory
and **real** client/transport (via ``realsim``): a synchronized read burst is
routed so each unique byte crosses the fabric exactly once (1x), versus ``m x``
for the naive baseline. The routing is a real
:class:`realsim.coordinator.model.ReadPolicy` (:class:`dedup_sim.policy.routing.DedupPolicy`),
so real types run end to end.

Laid out in the same role folders as ``kvcache_sim`` -- :mod:`dedup_sim.policy`
(the routing algorithm), :mod:`dedup_sim.workload` (the burst scenarios) and
:mod:`dedup_sim.report` (the fabric summary). It needs no ``runtime`` package and
no cost layer of its own: realsim's ``ReadCoordinator`` already drives a burst and
charges the cost model, so the ``ReadPolicy`` seam is the only hook required. See
``dedup_sim/README.md`` for the folder-by-folder comparison.
"""
