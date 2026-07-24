"""Dedup read-routing on the real TorchStore directory (a realsim capability).

``dedup_sim`` runs the dedup algorithm on the **real** ``Controller`` directory
and **real** client/transport (via ``realsim``): a synchronized read burst is
routed so each unique byte crosses the fabric exactly once (1x), versus ``m x``
for the naive baseline. The routing is a real
:class:`realsim.coordinator.model.ReadPolicy` (:class:`dedup_sim.policy.DedupPolicy`),
so real types run end to end.
"""
