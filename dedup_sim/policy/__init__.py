"""The algorithm under test: 1x-fabric dedup read routing.

``routing.py`` holds :class:`~dedup_sim.policy.routing.DedupPolicy`, a real
:class:`realsim.coordinator.model.ReadPolicy`, plus the per-reader directory view
that steers a real ``LocalClient`` to the policy-chosen source.
"""
