"""Outcome metrics and console rendering.

``summary.py`` renders the dedup-vs-naive fabric comparison plus the
source->dest tree. The metrics themselves are realsim's
:class:`~realsim.coordinator.model.BurstMetrics` -- unlike ``kvcache_sim``, this
capability defines no metrics of its own.
"""
