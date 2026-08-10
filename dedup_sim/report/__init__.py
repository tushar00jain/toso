"""Outcome metrics and console rendering.

``summary.py`` renders the dedup-vs-baseline fabric comparison plus the
source->dest tree. The measurements themselves are a shared
:class:`sim_common.report.Ledger` -- unlike ``kvcache_sim``, this capability
defines no metrics of its own.
"""
