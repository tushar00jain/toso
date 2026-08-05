"""Outcome metrics and console rendering.

``metrics.py`` holds the per-request outcome row (``RequestResult``), the
aggregate ``Metrics`` (hit rate, TTFT/TBT percentiles, fabric bytes, rejections)
and the scenario summary renderers. Metrics are on the DES *outcome*, never on
wall-clock. The generic event recorder is ``sim_common.trace.Trace``.
"""
