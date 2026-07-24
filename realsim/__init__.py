"""realsim -- a real-code cooperative DES over the real TorchStore.

``realsim`` drives the **real** TorchStore client planning core and controller
directory logic off-actor -- on a plain asyncio loop or the deterministic
virtual-clock engine (``sim_common.async_engine``) -- with an in-memory transport.
It therefore depends on the real ``torchstore`` / ``torch`` / ``monarch`` install
in the venv; that is the point. ``dedup_sim`` and ``kvcache_sim`` build on it.

See ``docs/realsim_design.md`` for the full design, including exactly how each
real object is driven off-actor.
"""
