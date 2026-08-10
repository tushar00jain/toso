"""What is simulated: the dedup read burst.

``burst.py`` runs ``realsim``'s ordinary put/get fixture twice -- once unrouted
(the ``m x`` baseline) and once with the dedup policy and read-through plane
installed (1x) -- so the two are compared on byte-for-byte the same topology,
payload and cost model.

There is no request *generator* here (unlike ``kvcache_sim.workload``): the
workload is one fixed synchronized burst, parameterized only by reader count.
"""
