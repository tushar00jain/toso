"""What is simulated: the dedup read-burst scenarios.

``scenarios.py`` builds and runs the synchronized read burst under the dedup
policy, and re-exports realsim's own burst as the naive ``m x`` baseline so both
are compared on byte-for-byte the same topology, payload and cost model.

There is no request *generator* here (unlike ``kvcache_sim.workload``): the
workload is one fixed synchronized burst, parameterized only by reader count.
"""
