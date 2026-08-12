"""The KV-cache data plane: everything that advances the clock or moves bytes.

* :mod:`~kvcache_sim.data.serving` -- one serving host: the router role every host
  plays, then the per-request loop for the ones it serves -- queue wait, the real
  prefix pull, the prefill charge, the real publish, the decode handoff, and the
  outcome rows;
* :mod:`~kvcache_sim.data._decode` -- one host's batched decode engine. It sleeps
  and emits tokens, so it is data. Underscored because nothing outside this package
  drives it: the host constructs it, owns it, and is the only thing it reports to --
  control learns the decode load as a value the host forwards, not by holding this
  object;
* :mod:`~kvcache_sim.data.store` -- what a KV block is stored as, and the verbs
  that follow from it (publish / reuse / fetch) as real ``put_batch`` / ``touch``
  / ``get_batch`` calls. Constructing one is where the block-size premise every
  fetch is priced against gets checked.

The test for what belongs here: does it advance the clock or move bytes? A
directory *read* does neither, so it is a control-plane view, not a verb here. So
is *delivering* a request to the host it landed on: which host that is belongs to a
load balancer, which a deployment has and this package is not
(:class:`kvcache_sim.workload._serving._LoadBalancer`).
"""
