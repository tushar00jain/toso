"""The KV-cache data plane: everything that advances the clock or moves bytes.

* :mod:`~kvcache_sim.data.serving` -- one serving host, as three things a client
  can ask it: where a request belongs (the router role every host plays), the
  prefill (queue wait, the real prefix pull, the prefill charge, the real publish
  -> the first token and an address) and the decode (the real KV handoff fetch, the
  batch, the inter-token gaps -> the remaining tokens). Each of the first two
  answers with an *address* rather than calling the host it names, so nothing here
  holds a reference to another host and no measurement row travels between them;
* :mod:`~kvcache_sim.data._decode` -- one host's batched decode engine: it sleeps and
  emits real tokens, accumulated per batch member and handed back when the request's
  last one lands. Underscored because the host constructs it, owns it and is the only
  thing it reports to, so control learns the decode load as a value the host forwards;
* :mod:`~kvcache_sim.data._store` -- publish / reuse / fetch, as real ``put_batch`` /
  ``touch`` / ``get_batch`` calls over whatever KV it is handed. The *only* path
  between two serving hosts: a prefill host publishes a request's KV and its decode
  host fetches it back, so the handoff is a transfer with a price rather than an
  argument to a method call;
* :mod:`~kvcache_sim.data._compute` and :mod:`~kvcache_sim.data._prefill` -- the
  accelerator an engine runs on, and the prefill engine that runs on it. What a block
  *is*, how big one is and what a token is are answered by whatever implements that
  port: a simulated run's meta tensors, a deployment's attention output and sampler.

The test for what belongs here: does it advance the clock or move bytes? A directory
*read* does neither, so it is a control-plane sensor read. Nor does *submitting* a request and
following the redirects it comes back with: which host it lands on belongs to a load
balancer and walking the chain belongs to a client
(:class:`kvcache_sim.workload._serving._Client`).
"""
