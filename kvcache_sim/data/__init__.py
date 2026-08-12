"""The KV-cache data plane: everything that advances the clock or moves bytes.

* :mod:`~kvcache_sim.data.serving` -- one serving host, as three things a client
  can ask it: where a request belongs (the router role every host plays), the
  prefill (queue wait, the real prefix pull, the prefill charge, the real publish)
  and the decode (the real KV handoff fetch, the batch, the inter-token gaps). Each
  of the first two answers with an *address* rather than calling the host it names,
  so nothing here holds a reference to another host and no measurement row travels
  between them;
* :mod:`~kvcache_sim.data._decode` -- one host's batched decode engine. It sleeps
  and emits tokens, so it is data. Underscored because nothing outside this package
  drives it: the host constructs it, owns it, and is the only thing it reports to --
  control learns the decode load as a value the host forwards, not by holding this
  object;
* :mod:`~kvcache_sim.data.store` -- what a KV block is stored as, and the verbs
  that follow from it (publish / reuse / fetch) as real ``put_batch`` / ``touch``
  / ``get_batch`` calls. Constructing one is where the block-size premise every
  fetch is priced against gets checked. It is also the *only* path between two
  serving hosts: a prefill host publishes a request's KV and its decode host
  fetches it back, which is what makes the handoff a transfer with a price rather
  than an argument to a method call.

The test for what belongs here: does it advance the clock or move bytes? A
directory *read* does neither, so it is a control-plane view, not a verb here. So
is *submitting* a request and following the redirects it comes back with: which
host it lands on belongs to a load balancer and walking the chain belongs to a
client, both of which a deployment has and this package is not
(:class:`kvcache_sim.workload._serving._Client`).
"""
