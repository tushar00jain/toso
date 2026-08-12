"""What is simulated: the domain model, the request stream, and the scenarios.

- ``_generator.py`` -- the seeded synthetic request stream (shared system prompt +
  per-conversation context + unique query suffix; Zipf popularity, Poisson
  arrivals), including each request's prompt tensor and the prefix-hash chain it
  addresses that prompt with. The chain is generated *beside* the prompt rather
  than hashed out of it, because a zero-storage prompt has no content to hash.
- ``_accelerator.py`` -- ``SimulatedAccelerator``: what a forward pass costs
  (a roofline) and how it is made to take that long (a sleep), plus what it
  *produces* -- one zero-storage ``device="meta"`` KV tensor per block, at the
  byte count the scheduler prices a fetch against, and one token per thing that
  generated one (the prefill's first, a decode step's per batch member). All of
  it is what a deployment answers by running the model, so it belongs to the run
  and not to ``data/``. ``BLOCK_TOKENS`` is here too -- how much of a prompt one
  KV block covers is the engine's cache-page size -- and so is ``TOKEN_DTYPE``,
  the one definition of what a token id is, which both ends of a request use.
- ``_serving.py`` -- ``KVWorkload`` (the request stream, a
  ``realsim.run.Workload``) and ``serving_plane``, the factory that wires the
  store, view, scheduler and serving plane onto an assembled stack.
- ``scenarios.py`` -- the comparisons, each a list of ``realsim.run.Run`` values
  over one request stream. It builds no clock, mesh or plane.
"""
