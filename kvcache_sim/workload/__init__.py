"""What is simulated: the domain model, the request stream, and the scenarios.

- ``request.py`` -- the domain model: an inference ``Request`` plus prefix-hash
  block addressing (a block key is a plain directory key).
- ``_generator.py`` -- the seeded synthetic request stream (shared system prompt +
  per-conversation context + unique query suffix; Zipf popularity, Poisson
  arrivals).
- ``_serving.py`` -- ``KVWorkload`` (the request stream, a
  ``realsim.workload.Workload``) and ``serving_plane``, the factory that wires the
  store, view, scheduler and serving plane onto an assembled stack. Also
  ``sim_block_carrier``: what one KV block is stored as under simulation, which
  belongs to the run rather than to ``data/`` -- a real deployment stores the KV
  tensors.
- ``scenarios.py`` -- the comparisons, each a list of ``realsim.run.Run`` values
  over one request stream. It builds no clock, mesh or plane.
"""
