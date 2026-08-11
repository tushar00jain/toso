"""What is simulated: the domain model, the request stream, and the scenarios.

- ``request.py`` -- the domain model: an inference ``Request`` plus prefix-hash
  block addressing (a block key is a plain directory key).
- ``_generator.py`` -- the seeded synthetic request stream (shared system prompt +
  per-conversation context + unique query suffix; Zipf popularity, Poisson
  arrivals).
- ``serving.py`` -- ``KVWorkload``, which builds both planes onto an assembled
  stack, and ``sim_block_carrier``: what one KV block is stored as under
  simulation. That choice belongs to the run, not to ``data/`` -- a real
  deployment stores the KV tensors.
- ``scenarios.py`` -- the scenario builders and the deterministic run harness that
  wires a topology, a workload and a scheduler onto the shared engine.
"""
