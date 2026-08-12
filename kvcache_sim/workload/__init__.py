"""What is simulated: the domain model, the request stream, and the scenarios.

- ``_generator.py`` -- the seeded synthetic stream of **multi-turn conversations**.
  Turn 1 is a shared system prompt + per-tenant context + a user message; turn N+1
  is turn N's whole sequence + the keys turn N's **output** left behind + a new
  message, so the reusable prefix grows and contains generated tokens. A Zipf law
  over tenants decides how many turns each contributes (cut into dialogues of at
  most ``max_turns``), dialogue starts are Poisson and the pause between turns is
  exponential. Each turn carries its prompt tensor and the prefix-hash chain that
  addresses it; the chain is generated *beside* the prompt rather than hashed out
  of it, because a zero-storage prompt has no content to hash, and the generated
  blocks in the middle of it come from the previous turn's own
  ``Request.continuation_keys`` rather than from a second spelling of the same
  rule.
- ``_accelerator.py`` -- ``SimulatedAccelerator``: what a forward pass costs
  (a roofline) and how it is made to take that long (a sleep), plus what it
  *produces* -- one zero-storage ``device="meta"`` KV tensor per block, at the
  byte count the scheduler prices a fetch against, and one token per thing that
  generated one (the prefill's first, a decode step's per batch member). All of
  it is what a deployment answers by running the model, so it belongs to the run
  and not to ``data/``. ``BLOCK_TOKENS`` is here too -- how much of a prompt one
  KV block covers is the engine's cache-page size -- and so is ``TOKEN_DTYPE``,
  the one definition of what a token id is, which both ends of a request use.
- ``_serving.py`` -- ``KVWorkload`` (one work item per conversation, a
  ``realsim.run.Workload``) and ``serving_plane``, the factory that wires the
  store, view, scheduler and serving plane onto an assembled stack. The client it
  builds stands in for the user as well as the SDK: it walks a dialogue's turns
  one at a time, because turn N+1 contains turn N's answer.
- ``scenarios.py`` -- the comparisons, each a list of ``realsim.run.Run`` values
  over one request stream. It builds no clock, mesh or plane.
"""
