"""DATA PLANE: the things that actually move bytes and burn compute.

Everything here executes a decision the control plane
(:mod:`kvcache_sim.policy`) already made, on ``realsim``'s real TorchStore objects
and the shared virtual clock:

- ``cluster.py`` -- the four KV directory verbs (``prefix_lengths`` / ``publish``
  / ``fetch`` / ``evict``) over a :class:`realsim.mesh.Mesh`, so block presence is
  the real ``Controller`` directory and a prefix pull is a real ``client.get``.
  (``publish`` straddles both planes, because a real ``put`` writes bytes *and*
  registers metadata -- that is TorchStore's shape, not a layering slip.)
- ``driver.py`` -- the serving engine's request loop. It holds no decisions and no
  data; it turns a ``Plan`` into clock advances and real operations (sleep to
  arrival, wait out the queue, pull the prefix, charge prefill, report done) and
  records the outcome.
- ``decode.py`` -- the batched decode engine: it *runs* decode, charging
  ``decode_step_time(batch)`` per step on the virtual clock. It lives here rather
  than beside the scheduler because charging compute is doing work, not deciding.

The compute-time functions are shared with the control plane (which predicts with
the same numbers) and live at :mod:`kvcache_sim.utils`.
"""
