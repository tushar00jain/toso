"""This host's prefill compute: :class:`PrefillEngine`.

The sibling of :class:`~kvcache_sim.data._decode.DecodeEngine`, and the reason it
exists is that it did not. Decode had an engine -- a batch, a step loop, a
timeline -- while prefill was three lines inlined in the serving loop: sleep the
queue wait, sleep the forward pass, and a cost function to re-price it when a
planned reuse turned out to be gone. Same physical resource, same kind of work,
two very different shapes, and the asymmetry was the reason a host could not
simply be asked which engines it runs.

So prefill is an engine too, and a host runs one, the other, or both. Both are
handed the host's :class:`~kvcache_sim.data._compute.ComputeTimeline`, which is
what makes "coupled" a fact about a host rather than a flag beside it: two engines
on one timeline collide, and there is nothing to configure.

What is deliberately still missing
----------------------------------
A queue. :meth:`PrefillEngine.wait_turn` sleeps the wait the *control plane
predicted*, which is what the serving loop did before it and what keeps this a
move rather than a change. It is also the weakest thing in the model: a predicted
wait that the data plane then sleeps is self-fulfilling, so a scheduler that
mispredicts its own queue is never contradicted by the run. Making the queue real
-- submit work, run when the accelerator is free, let the wait be emergent -- is
the fidelity step this shape is here to allow, and it moves measured numbers, so
it is not taken here.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ._compute import Accelerator

__all__ = ["PrefillEngine"]


class PrefillEngine:
    """Prefill for one host: what it costs, and occupying the GPU for that long.

    Args:
        compute: the host's :class:`~kvcache_sim.data._compute.Accelerator`.
            The *same object* as this host's
            :class:`~kvcache_sim.data._decode.DecodeEngine` was handed when the two
            are modelled as contending -- which is what coupling is.
    """

    def __init__(self, compute: Accelerator) -> None:
        self.compute = compute

    def cost(self, uncached_tokens: int) -> float:
        """What prefilling ``uncached_tokens`` costs on this host.

        Asked of the accelerator rather than computed here, because what a forward
        pass costs is a fact about the machine running it. Needed for the one case
        control cannot have priced: a plan whose remote prefix was evicted between
        routing and fetching, which has to be recomputed and re-reported.
        """
        return self.compute.prefill_cost(uncached_tokens)

    async def wait_turn(self, queue_wait: float) -> None:
        """Wait out the queue this request was routed behind.

        ``queue_wait`` is control's prediction rather than this engine's own
        backlog -- see the module docstring for why that is the model's weakest
        joint and why it is still what happens here.
        """
        if queue_wait > 0:
            await asyncio.sleep(queue_wait)

    async def run(self, uncached_tokens: int) -> None:
        """Run the forward pass for ``uncached_tokens`` on this host's accelerator.

        Named in tokens rather than seconds, because how long that takes is the
        accelerator's answer and not this engine's -- and because a deployment
        implementing the port runs a model, which needs the work and not a duration.
        """
        await self.compute.prefill(uncached_tokens)

    def reserve(self, until: float) -> None:
        """Hold the accelerator until ``until`` on this request's behalf.

        Called when the host also decodes: the prefill about to run is on the
        timeline decode steps use, so a step must not be scheduled through it. A
        host with no decode engine shares its timeline with nothing and the call is
        harmless.
        """
        self.compute.reserve(until)
