"""This host's prefill compute: :class:`PrefillEngine`.

The sibling of :class:`~kvcache_sim.data._decode.DecodeEngine`, and the reason it
exists is that it did not. Decode had an engine -- a batch, a step loop, a
timeline -- while prefill was three lines inlined in the serving loop: sleep the
queue wait, sleep the forward pass, and a cost function to re-price it when a
planned reuse turned out to be gone. Same physical resource, same kind of work,
two very different shapes, and the asymmetry was the reason a host could not
simply be asked which engines it runs.

So prefill is an engine too, and a host runs one, the other, or both. Both are
handed the host's :class:`~kvcache_sim.data._compute.Accelerator`, which is what
makes "coupled" a fact about a host rather than a flag beside it: two engines on
one accelerator collide, and there is nothing to configure.

The queue is real, and that is why this engine has one member fewer
-------------------------------------------------------------------
This class used to have a ``wait_turn(queue_wait)``: the serving host handed it the
number the *control plane predicted* the request would wait at this host, and it
slept exactly that. The prediction was therefore unfalsifiable. A data plane that
sleeps a forecast waits precisely as long as the forecast said, so a scheduler that
mispredicts its own queue is never contradicted by the run, and every column
containing the wait -- TTFT, end-to-end latency, the next request's predicted queue
-- inherits the prediction rather than testing it. That was the weakest joint in the
whole model and the one thing here that could not be wrong.

There is no member for it now, and no flag that brings it back. The wait is
**emergent**: :meth:`PrefillEngine.run` submits a forward pass to this host's
accelerator, the accelerator runs it when it is free -- behind whatever prefill or
decode step already has the device -- and how long that took is something the
caller measures. ``plan.queue_wait`` still exists and control still computes it;
it is now a *prediction the run can disagree with*, which is the point.

Two smaller things fell out with it, and both are simplifications rather than
losses. The wait moved to the far side of the request's KV fetch, where a forward
pass that cannot start before its inputs arrive actually has it (control's
arithmetic puts the wait first, which also means control charges a fabric transfer
to a device that is idle during it -- a divergence the run can now show). And
``reserve``, which existed so that a decode step would not be scheduled through a
prefill the accelerator did not know about, is gone: it knows about it.

What is deliberately still missing
----------------------------------
**Batching.** A real prefill engine runs several prompts in one forward pass, and
chunks a long one across several. Here a submission is one prompt and occupies the
device alone for what a roofline says it costs, so this queue is a single server
with FIFO service, not a scheduler: it can show that a wait was mispredicted, but
it cannot show a scheduler recovering by co-batching two short prompts. That needs
:meth:`~kvcache_sim.data._compute.Accelerator.prefill_cost` to price a *set* of
prompts, which is a change to the cost model and not to this shape.

**Preemption.** A queued prefill cannot be cancelled, reordered by priority, or
evicted for a request with a tighter SLO once it is behind one that is not. The
service order is fixed at submission, which is the honest version of what this
models and the one a deployment's continuous batcher is a departure from.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

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

    @property
    def block_tokens(self) -> int:
        """Tokens per KV block, as this host's accelerator lays them out.

        Forwarded rather than stored, so there is one answer on the host and it is
        the one the KV was actually cut into (see
        :attr:`~kvcache_sim.data._compute.Accelerator.block_tokens`).
        """
        return self.compute.block_tokens

    def cost(self, uncached_tokens: int) -> float:
        """What prefilling ``uncached_tokens`` costs on this host.

        Asked of the accelerator rather than computed here, because what a forward
        pass costs is a fact about the machine running it. Needed for the one case
        control cannot have priced: a plan whose remote prefix was evicted between
        routing and fetching, which has to be recomputed and re-reported.
        """
        return self.compute.prefill_cost(uncached_tokens)

    async def run(
        self,
        prompt: torch.Tensor,
        cached: Sequence[torch.Tensor] = (),
        *,
        tag: str = "",
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run the forward pass on this host's accelerator; answer with KV and token.

        Named in tokens rather than seconds, because how long that takes is the
        accelerator's answer and not this engine's -- and because a deployment
        implementing the port runs a model, which needs the work and not a duration.
        ``prompt`` is the uncached suffix of the request's prompt, as ids: the count
        it used to be was the last place a duration could have been substituted for
        the work, and it was also the reason a deployment's engine could not have
        been dropped in behind this call.

        ``cached`` is the prefix this host pulled out of the store, and the first
        half of the answer is every KV block this host now holds and did not before
        (that prefix, then the computed suffix). Passed straight through: which
        blocks a prefill ends up holding is the accelerator's account of its own
        output, and an engine in between that reassembled the list would be a second
        place for the order to be wrong.

        The second half is the request's **first token** -- the one TTFT is the time
        to. It travels the same way and for the same reason: the pass produced it,
        so the pass answers with it, and this engine adds nothing to either.

        This is where the pass is **submitted**, and therefore where it queues:
        after the caller's fetch, because a forward pass cannot start before its
        inputs are here, and with the tokens that fetch actually left it with -- a
        planned reuse that turned out to be evicted makes this a bigger pass than
        the plan priced, and the queue slot has to be the real one. ``tag`` names
        the submission so two passes handed in at the same instant queue in a fixed
        order; the caller passes the request's id.
        """
        return await self.compute.prefill(prompt, cached, tag=tag)
