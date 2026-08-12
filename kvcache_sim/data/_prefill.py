"""This host's prefill compute: :class:`PrefillEngine`.

The sibling of :class:`~kvcache_sim.data._decode.DecodeEngine`. A host runs one,
the other, or both, and both are handed the host's
:class:`~kvcache_sim.data._compute.Accelerator` -- which is what makes "coupled" a
fact about a host rather than a flag beside it: two engines on one accelerator
collide, and there is nothing to configure.

The queue is emergent
---------------------
There is no member here that sleeps a predicted wait. :meth:`PrefillEngine.run`
submits a forward pass to this host's accelerator, the accelerator runs it when it
is free -- behind whatever prefill or decode step already has the device -- and how
long that took is something the caller measures. ``plan.queue_wait`` is therefore a
*prediction the run can disagree with*, rather than an input the run replays: a
data plane that sleeps a forecast waits exactly as long as the forecast said, so
every column containing the wait would inherit the prediction instead of testing
it.

The wait also lands on the far side of the request's KV fetch, where a forward pass
that cannot start before its inputs arrive actually has it. Control's arithmetic
puts the wait first, and so charges a fabric transfer to a device that is idle
during it -- a divergence the run can now show.

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

        Forwarded rather than stored, so the host has one answer and it is the one
        the KV was actually cut into.
        """
        return self.compute.block_tokens

    def cost(self, uncached_tokens: int) -> float:
        """What prefilling ``uncached_tokens`` costs on this host.

        Asked of the accelerator: what a forward pass costs is a fact about the
        machine running it. Needed for the one case control cannot have priced -- a
        plan whose remote prefix was evicted between routing and fetching.
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

        ``prompt`` is the uncached suffix of the request's prompt, as ids -- the
        work, not a duration, so a deployment's engine drops in behind this call.
        ``cached`` is the prefix this host pulled out of the store.

        Answers with every KV block this host now holds and did not before (that
        prefix, then the computed suffix) and the request's **first token**, the one
        TTFT is the time to. Both are passed straight through from the accelerator,
        which is what produced them.

        This is where the pass is **submitted**, and therefore where it queues:
        after the caller's fetch, because a forward pass cannot start before its
        inputs are here, and with the tokens that fetch actually left it with -- an
        evicted reuse makes this a bigger pass than the plan priced, and the queue
        slot has to be the real one. ``tag`` (the request's id) fixes the order of
        two passes submitted at the same instant.
        """
        return await self.compute.prefill(prompt, cached, tag=tag)
