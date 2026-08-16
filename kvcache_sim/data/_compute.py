"""The accelerator an engine runs on, as the engine sees it: :class:`Accelerator`.

Two things a serving host does are not store calls: it runs a forward pass, and it runs
decode steps. So the compute is a port. An engine says *what* work it has; the thing
behind this port says how long that takes and makes it take that long -- under
simulation a roofline and an ``asyncio.sleep``
(:class:`kvcache_sim.workload._accelerator.SimulatedAccelerator`), in a deployment the
model's forward pass with measured durations.

The compute counterpart of :data:`proposed.cost.TransferCost`, which only predicts:
fabric cost is *incurred* by the client call the data plane already makes, while
nothing else makes a GPU busy, so this both predicts and incurs.

What the port hands back
------------------------
:meth:`Accelerator.prefill` answers with the KV blocks the pass produced and with the
request's **first token**, sampled from its last position -- the token TTFT is the time
to. :attr:`Accelerator.block_tokens` says how many tokens one block holds, since the
block size is the engine's own cache layout. What the store then publishes is whatever
it is handed: ``device="meta"`` tensors under simulation, attention output in a
deployment, neither a shape this port describes.

:meth:`Accelerator.step_tokens` is the decode-side twin: the one token per batch member
a step just emitted. Between them the two account for every token of the answer, and
the client walking the redirect chain collects both halves. Not streaming: decode
answers with its tokens when the request finishes.

:meth:`Accelerator.generated_kv` is the other decode-side twin. Every generated token
is fed back in at the next step and leaves a position of KV behind it, so ``n``
generated tokens grow the decode host's cache by ``ceil(n / block_tokens)`` blocks.
Without it a volume holds only what a *prefill* published and decode is free in
capacity terms, flattering every eviction and hit-rate number a decode-simulating run
reports.

One object is one accelerator
-----------------------------
A host has one and hands it to whichever engines run on it, and whether prefill and
decode were handed the **same** object is the whole of what "coupled" means: two
engines on one accelerator collide, so a long prefill delays the next decode step. A
run that models them as not colliding hands out two.

One accelerator runs one thing at a time, which is why the occupancy is here and not in
an engine. A decode step books it (:meth:`claim_step`) and a forward pass runs on it
(:meth:`prefill`), so whichever asks second waits for the first without either engine
knowing the other exists. An engine never *reads* the occupancy -- it asks for work and
is told when it was done -- so there is no "when are you free?" member and no way to
schedule around the other engine except by queueing behind it. One owner of
``busy_until``, and no prediction is written over a measurement of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import torch

__all__ = ["Accelerator"]


class Accelerator(ABC):
    """What an engine may ask of the compute it runs on.

    Split into asking what work costs and doing it, because only one of the two is a
    prediction. A deployment answers the cost members by measuring what it has already
    run; a simulation answers them from a roofline and then sleeps exactly that long,
    so a simulated run's charged time equals its predicted time.

    Every member is abstract: a stub answering zero would make a run that forgot to
    supply one look merely fast.
    """

    @property
    @abstractmethod
    def block_tokens(self) -> int:
        """Tokens one KV block holds -- this engine's cache-page size.

        What :meth:`prefill` returns one tensor per. Read off the accelerator rather
        than configured again next to the store: two numbers that have to agree and are
        set in two places eventually do not.
        """

    @abstractmethod
    def prefill_cost(self, tokens: int) -> float:
        """What prefilling ``tokens`` uncached tokens costs here.

        A count, where :meth:`prefill` takes the ids: this asks about work that may
        never be submitted (re-pricing a plan whose reuse was evicted), and a tensor
        would oblige every caller to materialise a prompt to ask.
        """

    @abstractmethod
    def step_cost(self, batch_size: int) -> float:
        """What one decode step over ``batch_size`` requests costs here."""

    @abstractmethod
    async def prefill(
        self, prompt: torch.Tensor, cached: Sequence[torch.Tensor] = (), *,
        tag: str = "",
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Submit a forward pass over ``prompt``; answer with its KV and first token.

        **Runs when this accelerator is free to run it.** The pass books the same
        occupancy every other kind of work does, so it queues behind a prefill already
        running and behind a decode step in progress (:meth:`claim_step`), and the next
        decode step queues behind it. How long a request waited is therefore the
        caller's to measure; nothing here is told.

        ``prompt`` is the **uncached suffix**: the ids this pass computes KV for, with
        whatever ``cached`` covers chopped off the front. Under simulation a
        ``device="meta"`` tensor -- right length and dtype, no ids in it -- which is why
        a request's block keys are generated alongside the prompt rather than hashed out
        of it (:class:`kvcache_sim.control.request.Request`).

        ``cached`` is the KV this host pulled for the prefix in front of those tokens.
        Passed in because fetching it is a store call and this port makes none; the
        engine loads it into the cache the pass attends over, so it comes back out in
        the answer.

        ``tag`` is a stable name for this submission (the request's id), needed to
        *order* it: an implementation serves its queue by ``(submitted at, tag)``, so
        two passes handed in at the same virtual instant run in an order the workload
        fixes. Ordering by whichever coroutine resumed first would reorder when an
        unrelated ``await`` is added upstream of a caller.

        Answers with **one tensor per KV block this host now holds and did not before**,
        in prompt order (the pulled prefix, then the computed suffix) -- exactly the set
        the caller publishes. Returning only the new blocks would move the ordering of a
        request's KV into the serving loop, where an off-by-one publishes real bytes
        under the wrong keys.

        And with the request's **first token**, sampled from the pass's last position:
        exactly one however long the prompt was, everything after it decode's
        (:meth:`step_tokens`), and returned unconditionally, since a fully cached prompt
        still attends and samples its last position.
        """

    @abstractmethod
    def generated_kv(self, positions: int) -> List[torch.Tensor]:
        """The KV one request's ``positions`` generated tokens left in this cache.

        One tensor per **block**, in generation order, as :meth:`prefill` answers for a
        prompt: ``ceil(positions / block_tokens)`` of them, because a paged cache
        allocates whole blocks and a generation running a single token past a boundary
        has taken the next block. ``positions == 0`` answers with nothing -- a request
        whose whole output was the prefill's first token appended no KV.

        **The trailing block is charged whole**, at a full block's size, which is what
        paged allocation does: the remainder is internal fragmentation the host really
        pays for. It is usually partial -- 31 generated positions in a 512-token block
        here -- and dropping it would make decode residency vanish outright, since no
        scenario here generates the 512 tokens it takes to fill one. Charging exact
        positions would be a size no key in this system names.

        The whole generation arrives at once rather than per step, so the caller makes
        one publish instead of reassembling ``positions`` sub-block tensors. **Missing:**
        intra-generation residency, charged when the generation ends rather than as it
        grows; fixing it means a store call inside the step loop
        (:meth:`kvcache_sim.data.serving.ServingHost.decode`).

        Not async and not charged: the steps that produced these positions were paid
        for one by one.
        """

    @abstractmethod
    def claim_step(self, batch_size: int) -> float:
        """Book the next decode step and answer when it will end.

        Two members rather than one ``await decode_step(...)``: the caller has to
        announce the end *before* waiting for it, since a report arriving after the
        step would tell control about a collision it could no longer avoid.
        """

    @abstractmethod
    async def wait_until(self, when: float) -> None:
        """Wait until ``when``, the moment :meth:`claim_step` answered with."""

    @abstractmethod
    def step_tokens(self, batch_size: int) -> List[torch.Tensor]:
        """The tokens a finished decode step emitted -- one per batch member.

        Asked after :meth:`claim_step` and :meth:`wait_until`, neither of which can
        answer with tokens: the first runs before the step, the second knows nothing
        about a batch.

        In batch order, one per member, so the caller pairs them positionally with the
        batch it froze for the step. One ``(batch_size,)`` tensor -- what a sampling
        kernel writes -- would be indexed against a list held separately, and a batch
        reordered between the claim and the read would attribute every token to the
        wrong request with nothing raising.

        Not async and not charged: the step's whole cost is the claim, sampling
        included.
        """
