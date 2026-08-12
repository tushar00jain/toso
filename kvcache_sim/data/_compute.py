"""The accelerator an engine runs on, as the engine sees it: :class:`Accelerator`.

Two things a serving host does are not store calls: it runs a forward pass, and it
runs decode steps. Everything else in ``data/`` is an ordinary torchstore call that
lifts unchanged, so the compute is a port instead. An engine says *what* work it
has; the thing behind this protocol says how long that takes and makes it take that
long. Under simulation that is a roofline and an ``asyncio.sleep``
(:class:`kvcache_sim.workload._accelerator.SimulatedAccelerator`, which lives with
the run's wiring because it is the simulation); in a deployment it is the model's
forward pass, with measured durations.

The compute counterpart of :class:`proposed.cost.TransferCost`, and it does one
thing that port does not: fabric cost is *incurred* by the real client call the data
plane already makes, so predicting it is enough, whereas nothing else makes a GPU
busy -- so this both predicts and incurs.

A pass produces KV and tokens, so this port hands both back
-----------------------------------------------------------
:meth:`Accelerator.prefill` answers with the KV blocks the pass produced and with
the request's **first token**, sampled from its last position -- the token TTFT is
the time to. :attr:`Accelerator.block_tokens` says how many tokens one block holds.
All of it lives here rather than beside the store because the thing that knows what
a forward pass *costs* is the thing that knows what it *produces*: the block size is
the engine's own cache layout and the tensors are its own output. So the serving
host publishes what prefill handed it, and the store publishes what it is handed --
``device="meta"`` tensors under simulation, attention output in a deployment,
neither a shape this port has to describe.

:meth:`Accelerator.step_tokens` is the decode-side twin: the one token per batch
member a step just emitted. Between them the two members account for every token of
the answer, and the client walking the redirect chain collects both halves.
Deliberately **not** streaming -- decode answers with its tokens when the request
finishes, which is the ``stream=False`` shape of every serving API and the one that
matches a leg that already returns at the last token.

:meth:`Accelerator.generated_kv` is the other decode-side twin. Every generated
token is fed back in at the next step and leaves a position of KV behind it, so a
request that generates ``n`` tokens grows its decode host's cache by
``ceil(n / block_tokens)`` blocks. Without it a volume only ever holds what a
*prefill* published, decode is free in capacity terms, and every eviction and
hit-rate number a decode-simulating run reports is flattered by the memory the
generation should have been taking.

One object is one accelerator
-----------------------------
A host has one and hands it to whichever engines run on it. Whether its prefill and
decode engines were handed the **same** object is the whole of what "coupled" means:
two engines on one accelerator collide, so a long prefill delays the next decode
step. A run that models them as not colliding hands out two, which is the wiring's
choice, not a flag an engine carries.

...and one accelerator runs one thing at a time, which is why the occupancy is here
and not in an engine. A decode step books it (:meth:`claim_step`) and a forward pass
runs on it (:meth:`prefill`), so the two compose without either engine knowing the
other exists: whichever asks second waits for the first. An engine never *reads* the
occupancy -- it asks for work and is told when it was done -- so there is no "when
are you free?" member and no way to schedule around the other engine except by
queueing behind it. One owner of ``busy_until``, and no prediction is ever written
over a measurement of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import torch

__all__ = ["Accelerator"]


class Accelerator(ABC):
    """What an engine may ask of the compute it runs on.

    Split into asking what work costs and doing it, because both are needed and
    only one of them is a prediction. A deployment answers the cost members by
    measuring what it has already run; a simulation answers them from a roofline
    and then sleeps exactly that long, which is what makes a simulated run's
    charged time equal its predicted time.

    A base class rather than a ``Protocol``, for the reason
    :class:`proposed.coordinator.Coordinator` gives: the protocols in this repo are
    surfaces a caller *reaches*, behind a handle whose real implementation is
    somebody else's class. This is a surface somebody *writes*, in this process, so
    an implementation should say so in its bases and be told at construction when it
    has missed a member.

    Every member is abstract: a stub that answered zero would make a run that forgot
    to supply one look merely fast.
    """

    @property
    @abstractmethod
    def block_tokens(self) -> int:
        """Tokens one KV block holds -- this engine's cache-page size.

        A fact about how the engine lays its KV cache out, and therefore what
        :meth:`prefill` returns one tensor per. Read off the accelerator rather than
        configured a second time next to the store: two numbers that have to agree
        and are set in two places eventually do not.
        """

    @abstractmethod
    def prefill_cost(self, tokens: int) -> float:
        """What prefilling ``tokens`` uncached tokens costs here.

        A count, where :meth:`prefill` takes the ids: this is a question about work
        that may never be submitted (the serving host asks it to re-price a plan
        whose reuse was evicted), so requiring a tensor would oblige every caller to
        materialise a prompt to ask about one.
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

        **Runs when this accelerator is free to run it.** The pass is booked on the
        same occupancy every other kind of work books on, so it queues behind a
        prefill already running and behind a decode step in progress
        (:meth:`claim_step`), and the next decode step queues behind it -- which is
        what makes "coupled" cost something rather than merely mean two engines were
        handed one object. How long a request waited is therefore something the
        caller measures rather than something anything here is told.

        ``prompt`` is the **uncached suffix** of the request's prompt: the ids this
        pass has to compute KV for, with whatever ``cached`` covers chopped off the
        front. Ids rather than a count, because an engine runs a model over ids.
        Under simulation it is a ``device="meta"`` tensor -- right length and dtype,
        no ids in it -- which is why a request's block keys are generated alongside
        the prompt rather than hashed out of it
        (:class:`kvcache_sim.control.request.Request`).

        ``cached`` is the KV this host pulled for the prefix in front of those
        tokens. Passed in rather than fetched here, because getting it is a store
        call and this port makes none; an engine loads it into the cache the pass
        attends over, which is why it comes back out in the answer.

        ``tag`` is a stable name for this submission (the request's id). Not needed
        to run the pass, needed to *order* it: an implementation serves its queue by
        ``(submitted at, tag)``, so two passes handed in at the same virtual instant
        run in an order the workload fixes rather than one the event loop's ready
        queue happens to produce. Ordering by whichever coroutine resumed first
        would reorder when an unrelated ``await`` is added upstream of a caller.

        Answers with **one tensor per KV block this host now holds and did not
        before**, in prompt order (the pulled prefix, then the computed suffix),
        which is exactly the set the caller publishes -- returning only the new
        blocks would put the ordering of a request's KV in the serving loop, where a
        silent off-by-one publishes real bytes under the wrong keys. Per-block
        rather than a contiguous region plus a block table, because per-block is
        what the store wants and what a paged engine already has.

        And with the request's **first token**, sampled from the pass's last
        position, which is what makes the reported TTFT a time to a token that
        exists. Exactly one, however long the prompt was; everything after it is
        decode's (:meth:`step_tokens`). Returned unconditionally, since a fully
        cached prompt still attends and samples its last position.
        """

    @abstractmethod
    def generated_kv(self, positions: int) -> List[torch.Tensor]:
        """The KV one request's ``positions`` generated tokens left in this cache.

        One tensor per **block**, in generation order, exactly as :meth:`prefill`
        answers for a prompt: ``ceil(positions / block_tokens)`` of them, because a
        paged cache allocates whole blocks and a generation that runs a single
        token past a boundary has taken the next block whether or not it fills it.
        ``positions == 0`` answers with nothing: a request whose whole output was the
        prefill's first token ran no decode step and appended no KV.

        **Whole blocks, and the trailing one is charged whole.** The last block of a
        generation is usually partial -- 31 generated positions in a 512-token block
        here -- and is still returned at a full block's size, which is what paged
        allocation does: the block manager hands out a physical block for the
        position that first lands in it, and the remainder is internal
        fragmentation the host really pays for. Dropping the partial block would
        make decode residency vanish outright at this model's block size, since no
        scenario here generates the 512 tokens it takes to fill one. Charging exact
        positions would be a size no key in this system names -- the store's unit is
        a block and the volume's accounting is per key.

        **The whole generation at once, not one call per step**, unlike
        :meth:`step_tokens`. A token is accumulated as it is produced; the KV is a
        single publish, and asking per position would hand the caller ``positions``
        sub-block tensors to reassemble into the blocks it was always going to
        publish. The cost is intra-generation residency -- a generation's bytes are
        charged when it ends rather than as it grows -- and fixing that means a
        store call inside the step loop (see
        :meth:`kvcache_sim.data.serving.ServingHost.decode`).

        Not async and not charged, for :meth:`step_tokens`' reason: the steps that
        produced these positions were already paid for one by one.
        """

    @abstractmethod
    def claim_step(self, batch_size: int) -> float:
        """Book the next decode step and answer when it will end.

        Two members rather than one ``await decode_step(...)`` because the caller has
        to announce the end *before* waiting for it: a coupled host reports each
        step's end onward, and a report arriving after the step would tell the
        control plane about a collision it could no longer avoid.
        """

    @abstractmethod
    async def wait_until(self, when: float) -> None:
        """Wait until ``when``, the moment :meth:`claim_step` answered with."""

    @abstractmethod
    def step_tokens(self, batch_size: int) -> List[torch.Tensor]:
        """The tokens a finished decode step emitted -- one per batch member.

        A third member rather than a return value on the other two:
        :meth:`claim_step` answers before the step has run, and ``wait_until``
        answers for whoever is waiting on the clock and knows nothing about a batch.
        So the caller books, waits, then asks what came out -- the order a real step
        loop runs in.

        In batch order, one per member, so the caller pairs them positionally with
        the batch it froze for the step. One ``(batch_size,)`` tensor -- what a
        sampling kernel actually writes -- would have the engine index it against a
        list it holds separately, and a batch that reordered between the claim and
        the read would attribute every token to the wrong request with nothing
        raising.

        Not async and not charged: the step's whole cost is the claim, and sampling
        is inside it.
        """
