"""The simulated accelerator: :class:`SimulatedAccelerator`.

The implementation of :class:`kvcache_sim.data._compute.Accelerator` that a *run*
supplies, and the one piece of the compute story that could never lift: it answers
what a forward pass costs from a roofline over the model's flop counts and the
target machine's, and it makes the work take that long by sleeping the virtual
clock. A deployment implements the same protocol by running the model and
measuring.

It is also where the run answers the other half of that port: **what a forward pass
produces**. A deployment's KV is whatever its attention kernels wrote; here it is
one ``device="meta"`` tensor per block -- a real ``torch.Tensor`` with a real dtype
and a real byte count and no storage behind it. That pairing is why this class owns
the block size and the served model: the thing that knows what a pass costs is the
thing that knows what it produces, and both come out of the same two descriptors.

What that replaced, and why
---------------------------
The run used to hand the *store* a "block carrier": one ``(shape, dtype)``
:class:`~realsim.seams.transport.TensorDescriptor` sized ``(block_bytes,) uint8``,
written under every key. Three things were wrong with it and all three are fixed by
producing the KV here instead:

* it was not a tensor, so ``put_batch`` -- which types its value, sending a
  ``Tensor``/``DTensor`` down ``Request.from_any`` and everything else down
  ``Request.from_objects`` -- put every KV block on torchstore's **object** path.
  Every kvcache run exercised the one path a KV deployment never takes. A meta
  tensor takes the tensor path and charges the same bytes;
* its shape was a byte count wearing a tensor's clothes (``(block_bytes,) uint8``),
  which is not a shape any KV cache has;
* it lived next to the store, which meant the store had to be told the served model
  and the tokens per block purely to check that the number it had been handed was
  the number the scheduler prices against.

Shape realism is explicitly *not* the goal here and is not attempted: the KV of one
block is a flat 1-D tensor of the right dtype and the right size, not a
``(layers, 2, kv_heads, block_tokens, head_dim)`` region.
:class:`~domain.llm.Model` exposes ``kv_bytes_per_token`` and nothing about layers
or heads, so a "realistic" shape would be invented factors multiplying out to the
same total -- more to read, no more true. What matters, and what is checked at
construction, is that a block occupies exactly
:meth:`~domain.llm.Model.block_bytes` -- the number the scheduler priced the fetch
against.

Occupancy is here too, and it is what makes coupling structural: one of these is
one accelerator, and two engines handed the same object contend on
:attr:`busy_until` while two handed different objects do not.

The prefill queue, and what it replaced
---------------------------------------
Decode has always booked its steps on that occupancy; prefill did not. A prefill
slept the wait the *control plane predicted* (``plan.queue_wait``) and then slept
its own cost, which made the prediction unfalsifiable -- the data plane waited
exactly as long as the scheduler said it would, so a scheduler that mispredicted
its own queue was right by construction in every column the wait lands in: TTFT,
end-to-end latency, and the next request's predicted queue.

Now :meth:`SimulatedAccelerator.prefill` is *submitted* to this object and runs
when this object is free, behind whatever prefill or decode step already has it,
so the wait is what the run produced rather than what the run was told. There is
no switch for the old behaviour and no branch left over from it. A flag would be
right if sleeping a forecast were a defensible alternative model, the way
``contention="none"`` is a defensible model of an uncontended fabric; it is not
one, it is only less true, and a second path nobody runs is a second path
everybody maintains.

The queue is served in an explicitly sorted order -- ``(submission instant, tag)``,
the tag being the request id -- so two passes handed in at the same virtual instant
run in an order the workload fixes, not one the event loop's ready queue produces;
nothing here iterates a set or a dict, and nothing depends on which coroutine a
gather happened to resume first.

Deterministic: the only clock is the running loop's, virtual under simulation, and
every duration comes from the run's profiles rather than from anything measured. A
meta tensor allocates nothing, so a run's peak RSS does not move with the KV it
"stores".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Sequence

import torch

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, Model, prefill_time,
)

from ..data._compute import Accelerator

__all__ = ["BLOCK_TOKENS", "SimulatedAccelerator"]

#: Tokens per KV block. The engine's cache-page size, so it lives with the
#: accelerator that lays the KV out; fixed for every scenario so runs stay
#: comparable (``kvcache_sim.workload._serving`` re-exports it, because the
#: scheduler has to be told the same number to price a prefix match with).
BLOCK_TOKENS = 512


@dataclass
class _Queued:
    """One forward pass waiting for the device, and how it is ordered against the
    others waiting with it.

    ``submitted`` is when the pass was handed in and ``tag`` is the request's id;
    together they are the sort key, and the second half is there because the first
    can tie. Two requests can finish their KV fetches on the very same virtual
    instant, and if the queue then served them in the order the event loop resumed
    them, the run's answer would depend on ready-queue order -- stable today,
    silently different the moment an ``await`` is added anywhere upstream of one of
    them. Sorting on the request id instead makes the order a property of the
    workload.
    """

    submitted: float
    tag: str
    turn: "asyncio.Future" = field(repr=False)


class _PrefillQueue:
    """A single-server queue for one accelerator's forward passes.

    What "the accelerator serialises prefill" is, concretely: one pass at a time,
    each booked on the accelerator's :attr:`SimulatedAccelerator.busy_until` -- the
    same occupancy a decode step books on -- so a prefill waits for a prefill *and*
    for a decode step, and the next decode step waits for the prefill. Nothing here
    knows about coupling; coupling is just what two engines holding one of these
    turns out to mean.

    **Ownership is handed over, never released into a race.** A finishing pass does
    not set "free" and let whoever wakes first take the device; it picks its
    successor and resolves that one's future while the queue stays marked busy. A
    pass submitted at that same instant therefore joins the queue rather than
    slipping in front of a request that has been waiting, which is the check-then-act
    hole the obvious version has.

    No server task, deliberately. A background task driving the queue would need
    every waiter's future failed if it ever died (the shape
    :meth:`~kvcache_sim.data._decode.DecodeEngine._abandon` exists for), because a
    caller parked on a future nobody resolves hangs the run instead of failing it.
    Here the coroutine that owns the device is the caller's own, so an exception
    raised in the pass propagates to the request that submitted it, and the
    ``finally`` still hands the device on.

    Not modelled, and it would matter in a deployment: cancellation. A waiter that
    is cancelled stays in the queue and is handed a device it will not use, which
    costs the next request its turn. Nothing in this simulation cancels a request
    in flight, so the guard is left out rather than written and never exercised.
    """

    def __init__(self, accelerator: "SimulatedAccelerator") -> None:
        self._accelerator = accelerator
        self._waiting: List[_Queued] = []
        #: Whether some pass owns the device. Handed from one pass to the next
        #: without going false in between -- see the class docstring.
        self._running = False

    async def run(self, tokens: int, tag: str) -> None:
        """Queue for the device, then occupy it for what ``tokens`` costs."""
        loop = asyncio.get_running_loop()
        submitted = loop.time()
        if self._running:
            waiter = _Queued(submitted, tag, loop.create_future())
            self._waiting.append(waiter)
            await waiter.turn
        else:
            self._running = True
        try:
            await self._accelerator.wait_until(
                self._accelerator.claim_prefill(tokens)
            )
        finally:
            self._hand_off()

    def _hand_off(self) -> None:
        """Give the device to the longest-waiting pass, or mark it idle."""
        if not self._waiting:
            self._running = False
            return
        self._waiting.sort(key=lambda w: (w.submitted, w.tag))
        self._waiting.pop(0).turn.set_result(None)


class SimulatedAccelerator(Accelerator):
    """One host's compute, priced off a roofline, incurred by sleeping, KV and all.

    Args:
        profile: the target machine's
            :class:`~sim_common.cost_model.MachineProfile` -- what the flops run on.
        model: the served :class:`~domain.llm.Model` -- how many flops the work is,
            and how many bytes of KV a token leaves behind.
        block_tokens: tokens per KV block, i.e. how much of a prompt one published
            block covers.

    Raises:
        ValueError: if a block's modeled byte count is not a whole number of
            elements of the model's compute dtype. That is this class's half of the
            premise the store used to check at construction -- see
            :attr:`block_nbytes`.
    """

    def __init__(
        self,
        *,
        profile=DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        block_tokens: int = BLOCK_TOKENS,
    ) -> None:
        self.profile = profile
        self.model = model
        self._block_tokens = block_tokens
        # KV is cached in the dtype the model computes in, so that is the dtype of
        # what a pass here produces. Taken off the model rather than fixed at
        # float16, because a model that says bfloat16 and a KV cache that is float16
        # would be two different claims about the same bytes.
        self._kv_dtype = getattr(torch, model.compute_dtype, None)
        if not isinstance(self._kv_dtype, torch.dtype):
            raise ValueError(
                f"model.compute_dtype {model.compute_dtype!r} is not a torch dtype, "
                f"so there is no dtype to produce KV in"
            )
        # The premise the KVStore constructor used to enforce, moved to the thing
        # that now produces the bytes: one block must occupy exactly what
        # Model.block_bytes predicts, because that is the number the scheduler
        # prices every fetch against. Here it is not a comparison against a carrier
        # somebody else built -- the count is *derived* from block_bytes, so the two
        # can only disagree if the model's bytes are not a whole number of elements,
        # which is the one case left to refuse.
        nbytes = model.block_bytes(1, block_tokens)
        numel, remainder = divmod(nbytes, self._element_size())
        if remainder:
            raise ValueError(
                f"a {block_tokens}-token KV block is {nbytes}B, which is not a "
                f"whole number of {model.compute_dtype} elements: a block that has "
                f"to round would be charged a different size than the scheduler "
                f"predicted for it"
            )
        self._block_numel = numel
        #: Sim time this accelerator is occupied until. One field, written by the
        #: two things that occupy the device: a decode step (:meth:`claim_step`)
        #: and a forward pass (:meth:`claim_prefill`, through the queue below).
        self.busy_until: float = 0.0
        #: This device's forward passes, one at a time and in a fixed order.
        self._queue = _PrefillQueue(self)

    def _element_size(self) -> int:
        """Bytes per KV element. A 0-element meta tensor allocates nothing."""
        return torch.empty(0, dtype=self._kv_dtype, device="meta").element_size()

    # -- what this accelerator's KV looks like ------------------------------ #
    @property
    def block_tokens(self) -> int:
        """:class:`~kvcache_sim.data._compute.Accelerator` -- tokens per KV block."""
        return self._block_tokens

    @property
    def block_nbytes(self) -> int:
        """Bytes one KV block occupies -- what the transport charges to move it.

        Equal to ``model.block_bytes(1, block_tokens)`` by construction, which is
        the coupling the whole cost story rests on: the scheduler predicts a fetch
        against that number and the transport charges the tensor's real
        ``numel * element_size``. ``kvcache_sim/tests/test_cost_premises.py``
        asserts it against a produced block rather than against this property, so
        the check survives someone making this property lie.
        """
        return self._block_numel * self._element_size()

    def kv_blocks(self, count: int) -> List[torch.Tensor]:
        """``count`` blocks of KV, as a forward pass would have left them.

        Allocation-free: ``device="meta"`` tensors have exact shape, dtype and
        ``nbytes`` and no storage, so a run can "hold" a terabyte of KV in a test
        process. They are real ``torch.Tensor`` objects, which is what puts every
        publish on torchstore's tensor path.

        A distinct object per block rather than one shared N times. The store keys
        them separately, a volume evicts them separately, and one aliased tensor
        under every key is exactly the shortcut that let the old carrier be a
        descriptor: sharing costs nothing here, and pretending each block is its own
        thing costs nothing either.
        """
        return [
            torch.empty(self._block_numel, dtype=self._kv_dtype, device="meta")
            for _ in range(count)
        ]

    def blocks_for(self, tokens: int) -> int:
        """How many KV blocks ``tokens`` tokens fill (a partial block still is one)."""
        return max(0, -(-tokens // self._block_tokens))

    # -- what work costs --------------------------------------------------- #
    def prefill_cost(self, tokens: int) -> float:
        return prefill_time(tokens, self.profile, self.model)

    def step_cost(self, batch_size: int) -> float:
        return decode_step_time(batch_size, self.profile, self.model)

    # -- making it take that long ------------------------------------------ #
    def claim_prefill(self, tokens: int) -> float:
        """Book a forward pass after whatever already has this accelerator.

        The prefill twin of :meth:`claim_step`, and deliberately the same two
        lines: one occupancy, written the one way, so a prefill and a decode step
        cannot each think they have the device. Not on the
        :class:`~kvcache_sim.data._compute.Accelerator` port, though, and the
        asymmetry with ``claim_step`` is intended -- decode has to *announce* a
        step's end before awaiting it, because a coupled host reports it onward to
        the control plane while there is still time to schedule around it, whereas
        nothing outside this object needs to know when a prefill will end before it
        does. So this is an implementation detail of the queue and stays one.
        """
        now = asyncio.get_running_loop().time()
        self.busy_until = max(now, self.busy_until) + self.prefill_cost(tokens)
        return self.busy_until

    async def prefill(
        self, tokens: int, cached: Sequence[torch.Tensor] = (), *, tag: str = ""
    ) -> List[torch.Tensor]:
        """Submit the forward pass; it runs when this device is free.

        The pass joins this accelerator's single-server queue and is booked on the
        same :attr:`busy_until` a decode step books on, so it waits for a prefill
        ahead of it *and* for a decode step in progress, and the next decode step
        waits for it. Everything between submission and the pass finishing is the
        request's real queue wait, and no number the control plane produced took
        part in producing it -- which is the whole point, since the control plane's
        forecast of that same wait is what the run is now able to check.

        A pass with nothing to compute (a prompt entirely covered by cached blocks)
        does not queue. Zero flops occupy no device, and letting an empty submission
        take a turn would make a fully-cached request wait behind a full-length
        prefill for the privilege of doing nothing.

        Answers with ``cached`` (the prefix this host pulled, which a real engine
        would have written into its cache before attending over it) followed by one
        fresh block per block of ``tokens``. Nothing is charged for carrying the
        pulled blocks through: the fetch that produced them already paid, and a
        forward pass over an uncached suffix is what this is priced as.

        The pulled blocks are passed along **as they arrived** rather than replaced
        with locally-made ones of the same size. They are the objects the store
        returned, so a run that later cares whether the bytes a host publishes are
        the bytes it pulled has something to compare; making new ones here would be
        a simulation detail quietly deciding they are interchangeable.
        """
        if self.prefill_cost(tokens) > 0:
            await self._queue.run(tokens, tag)
        return [*cached, *self.kv_blocks(self.blocks_for(tokens))]

    def claim_step(self, batch_size: int) -> float:
        """Book a decode step after whatever already has this accelerator."""
        now = asyncio.get_running_loop().time()
        self.busy_until = max(now, self.busy_until) + self.step_cost(batch_size)
        return self.busy_until

    async def wait_until(self, when: float) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(when - loop.time())
