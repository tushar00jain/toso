"""Domain model: the inference request all three planes pass around.

A prompt is chunked into fixed ``B``-token blocks. Each block is content-addressed
by a **prefix-hash chain**: a block's key encodes the whole prefix up to it, so two
prompts that share a leading run of blocks share the exact same keys -- prefix
reuse falls out for free. A block key is a plain ``str`` and is used directly as a
key in the **real** TorchStore directory (``Controller.keys_to_storage_volumes``),
so no separate key type is needed.

Building that chain is the *prompt generator's* job
(``workload/_generator.py``) and walking it against a directory snapshot is the
*view's* (``control/_view.py``); both are private to those modules. What is left
here is the request itself.

The prompt is a tensor, and the keys still are not derived from it
------------------------------------------------------------------
A request carries the prompt it was submitted with -- a ``device="meta"`` tensor
of token ids, real dtype and real shape with no storage behind it, the same
compromise the KV blocks are (:mod:`kvcache_sim.workload._accelerator`). It used
to be an ``int`` count, which made the count the only account of the prompt and
left the data plane taking a number where a deployment takes a batch of ids.

What that deliberately does **not** buy is content-addressing. A block key is a
hash of the tokens in the prefix up to it, and a meta tensor has no tokens in it
to hash -- reading one is an error, not a zero. So the chain cannot be computed
from :attr:`Request.prompt` and is still handed in by whatever generated the
prompt, which is exactly the compromise a meta KV block makes: the right object,
the right shape, the right size, no data. Hashing the *shape* instead would be
worse than the stand-in the generator already builds -- every prompt of a given
length would collide, so every request would "reuse" every other one's prefix.
The honest fix is real token ids, which means a real tokenizer and a real corpus,
and that is a workload change rather than a plumbing one.

The chain does not stop at the prompt
-------------------------------------
Generated tokens extend the sequence, so the KV a decode host produces belongs
under keys that continue the same chain -- and :meth:`Request.continuation_keys`
builds them, here, next to the caveat that explains why they cannot be hashed
either. It is the same compromise one step further out: a real engine hashes the
block's tokens, this one concatenates a counter onto the prompt's last key, and
the sharing structure it produces is the structure a content hash would produce
over the same segments.

What it deliberately does not claim is that anything in *this* workload will look
those keys up. A multi-turn front end submits turn N+1 as "prompt + turn N's
output + the new query", so turn N+1's chain really does walk through turn N's
generated blocks and really would hit them -- but the generator here builds each
request's prompt from fixed conversation segments plus a fresh query and never
splices a previous turn's output in (:mod:`kvcache_sim.workload._generator`). So
the entries a decode host publishes are findable and nothing looks for them. That
is a workload gap, stated rather than papered over, and it is the reason
publishing costs capacity here without yet buying a hit rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch

__all__ = ["Request"]


@dataclass(frozen=True)
class Request:
    """One inference request.

    ``block_keys`` is the prefix-hash chain for the prompt (one directory key per
    ``B``-token block). ``prompt_tokens == len(block_keys) * B``. ``output_tokens``
    drives the decode-side occupancy used by the TBT / overload scenarios -- it is
    what control *predicts* against, and no longer what the produced output is
    counted from: the data plane counts the tokens it made (see
    :attr:`kvcache_sim.report.metrics.RequestResult.output_tokens`).
    """

    id: str
    arrival: float
    block_keys: Tuple[str, ...]
    prompt_tokens: int
    output_tokens: int
    #: The prompt itself: a 1-D tensor of ``prompt_tokens`` token ids, as the
    #: caller submitted it. Under simulation it is a ``device="meta"`` tensor and
    #: therefore has no ids *in* it -- see the module docstring for what that
    #: costs (the block keys cannot be derived from it) and why it is still worth
    #: carrying (every downstream signature says "a prompt" instead of "a
    #: number", and a deployment's real ids drop straight in).
    #:
    #: ``compare=False`` because this dataclass is frozen and therefore has a
    #: generated ``__eq__`` and ``__hash__``: comparing two tensors with ``==``
    #: answers with a *tensor*, and the first ``request_a == request_b`` anywhere
    #: would raise "Boolean value of Tensor is ambiguous". What makes two requests
    #: the same request is the id and the chain, not a bytewise read of the
    #: prompt, which for a meta tensor is not even possible.
    prompt: torch.Tensor = field(compare=False)
    #: Which conversation this request continues. Not used to find a *cache* --
    #: that is what the block keys are for -- but a real front end has it (a
    #: session or tenant id) and uses it to pick which host a request lands on.
    conversation: str = ""

    def __post_init__(self) -> None:
        """Refuse a prompt whose length is not the length everything prices.

        ``prompt_tokens`` is what the control plane routes on -- the prefix match,
        the uncached suffix, the predicted TTFT -- and the tensor is what the data
        plane runs the forward pass over. Two descriptions of one prompt that are
        allowed to disagree eventually do, and the failure would be silent in the
        worst way: the scheduler pricing one length while the accelerator computes
        another, with every reported number internally consistent.

        Checked here rather than left to the generator because every construction
        site goes through this, tests included, and a shape read costs nothing on
        a meta tensor.
        """
        if self.prompt.numel() != self.prompt_tokens:
            raise ValueError(
                f"request {self.id!r} says {self.prompt_tokens} prompt tokens and "
                f"carries a prompt of {self.prompt.numel()}: the scheduler would "
                f"price one length and the forward pass would compute the other"
            )

    def continuation_keys(self, count: int) -> Tuple[str, ...]:
        """``count`` directory keys continuing this prompt's chain past its end.

        What the decode host publishes its **generated** KV under. The prompt's
        last key names the whole prompt, so ``continuation_keys(2)`` answers
        ``("<last>|g1", "<last>|g1|g2")``: each key contains the entire prefix
        before it, which is the one property that makes a prefix-hash chain work
        -- a later sequence that really did continue this one walks the same keys
        and stops where they stop.

        **Synthetic, and it says so in the name of the segment.** A real chain
        hashes each block's token ids; this run's tokens are ``device="meta"`` and
        have none (see the module docstring), so the prompt's chain is built from
        the generator's segment ids and this is the same stand-in one step further
        out. ``g<i>`` rather than another integer segment precisely so the two
        cannot collide: the generator's segments are decimal integers, so no prompt
        chain can ever name a key that ends in ``|g1``, and a request's generated
        blocks therefore cannot be mistaken for another request's prompt blocks.

        **A counter, not the content, and that is a modelling limit worth stating.**
        Two requests that generated the same tokens after the same prompt would get
        the same keys here and would in a real system too; two requests that
        generated *different* tokens after the same prompt would collide here and
        would not in a real system. Nothing in this workload can produce that case
        -- every request's prompt chain is unique, because the generator gives each
        one a fresh query segment -- but it is a property of the workload rather
        than of this method, so it is written down instead of assumed.

        Content hashing over the meta tensors the batch produced is *not* the fix
        and is not attempted: a meta token has no id in it to hash, so hashing
        would either raise or, worse, hash the shape and make every generation of a
        given length alias every other one.
        """
        # The prompt's last key when there is one. A request with no prompt blocks
        # is degenerate here (the generator never makes one) but is not a reason to
        # raise: its id is a unique root, so its generated blocks still get keys
        # that collide with nothing.
        acc = self.block_keys[-1] if self.block_keys else self.id
        keys: list[str] = []
        for i in range(1, count + 1):
            acc = f"{acc}|g{i}"
            keys.append(acc)
        return tuple(keys)
