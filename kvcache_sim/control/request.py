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
