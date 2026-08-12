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

The keys are not derived from the prompt
----------------------------------------
A request carries the prompt it was submitted with -- under simulation a
``device="meta"`` tensor of token ids: real dtype and shape, no storage behind it,
the same compromise the KV blocks are (:mod:`kvcache_sim.workload._accelerator`).
A meta tensor has no tokens *in* it to hash (reading one is an error, not a zero),
so the block-key chain cannot be computed from :attr:`Request.prompt` and is handed
in by whatever generated it. Hashing the *shape* instead would be worse: every
prompt of a given length would collide, so every request would "reuse" every other
one's prefix. The honest fix is real token ids, i.e. a real tokenizer and corpus,
which is a workload change rather than a plumbing one.

Generated tokens extend the sequence, so the KV a decode host produces belongs
under keys that continue the same chain; :meth:`Request.continuation_keys` builds
them, with the same stand-in one step further out.
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
    is what control *predicts* decode occupancy against; the produced output is
    counted by the data plane from the tokens it actually made (see
    :attr:`kvcache_sim.report.metrics.RequestResult.output_tokens`).
    """

    id: str
    arrival: float
    block_keys: Tuple[str, ...]
    prompt_tokens: int
    output_tokens: int
    #: The prompt itself: a 1-D tensor of ``prompt_tokens`` token ids, as the
    #: caller submitted it. Under simulation a ``device="meta"`` tensor with no ids
    #: in it -- see the module docstring.
    #:
    #: ``compare=False``: this dataclass is frozen and so has a generated ``__eq__``
    #: / ``__hash__``, and comparing two tensors with ``==`` answers with a *tensor*,
    #: so the first ``request_a == request_b`` anywhere would raise "Boolean value of
    #: Tensor is ambiguous". Two requests are the same request by id and chain.
    prompt: torch.Tensor = field(compare=False)
    #: Which conversation *turn* this request continues. Not used to find a cache --
    #: that is the block keys' job -- but to pick which host a request lands on, the
    #: way a real front end uses a session or tenant id.
    conversation: str = ""

    def __post_init__(self) -> None:
        """Refuse a prompt whose length is not the length everything prices.

        ``prompt_tokens`` is what the control plane routes on (prefix match,
        uncached suffix, predicted TTFT) and the tensor is what the data plane runs
        the forward pass over. If they disagree the scheduler prices one length
        while the accelerator computes another, and every reported number stays
        internally consistent while being wrong.
        """
        if self.prompt.numel() != self.prompt_tokens:
            raise ValueError(
                f"request {self.id!r} says {self.prompt_tokens} prompt tokens and "
                f"carries a prompt of {self.prompt.numel()}: the scheduler would "
                f"price one length and the forward pass would compute the other"
            )

    def continuation_keys(self, count: int) -> Tuple[str, ...]:
        """``count`` directory keys continuing this prompt's chain past its end.

        Each key contains the whole prefix before it, so ``continuation_keys(2)``
        answers ``("<last>|g1", "<last>|g1|g2")`` -- a later sequence that really did
        continue this one walks the same keys and stops where they stop.

        Two callers, and it matters that they share this method rather than spelling
        the keys out twice: the decode host publishing its generated KV
        (:meth:`kvcache_sim.data.serving.ServingHost.decode`) and the workload
        building turn N+1 out of turn N (:mod:`kvcache_sim.workload._generator`).

        **Synthetic.** A real chain hashes each block's token ids; this run's tokens
        are ``device="meta"`` and have none, so this concatenates a counter instead.
        ``g<i>`` cannot collide with a prompt key: the generator's segments are
        decimal integers, so no prompt chain names a key ending in ``|g1``.

        Modelling limit: two requests generating *different* tokens after the same
        prompt would collide here and would not in a real system. This workload
        cannot produce that -- every prompt chain is unique, because each request
        gets a fresh query segment -- but that is a property of the workload.
        """
        # A request with no prompt blocks is degenerate (the generator never makes
        # one) but not an error: its id is a unique root.
        acc = self.block_keys[-1] if self.block_keys else self.id
        keys: list[str] = []
        for i in range(1, count + 1):
            acc = f"{acc}|g{i}"
            keys.append(acc)
        return tuple(keys)
