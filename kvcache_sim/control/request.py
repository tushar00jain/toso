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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["Request"]


@dataclass(frozen=True)
class Request:
    """One inference request.

    ``block_keys`` is the prefix-hash chain for the prompt (one directory key per
    ``B``-token block). ``prompt_tokens == len(block_keys) * B``. ``output_tokens``
    drives the decode-side occupancy used by the TBT / overload scenarios.
    """

    id: str
    arrival: float
    block_keys: Tuple[str, ...]
    prompt_tokens: int
    output_tokens: int
