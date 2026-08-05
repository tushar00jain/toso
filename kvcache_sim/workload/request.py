"""Domain model: inference requests and prefix-hash block addressing.

A prompt is chunked into fixed ``B``-token blocks. Each block is content-addressed
by a **prefix-hash chain**: a block's key encodes the whole prefix up to it, so two
prompts that share a leading run of blocks share the exact same keys -- prefix
reuse falls out for free. A block key is a plain ``str`` and is used directly as a
key in the **real** TorchStore directory (``Controller.keys_to_storage_volumes``),
so no separate key type is needed.

We model the "hash" as the concatenation of the prompt's *segment ids* up to a
block (a deterministic, collision-free stand-in for a real content hash), so the
sim never needs Python's (salted) ``hash``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


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


def block_keys_for(model_id: str, segments: Sequence[int]) -> Tuple[str, ...]:
    """Build the prefix-hash chain for a prompt made of ``segments``.

    ``block_keys_for("m0", [3, 7, 2])`` -> ``("m0|3", "m0|3|7", "m0|3|7|2")``.
    Sharing a leading run of segments yields identical leading keys, which is
    exactly what makes a shared prefix a single set of entries in the directory.
    ``model_id`` is included so caches for different models never alias.
    """
    keys: List[str] = []
    acc = model_id
    for seg in segments:
        acc = f"{acc}|{seg}"
        keys.append(acc)
    return tuple(keys)


def longest_prefix_run(block_keys: Sequence[str], present: set) -> int:
    """Return how many leading blocks of ``block_keys`` are in ``present``.

    The prefix match stops at the first missing block (a cache is only useful as a
    contiguous prefix), matching block-by-block prefix comparison.
    """
    n = 0
    for k in block_keys:
        if k in present:
            n += 1
        else:
            break
    return n
