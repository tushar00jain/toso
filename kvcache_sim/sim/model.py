"""Domain model: instances, KV blocks, requests and prefix-hash addressing.

A prompt is chunked into fixed ``B``-token blocks. Each block is content-addressed
by a **prefix-hash chain** (K1): a block's key encodes the whole prefix up
to it, so two prompts that share a leading run of blocks share the exact same keys
-- dedup and prefix reuse fall out for free.

We model the "hash" as the concatenation of the prompt's *segment ids* up to a
block (a deterministic, collision-free stand-in for a real content hash), so the
sim never needs Python's (salted) ``hash``. A KV block's byte size is
``B * bytes_per_token``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

BlockKey = str  # prefix-hash chain id, e.g. "m0|3|7|2"


@dataclass(frozen=True)
class Instance:
    """A serving instance owning one KV-cache volume (KV-cache pool).

    ``host`` is the shared-memory domain, ``node`` the NVLink / intra-node domain;
    these drive the locality cost tier for cross-instance KV transfer.
    """

    id: str
    host: str
    node: str


@dataclass(frozen=True)
class Request:
    """One inference request.

    ``block_keys`` is the prefix-hash chain for the prompt (one key per ``B``-token
    block). ``prompt_tokens`` == ``len(block_keys) * B``. ``output_tokens`` drives
    the decode-side occupancy used by the overload/SLO scenario.
    """

    id: str
    arrival: float
    block_keys: Tuple[BlockKey, ...]
    prompt_tokens: int
    output_tokens: int


def block_keys_for(model_id: str, segments: Sequence[int]) -> Tuple[BlockKey, ...]:
    """Build the prefix-hash chain for a prompt made of ``segments``.

    ``block_keys_for("m0", [3, 7, 2])`` -> ``("m0|3", "m0|3|7", "m0|3|7|2")``.
    Sharing a leading run of segments yields identical leading keys (K1), which is
    exactly what makes the shared prefix a single set of entries in the store.
    ``model_id`` is included so caches for different models never alias.
    """
    keys: List[BlockKey] = []
    acc = model_id
    for seg in segments:
        acc = f"{acc}|{seg}"
        keys.append(acc)
    return tuple(keys)


def longest_prefix_run(block_keys: Sequence[BlockKey], present: set) -> int:
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


def block_bytes(num_blocks: int, block_tokens: int, bytes_per_token: int) -> int:
    """Byte size of ``num_blocks`` KV blocks."""
    return num_blocks * block_tokens * bytes_per_token
