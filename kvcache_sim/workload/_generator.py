"""Synthetic request generator (seeded, deterministic), and the block keys it
addresses prompts with.

Models the target workload shape: many requests that **share prefixes**.
Each request's prompt is three parts:

* a **system prompt** of ``system_blocks`` blocks common to *every* request (the
  hottest shared prefix),
* a per-**conversation** context of ``conv_base_blocks`` blocks shared by all
  requests of that conversation (a cached document / chat history), and
* a **unique query** suffix of ``query_blocks`` blocks, distinct per request (never
  reused -- it pollutes the cache and is what LRU evicts first).

Conversations are chosen by a **Zipf** popularity law (a few hot conversations
dominate), reproducing the heavy access skew seen in practice (a few blocks
accessed tens of thousands of times, most never reused). Prompts are **bounded**
(fixed prefix + short query) so prefill time -- and thus load -- is stable.

Determinism: a single ``random.Random(seed)`` drives conversation choice and
Poisson arrivals, so the whole request stream -- and every downstream metric -- is
byte-identical across runs of the same seed.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from ..control.request import Request

__all__ = ["make_workload"]


def _block_keys_for(model_id: str, segments: Sequence[int]) -> Tuple[str, ...]:
    """Build the prefix-hash chain for a prompt made of ``segments``.

    ``_block_keys_for("m0", [3, 7, 2])`` -> ``("m0|3", "m0|3|7", "m0|3|7|2")``.
    Sharing a leading run of segments yields identical leading keys, which is
    exactly what makes a shared prefix a single set of entries in the directory.
    ``model_id`` is included so caches for different models never alias.

    The "hash" is modelled as the concatenation of the prompt's *segment ids* up
    to a block -- a deterministic, collision-free stand-in for a real content
    hash, so the sim never needs Python's (salted) ``hash``. Only a generator of
    prompts computes these; the planes are handed the finished chain on a
    :class:`~kvcache_sim.control.request.Request` and treat it as opaque keys.
    """
    keys: List[str] = []
    acc = model_id
    for seg in segments:
        acc = f"{acc}|{seg}"
        keys.append(acc)
    return tuple(keys)


def _zipf_weights(n: int, s: float) -> List[float]:
    """Normalized Zipf weights for ``n`` items with exponent ``s``."""
    raw = [1.0 / ((rank + 1) ** s) for rank in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def make_workload(
    num_requests: int,
    *,
    num_conversations: int = 8,
    system_blocks: int = 4,
    conv_base_blocks: int = 4,
    query_blocks: int = 2,
    zipf_s: float = 1.1,
    arrival_rate: float = 2.5,
    block_tokens: int = 512,
    output_tokens: int = 64,
    model_id: str = "m0",
    seed: int = 0,
) -> List[Request]:
    """Generate a deterministic list of :class:`Request` sorted by arrival time.

    - ``system_blocks``: shared prefix present in every request (segments ``0..``).
    - ``conv_base_blocks``: per-conversation shared context (distinct per conv).
    - ``query_blocks``: unique suffix per request (never reused).
    - ``zipf_s``: conversation popularity skew (higher => hotter head).
    - ``arrival_rate``: Poisson rate for exponential inter-arrival times.
    """
    rng = random.Random(seed)
    weights = _zipf_weights(num_conversations, zipf_s)

    # Shared system prompt segments (identical across all requests -> shared keys).
    system = list(range(system_blocks))

    # Per-conversation fixed context prefix (distinct segment ranges).
    fresh = 1000  # global counter for unique segment ids (namespaced above system)
    conv_prefix: List[List[int]] = []
    for _c in range(num_conversations):
        base = list(range(fresh, fresh + conv_base_blocks))
        fresh += conv_base_blocks
        conv_prefix.append(system + base)

    requests: List[Request] = []
    t = 0.0
    for i in range(num_requests):
        t += rng.expovariate(arrival_rate)
        c = _sample(rng, weights)
        # Unique query suffix for this request (never shared -> never reused).
        query = list(range(fresh, fresh + query_blocks))
        fresh += query_blocks
        segments = conv_prefix[c] + query
        keys = _block_keys_for(model_id, segments)
        requests.append(Request(
            id=f"r{i}",
            arrival=t,
            block_keys=keys,
            prompt_tokens=len(keys) * block_tokens,
            output_tokens=output_tokens,
        ))
    return requests


def _sample(rng: random.Random, weights: List[float]) -> int:
    """Sample an index from ``weights`` (a normalized distribution)."""
    x = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x <= acc:
            return i
    return len(weights) - 1
