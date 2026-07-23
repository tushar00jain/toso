"""Schedulers under test: ``LoadBalanceScheduler`` and ``CacheAwareScheduler``.

Both share an interface so scenarios/tests can swap them:

    schedule(request, now) -> Plan | None      # None == rejected (SLO/overload)
    on_complete(plan) -> list[evicted]         # publish blocks, admit, evict

``schedule`` models the cache-aware coordinator's serialized mailbox: in the DES it
runs to completion before the next event, so routing sees a consistent directory
snapshot. Prefill cost is deterministic, so the *predicted* TTFT used for routing
equals the *actual* completion time (prefill time is highly predictable).

* ``LoadBalanceScheduler`` (baseline, ~vLLM): route to the least-loaded instance;
  reuse only that instance's **local** cache; never pull a remote prefix. This is
  "balance by load, cache is local-only."
* ``CacheAwareScheduler`` (cache-aware coordinator): route to minimize predicted TTFT
  using the **global** prefix-match directory, optionally pulling a remote prefix
  (which read-through-populates -> hot-block replication) under a balance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import cost
from .cache import LRUCache
from .cost import (
    locality,
    prefill_time,
    transfer_time,
)
from .decode import DecodeEngine
from sim_common.engine import Sim
from .index import BlockIndex
from .model import block_bytes, Instance, longest_prefix_run, Request


@dataclass
class Plan:
    """A routing decision for one request (or a rejection when ``None``)."""

    request: Request
    prefill: str                 # chosen prefill instance id
    decode: str                  # chosen decode instance id
    match_blocks: int            # reused prefix length (blocks)
    cached_tokens: int
    uncached_tokens: int
    reuse_source: Optional[str]  # remote instance a prefix gap is pulled from
    transfer_bytes: int
    queue_wait: float
    ttft: float                  # time-to-first-token (queue + transfer + prefill)
    done_time: float             # absolute sim time prefill completes
    decode_done: float
    pred_tbt: float = 0.0        # predicted time-between-tokens at admission
    pred_batch: int = 0          # predicted decode batch size at admission


class _Base:
    """Shared state + prediction/commit helpers for both schedulers."""

    def __init__(self, sim: Sim, index: BlockIndex, instances: List[Instance],
                 block_tokens: int, bytes_per_token: int,
                 capacity: Optional[int] = None,
                 decode_pool: Optional[List[str]] = None,
                 prefill_pool: Optional[List[str]] = None,
                 slo_ttft: float = float("inf"),
                 slo_tbt: float = float("inf"),
                 simulate_decode: bool = False, max_batch: int = 8,
                 coupled: bool = False, early_rejection: str = "off") -> None:
        self.sim = sim
        self.index = index
        self.topo: Dict[str, Instance] = {i.id: i for i in instances}
        self.ids: List[str] = sorted(self.topo)
        self.B = block_tokens
        self.bpt = bytes_per_token
        self.caches: Dict[str, LRUCache] = {
            i: LRUCache(capacity) for i in self.ids
        }
        # Prefill queue tail over ALL instances (the disaggregated-prefill pool
        # may be a subset; decode may be a disjoint or overlapping subset).
        self.busy_until: Dict[str, float] = {i: 0.0 for i in self.ids}
        self.prefill_ids: List[str] = sorted(prefill_pool) if prefill_pool else self.ids
        self.decode_ids: List[str] = sorted(decode_pool) if decode_pool else self.ids
        self.slo_ttft = slo_ttft
        self.slo_tbt = slo_tbt
        # ``off``  -> no TBT gate at routing; late-reject at decode admission.
        # ``early``-> reject at routing on predicted current occupancy.
        # ``predict`` -> reject at routing on occupancy predicted at prefill done,
        #             including in-flight prefills that will land on that pool.
        self.early_rejection = early_rejection
        self.max_batch = max_batch
        self.coupled = coupled
        self.tbt_enabled = simulate_decode
        # The client installs this to be notified when a request emits its last
        # decode token (carrying that request's worst observed inter-token gap).
        self.on_decode_finish = None
        # Prefills reserved but not yet admitted to decode, used only by the
        # ``predict`` mode: (prefill_done_time, decode_id, output_tokens).
        self._inflight: List[Tuple[float, str, int]] = []
        if simulate_decode:
            # A COUPLED instance shares its compute timeline with prefill (a long
            # prefill stalls the next decode step). A DISAGGREGATED pool gets its
            # own private timeline (engine makes one), so prefill never blocks it.
            self.engine: Optional[DecodeEngine] = DecodeEngine(
                sim, self.decode_ids, max_batch=max_batch,
                compute_busy=(self.busy_until if coupled else None),
                on_finish=self._decode_finished,
            )
        else:
            self.engine = None

    # -- prediction (no mutation) ---------------------------------------- #
    def _predict(self, inst: str, now: float, transfer_t: float,
                 prefill_t: float):
        """Return ``(queue_wait, ttft, done_time)`` without reserving the server."""
        avail = max(now, self.busy_until[inst])
        queue_wait = avail - now
        done = avail + transfer_t + prefill_t
        return queue_wait, done - now, done

    # -- decode-side TBT prediction / admission -------------------------- #
    def _decode_finished(self, request: Request, tbt: float) -> None:
        """Bridge the engine's completion callback to the client (if installed)."""
        if self.on_decode_finish is not None:
            self.on_decode_finish(request, tbt)

    def _predicted_batch(self, d: str, done_time: float) -> int:
        """Predicted decode batch size on ``d`` seen by a request admitted at
        ``done_time`` (its prefill completion). Drives TBT prediction.

        ``off``/``early`` use the live occupancy now (a cheap snapshot); ``predict``
        estimates the occupancy at ``done_time`` -- how many current occupants are
        still decoding then, plus any *in-flight* prefills that will have landed on
        ``d`` and not yet drained. This is the Mooncake decode-load prediction that
        lets the scheduler reject before wasting prefill.
        """
        if self.engine is None:
            return 0
        if self.early_rejection == "predict":
            n = self.engine.predict_occupancy(d, done_time)
            for pd_done, dec_id, out in self._inflight:
                if (dec_id == d and pd_done <= done_time
                        and pd_done + max(0, out - 1) * cost.TBT_BASE > done_time):
                    n += 1
            return n
        return self.engine.occupancy(d)

    def _select_decode(self, done_time: float) -> Tuple[str, int]:
        """Pick the decode instance with the smallest predicted batch (id tie-break)
        and return ``(decode_id, predicted_batch)``."""
        if self.engine is None:
            return (min(self.decode_ids), 0)
        d = min(self.decode_ids,
                key=lambda d: (self._predicted_batch(d, done_time), d))
        return (d, self._predicted_batch(d, done_time))

    def _finalize_admission(self, plan: Plan) -> Optional[Plan]:
        """Apply the SLO gates, then commit. ``None`` == rejected.

        TTFT is always gated. TBT is gated here only in the ``early``/``predict``
        modes (reject *before* prefill on a predicted violation); ``off`` defers the
        TBT decision to :meth:`admit_decode` at prefill completion (a late reject
        that wastes the prefill).
        """
        if plan.ttft > self.slo_ttft:
            return None
        if (self.tbt_enabled and self.early_rejection in ("early", "predict")
                and plan.pred_tbt > self.slo_tbt):
            return None
        return self._commit(plan)

    def admit_decode(self, plan: Plan, now: float) -> bool:
        """Enter an accepted request into its decode batch at prefill completion.

        Returns ``False`` when decode cannot honour the TBT SLO. In ``off`` mode
        this is the only TBT gate, so a rejection here means the prefill was already
        spent -- a *wasted* prefill (the cost of not predicting decode load).
        """
        if self.engine is None:
            return True
        if self.early_rejection == "off":
            pred = cost.decode_step_time(self.engine.occupancy(plan.decode) + 1)
            if pred > self.slo_tbt:
                return False  # late reject -> wasted prefill
        self.engine.admit(plan.request, plan.decode)
        return True

    # -- commit / completion --------------------------------------------- #
    def on_complete(self, plan: Plan) -> List[str]:
        """Publish the request's blocks on its prefill instance (read-through, K4).

        After prefill the instance holds KV for the whole prompt, so we admit every
        block key into its cache (evicting the coldest past capacity) and register
        the presence in the directory. Returns the evicted keys.
        """
        keys = list(plan.request.block_keys)
        cache = self.caches[plan.prefill]
        evicted = cache.admit(keys)
        for k in keys:
            self.index.notify_put(k, plan.prefill)
        for k in evicted:
            self.index.notify_delete(k, plan.prefill)
        return evicted

    def _commit(self, plan: Plan) -> Plan:
        """Reserve the prefill server for an accepted plan (decode is admitted later,
        at prefill completion, via :meth:`admit_decode`)."""
        self.busy_until[plan.prefill] = plan.done_time
        # A cache hit on the matched prefix counts as an access (LRU recency).
        matched = list(plan.request.block_keys[: plan.match_blocks])
        self.caches[plan.prefill].touch(matched)
        # ``predict`` mode tracks reserved-but-not-yet-decoding prefills so a later
        # request can foresee the decode load they will add. Prune stale entries
        # (their prefill has already completed) before recording this one.
        if self.early_rejection == "predict" and self.engine is not None:
            self._inflight = [e for e in self._inflight if e[0] >= self.sim.now]
            self._inflight.append(
                (plan.done_time, plan.decode, plan.request.output_tokens))
        return plan


class LoadBalanceScheduler(_Base):
    """Baseline: route to the least-loaded instance; local-only cache reuse."""

    def schedule(self, request: Request, now: float) -> Optional[Plan]:
        keys = list(request.block_keys)
        prompt = request.prompt_tokens
        pick = min(self.prefill_ids, key=lambda i: (self.busy_until[i], i))
        match = longest_prefix_run(keys, self.caches[pick].held())
        cached = min(match * self.B, prompt)
        uncached = prompt - cached
        qw, ttft, done = self._predict(pick, now, 0.0, prefill_time(uncached))
        d, pred_batch = self._select_decode(done)
        pred_tbt = cost.decode_step_time(pred_batch + 1) if self.engine else 0.0
        plan = Plan(request, pick, d, match, cached, uncached, None, 0,
                    qw, ttft, done, 0.0)
        plan.pred_tbt = pred_tbt
        plan.pred_batch = pred_batch
        return self._finalize_admission(plan)


class CacheAwareScheduler(_Base):
    """Cache-aware coordinator: global prefix-match routing under a balance threshold."""

    def __init__(self, *args, balance_threshold: float = 1.5,
                 replicate: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Pull a remote prefix only when it is > threshold x the candidate's local
        # match; otherwise prefer recompute (the kvcache balancing threshold).
        self.balance_threshold = balance_threshold
        # When False, never pull a remote prefix (always recompute a missing
        # prefix locally). Used to isolate replication's contribution in the demo.
        self.replicate = replicate

    def _best_remote(self, prefix_counts: Dict[str, int]):
        """Return ``(best_len, best_instance)`` across the cluster (id tie-break)."""
        if not prefix_counts:
            return 0, None
        best_len = max(prefix_counts.values())
        best_inst = min(i for i, n in prefix_counts.items() if n == best_len)
        return best_len, best_inst

    def schedule(self, request: Request, now: float) -> Optional[Plan]:
        keys = list(request.block_keys)
        prompt = request.prompt_tokens
        prefix_counts = self.index.instances_with_prefix(keys)
        best_len, best_inst = self._best_remote(prefix_counts)

        best: Optional[Plan] = None
        for inst in self.prefill_ids:
            local_len = prefix_counts.get(inst, 0)
            use_remote = (
                self.replicate
                and best_inst is not None
                and best_inst != inst
                and best_len > local_len * self.balance_threshold
            )
            if use_remote:
                gap_blocks = best_len - local_len
                xbytes = block_bytes(gap_blocks, self.B, self.bpt)
                xt = transfer_time(self.topo[best_inst], self.topo[inst], xbytes)
                cached = min(best_len * self.B, prompt)
                uncached = prompt - cached
                qw, ttft, done = self._predict(inst, now, xt, prefill_time(uncached))
                src, match, xb = best_inst, best_len, xbytes
            else:
                cached = min(local_len * self.B, prompt)
                uncached = prompt - cached
                qw, ttft, done = self._predict(inst, now, 0.0, prefill_time(uncached))
                src, match, xb = None, local_len, 0
            cand = Plan(request, inst, "", match, cached, uncached, src, xb,
                        qw, ttft, done, 0.0)
            if best is None or (cand.ttft, cand.prefill) < (best.ttft, best.prefill):
                best = cand

        assert best is not None
        best.decode, best.pred_batch = self._select_decode(best.done_time)
        best.pred_tbt = (
            cost.decode_step_time(best.pred_batch + 1) if self.engine else 0.0)
        return self._finalize_admission(best)
