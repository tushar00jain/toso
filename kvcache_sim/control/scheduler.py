"""Schedulers under test: ``LoadBalanceScheduler`` and ``CacheAwareScheduler``.

Both share an interface so scenarios/tests can swap them::

    await schedule(request, now) -> Plan | None   # None == rejected (SLO/overload)
    complete(plan)               -> Completion    # what to publish / evict
    observe_prefill_done(inst, now)               # the clock the data plane reached
    observe_compute_busy(inst, until)             # a decode step's actual end

``schedule`` models the cache-aware coordinator's serialized mailbox: the real
directory read completes without suspending the loop, so the whole routing
decision runs atomically before the next event -- routing sees a consistent
directory snapshot (:class:`~kvcache_sim.control.view.PinnedKVView` makes that
snapshot explicit). Prefill cost is deterministic, so the *predicted* TTFT used
for routing equals the *actual* completion time.

* ``LoadBalanceScheduler`` (baseline, ~vLLM): route to the least-loaded instance;
  reuse only that instance's **local** cache; never pull a remote prefix.
* ``CacheAwareScheduler`` (cache-aware coordinator): route to minimize predicted
  TTFT using the **global** prefix-match directory, optionally pulling a remote
  prefix (whose *source* is chosen by
  :class:`~kvcache_sim.control.source.LongestPrefixPolicy`) under a balance
  threshold.

Control plane only
------------------
Nothing here executes: this module holds no client, no volume, no mesh and no
decode engine. It senses through a :class:`~kvcache_sim.control.view.KVView`,
returns decisions, and learns what actually happened through two ``observe_*``
calls the data plane makes.

That second point is what replaced the old ``busy_until`` / ``compute_busy``
alias. Control keeps a *predicted* prefill queue (:attr:`_Base.busy_until`), which
is a model: ``schedule`` reserves an instance until the TTFT it predicted. The
data plane keeps the *actual* compute timeline. On a **coupled** instance the two
are the same physical resource, so the data plane mirrors its decode steps back
here via :meth:`_Base.observe_compute_busy` -- an observation, not a shared dict,
and it is the data plane that decides whether coupling applies, because whether
prefill and decode contend is a fact about the deployment, not about the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from sim_common.cost_model import DEFAULT_PROFILE, get_time, MachineProfile
from sim_common.topology import Endpoint

from domain.llm import DEFAULT_MODEL, decode_step_time, Model, prefill_time

from .cache import LRUCache
from .source import LongestPrefixPolicy
from ..workload.request import Request


class DecodeLoad(Protocol):
    """How busy the decode side is -- an observation, not a handle to it.

    The data plane's decode engine satisfies this; control never learns anything
    else about it (it cannot admit, step or drain through this interface).
    """

    def occupancy(self, inst: str) -> int:
        """Requests currently decoding or queued on ``inst``."""

    def predict_occupancy(self, inst: str, at_t: float) -> int:
        """How many of those are estimated to still be decoding at ``at_t``."""


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
    prefill_t: float = 0.0       # prefill compute duration
    transfer_t: float = 0.0      # predicted remote-pull fetch duration
    pull_keys: List[str] = field(default_factory=list)  # gap blocks to fetch
    pred_tbt: float = 0.0        # predicted time-between-tokens at admission
    pred_batch: int = 0          # predicted decode batch size at admission


@dataclass(frozen=True)
class Completion:
    """What the data plane must do once a request's prefill has run.

    Returned by :meth:`_Base.complete`; the cache bookkeeping behind it has
    already happened, because *which* blocks to keep is a control decision. The
    store calls that make it true are the data plane's.
    """

    instance: str
    publish: List[str]   # newly-materialized blocks to register on ``instance``
    evict: List[str]     # blocks LRU dropped, to remove from the directory


class _Base:
    """Shared state + prediction/commit helpers for both schedulers.

    Args:
        view: a :class:`~kvcache_sim.control.view.KVView` -- the only way this
            object sees the world.
        block_tokens: tokens per KV block.
        capacity: per-instance cache capacity in blocks (``None`` = unbounded).
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: admission gates.
        simulate_decode: whether the run models batched decode at all.
        max_batch: VRAM cap the data plane's decode batch will use; control needs
            it only to reason about admission.
        early_rejection: ``"off"`` | ``"early"`` | ``"predict"``.
        source_policy: the :class:`~proposed.policy.Policy` that ranks peers for a
            prefix pull (default :class:`~kvcache_sim.control.source.LongestPrefixPolicy`).
    """

    def __init__(
        self,
        view,
        *,
        block_tokens: int,
        capacity: Optional[int] = None,
        profile: MachineProfile = DEFAULT_PROFILE,
        model: Model = DEFAULT_MODEL,
        decode_pool: Optional[List[str]] = None,
        prefill_pool: Optional[List[str]] = None,
        slo_ttft: float = float("inf"),
        slo_tbt: float = float("inf"),
        simulate_decode: bool = False,
        max_batch: int = 8,
        early_rejection: str = "off",
        source_policy: Optional[Any] = None,
    ) -> None:
        self.view = view
        self.topo: Dict[str, Endpoint] = view.topology
        self.ids: List[str] = sorted(self.topo)
        self.B = block_tokens
        self.profile = profile
        self.model = model
        self.source_policy = (
            source_policy if source_policy is not None else LongestPrefixPolicy()
        )
        self.caches: Dict[str, LRUCache] = {i: LRUCache(capacity) for i in self.ids}
        # PREDICTED prefill queue tail over ALL instances (the disaggregated-
        # prefill pool may be a subset; decode may be a disjoint or overlapping
        # subset). This is control's own model of the servers, corrected by the
        # data plane's observations -- see the module docstring.
        self.busy_until: Dict[str, float] = {i: 0.0 for i in self.ids}
        self.prefill_ids: List[str] = (
            sorted(prefill_pool) if prefill_pool else self.ids
        )
        self.decode_ids: List[str] = sorted(decode_pool) if decode_pool else self.ids
        self.slo_ttft = slo_ttft
        self.slo_tbt = slo_tbt
        self.early_rejection = early_rejection
        self.max_batch = max_batch
        self.tbt_enabled = simulate_decode
        # Attached by the data plane when decode is modelled (read-only: see
        # DecodeLoad). ``None`` until then, and always ``None`` when decode is
        # not simulated.
        self.decode_load: Optional[DecodeLoad] = None
        self._inflight: List[Tuple[float, str, int]] = []

    # -- what the data plane attaches / reports back ---------------------- #
    def attach_decode_load(self, load: DecodeLoad) -> None:
        """Let control observe decode occupancy (it can only read it)."""
        self.decode_load = load

    def observe_prefill_done(self, inst: str, now: float) -> None:
        """Correct the predicted queue with the clock the real ops reached.

        ``schedule`` reserved this instance until the *predicted* ``done_time``;
        the executed cost can differ (the pull may find fewer blocks still
        resident, may be served by a different peer than the one priced, or may
        be slowed by contention), and only now is the true completion time known.
        """
        if now > self.busy_until[inst]:
            self.busy_until[inst] = now

    def observe_compute_busy(self, inst: str, until: float) -> None:
        """A decode step on a **coupled** instance occupied its compute to ``until``.

        Only the data plane knows whether prefill and decode share a timeline, so
        only it makes this call; when they are disaggregated it never does, and
        the predicted prefill queue is untouched by decode.
        """
        self.busy_until[inst] = until

    # -- prediction (no mutation) ---------------------------------------- #
    def _predict(self, inst: str, now: float, transfer_t: float, prefill_t: float):
        """Return ``(queue_wait, ttft, done_time)`` without reserving the server."""
        avail = max(now, self.busy_until[inst])
        queue_wait = avail - now
        done = avail + transfer_t + prefill_t
        return queue_wait, done - now, done

    # -- decode-side TBT prediction / admission -------------------------- #
    def _predicted_batch(self, d: str, done_time: float) -> int:
        """Predicted decode batch size on ``d`` seen by a request admitted at
        ``done_time`` (its prefill completion). Drives TBT prediction."""
        if not self.tbt_enabled or self.decode_load is None:
            return 0
        if self.early_rejection == "predict":
            n = self.decode_load.predict_occupancy(d, done_time)
            for pd_done, dec_id, out in self._inflight:
                if (
                    dec_id == d
                    and pd_done <= done_time
                    and pd_done + max(0, out - 1) * decode_step_time(1, self.profile, self.model)
                    > done_time
                ):
                    n += 1
            return n
        return self.decode_load.occupancy(d)

    def _select_decode(self, done_time: float) -> Tuple[str, int]:
        """Pick the decode instance with the smallest predicted batch (id tie-break)."""
        if not self.tbt_enabled or self.decode_load is None:
            return (min(self.decode_ids), 0)
        d = min(
            self.decode_ids, key=lambda d: (self._predicted_batch(d, done_time), d)
        )
        return (d, self._predicted_batch(d, done_time))

    def _finalize_admission(self, plan: Plan) -> Optional[Plan]:
        """Apply the SLO gates, then commit. ``None`` == rejected."""
        if plan.ttft > self.slo_ttft:
            return None
        if (
            self.tbt_enabled
            and self.early_rejection in ("early", "predict")
            and plan.pred_tbt > self.slo_tbt
        ):
            return None
        return self._commit(plan)

    def decode_admission(self, plan: Plan) -> bool:
        """May this accepted request enter its decode batch now?

        ``False`` when decode cannot honour the TBT SLO. In ``off`` mode this is
        the only TBT gate, so a refusal here means the prefill was already spent
        -- a *wasted* prefill. The data plane performs (or skips) the admission;
        this only decides.
        """
        if not self.tbt_enabled or self.decode_load is None:
            return True
        if self.early_rejection == "off":
            pred = decode_step_time(
                self.decode_load.occupancy(plan.decode) + 1, self.profile, self.model
            )
            if pred > self.slo_tbt:
                return False  # late reject -> wasted prefill
        return True

    # -- commit / completion --------------------------------------------- #
    def complete(self, plan: Plan) -> Completion:
        """Admit the request's blocks into its prefill instance's cache.

        After prefill the instance holds KV for the whole prompt, so every block
        key is admitted (evicting the coldest past capacity). Returns the blocks
        the data plane must register in the real directory (the *new* ones -- the
        ones already cached locally are not re-put) and the ones it must remove.
        """
        keys = list(plan.request.block_keys)
        cache = self.caches[plan.prefill]
        already = cache.held()
        new_keys = [k for k in keys if k not in already]
        evicted = cache.admit(keys)
        return Completion(instance=plan.prefill, publish=new_keys, evict=evicted)

    def _commit(self, plan: Plan) -> Plan:
        """Reserve the prefill server for an accepted plan (decode admitted later)."""
        self.busy_until[plan.prefill] = plan.done_time
        matched = list(plan.request.block_keys[: plan.match_blocks])
        self.caches[plan.prefill].touch(matched)
        if self.early_rejection == "predict" and self.tbt_enabled:
            now = self.view.now()
            self._inflight = [e for e in self._inflight if e[0] >= now]
            self._inflight.append(
                (plan.done_time, plan.decode, plan.request.output_tokens)
            )
        return plan


class LoadBalanceScheduler(_Base):
    """Baseline: route to the least-loaded instance; local-only cache reuse."""

    async def schedule(self, request: Request, now: float) -> Optional[Plan]:
        keys = list(request.block_keys)
        prompt = request.prompt_tokens
        # Consult the real directory for per-instance prefix presence; the
        # baseline only reuses the instance it routes to (local-only cache), so
        # it never asks the source policy anything.
        counts = await self.view.prefix_lengths(keys)
        pick = min(self.prefill_ids, key=lambda i: (self.busy_until[i], i))
        match = counts.get(pick, 0)
        cached = min(match * self.B, prompt)
        uncached = prompt - cached
        pt = prefill_time(uncached, self.profile, self.model)
        qw, ttft, done = self._predict(pick, now, 0.0, pt)
        d, pred_batch = self._select_decode(done)
        pred_tbt = (
            decode_step_time(pred_batch + 1, self.profile, self.model)
            if self.tbt_enabled
            else 0.0
        )
        plan = Plan(
            request, pick, d, match, cached, uncached, None, 0, qw, ttft, done, 0.0
        )
        plan.prefill_t = pt
        plan.pred_tbt = pred_tbt
        plan.pred_batch = pred_batch
        return self._finalize_admission(plan)


class CacheAwareScheduler(_Base):
    """Cache-aware coordinator: global prefix-match routing under a balance threshold."""

    def __init__(self, *args, balance_threshold: float = 1.5, replicate: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Pull a remote prefix only when it is > threshold x the candidate's local
        # match; otherwise prefer recompute (the balancing threshold).
        self.balance_threshold = balance_threshold
        # When False, never pull a remote prefix (always recompute a missing
        # prefix locally). Used to isolate replication's contribution in the demo.
        self.replicate = replicate

    async def schedule(self, request: Request, now: float) -> Optional[Plan]:
        keys = list(request.block_keys)
        prompt = request.prompt_tokens
        # One directory snapshot for the whole decision: the candidate loop's
        # local matches and every source query below read the same state.
        view = self.view.pin(keys)
        prefix_counts = await view.prefix_lengths()

        best: Optional[Plan] = None
        for inst in self.prefill_ids:
            local_len = prefix_counts.get(inst, 0)
            # The one store question in this decision: which peer would serve the
            # gap? Everything after it -- pull or recompute, where to prefill,
            # whether to admit at all -- is ours.
            ranked = await self.source_policy.select(view, keys, inst)
            src_inst = ranked.sources[0] if ranked.sources else None
            src_len = prefix_counts.get(src_inst, 0) if src_inst is not None else 0
            use_remote = (
                self.replicate
                and src_inst is not None
                and src_inst != inst
                and src_len > local_len * self.balance_threshold
            )
            if use_remote:
                gap_blocks = src_len - local_len
                pull_keys = keys[local_len:src_len]
                xbytes = self.model.block_bytes(gap_blocks, self.B)
                # The one definition the transport also charges, so this
                # prediction equals the time the real pull will cost.
                xt = get_time(
                    self.topo[src_inst], self.topo[inst], xbytes, self.profile
                )
                cached = min(src_len * self.B, prompt)
                uncached = prompt - cached
                pt = prefill_time(uncached, self.profile, self.model)
                qw, ttft, done = self._predict(inst, now, xt, pt)
                src, match, xb = src_inst, src_len, xbytes
            else:
                pull_keys = []
                cached = min(local_len * self.B, prompt)
                uncached = prompt - cached
                pt = prefill_time(uncached, self.profile, self.model)
                xt = 0.0
                qw, ttft, done = self._predict(inst, now, 0.0, pt)
                src, match, xb = None, local_len, 0
            cand = Plan(
                request, inst, "", match, cached, uncached, src, xb, qw, ttft,
                done, 0.0,
            )
            cand.prefill_t = pt
            cand.transfer_t = xt
            cand.pull_keys = list(pull_keys)
            if best is None or (cand.ttft, cand.prefill) < (best.ttft, best.prefill):
                best = cand

        assert best is not None
        best.decode, best.pred_batch = self._select_decode(best.done_time)
        best.pred_tbt = (
            decode_step_time(best.pred_batch + 1, self.profile, self.model)
            if self.tbt_enabled
            else 0.0
        )
        return self._finalize_admission(best)
