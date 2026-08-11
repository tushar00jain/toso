"""Schedulers under test: ``LoadBalanceScheduler`` and ``CacheAwareScheduler``.

Both implement :class:`~proposed.coordinator.Coordinator`, whose two members are
the whole surface the data plane may touch -- and, because control runs in a
coordinator service rather than on the serving host, the whole surface that would go
on a wire. The *questions* are this application's, carried as values::

    await decide(Route(request))        -> Plan | None   # None == rejected (SLO)
    await decide(AdmitDecode(plan))     -> True | None   # may it enter a decode batch
    await decide(PrefillFinished(i, t)) -> float         # the corrected queue tail
    observe(ComputeBusy(i, until))                       # a decode step's actual end
    observe(DecodeState(i, finishes))                    # who is decoding, until when

Every one of those is about *compute*: where work runs, whether it may run, and what
the machines are doing. What becomes of the *data* -- which blocks the directory
should hold, and which stop existing -- is not asked here. Publishing is a fact the
serving host already has (the blocks past the reused prefix are the ones it just
computed), and retention is a store question, asked of this object's other half:
:meth:`_Base.evict`, which the volume reaches through the directory when it is
actually out of room.

Every argument and every return is a value. Nothing here takes a handle to a
data-plane object, and nothing here is a field the data plane reads -- see
"Control plane only" below.

Routing models the cache-aware coordinator's serialized mailbox: the real
directory read completes without suspending the loop, so the whole routing
decision runs atomically before the next event -- routing sees a consistent
directory snapshot (:class:`~kvcache_sim.control._view.PinnedKVView` makes that
snapshot explicit). Prefill cost is deterministic, so the *predicted* TTFT used
for routing equals the *actual* completion time.

* ``LoadBalanceScheduler`` (baseline, ~vLLM): route to the least-loaded instance;
  reuse only that instance's **local** cache; never pull a remote prefix.
* ``CacheAwareScheduler`` (cache-aware coordinator): route to minimize predicted
  TTFT using the **global** prefix-match directory, optionally pulling a remote
  prefix (whose *source* is chosen by
  :class:`~kvcache_sim.control._source.LongestPrefixPolicy`) under a balance
  threshold.

Control plane only
------------------
Nothing here executes: this module holds no client, no volume, no deployment and no
decode engine. It senses through a :class:`~kvcache_sim.control._view.KVView`,
returns decisions, and learns what actually happened from the facts the data plane
reports.

Everything the scheduler knows about the running cluster is therefore a *model*
corrected by observations, never a live read:

* the **prefill queue** (:attr:`_Base.busy_until`) is predicted -- routing reserves
  an instance until the TTFT it predicted -- and corrected by :class:`PrefillFinished`.
  On a **coupled** instance prefill and decode are one physical resource, so the data
  plane also mirrors each decode step back as :class:`ComputeBusy`; it is the data
  plane that decides whether coupling applies, because whether the two contend is a
  fact about the deployment, not about the policy;
* **decode occupancy** (:attr:`_Base._decode_finishes`) is a per-instance list of
  estimated finish times, replaced wholesale by :class:`DecodeState` whenever a batch
  changes. This used to be a
  handle: the data plane passed its ``DecodeEngine`` in and the scheduler called
  ``occupancy()`` on it mid-decision. Same numbers, but a pointer into another
  host, so it became a value the data plane pushes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from proposed import Coordinator, Endpoint, Policy, Selection

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, MachineProfile, Model,
    prefill_time,
)
from proposed import TransferCost

from ._view import KVView
from ._source import LongestPrefixPolicy
from .request import Request

__all__ = [
    "Route",
    "AdmitDecode",
    "PrefillFinished",
    "ComputeBusy",
    "DecodeState",
    "Plan",
    "LoadBalanceScheduler",
    "CacheAwareScheduler",
]


# -- what this application asks its coordinator, and tells it ---------------- #
# :class:`proposed.coordinator.Coordinator` declares two members, ``decide`` and
# ``observe``, and leaves the questions to the application. These are this
# application's: four demands it asks and two facts it reports. Values, so they
# cross a process boundary unchanged, and frozen, so an answer cannot be edited
# after it was given.


@dataclass(frozen=True)
class Route:
    """*Where should this request run?* Answered with a :class:`Plan`, or refused."""

    request: Request


@dataclass(frozen=True)
class AdmitDecode:
    """*May this request enter its decode batch?* Answered ``True`` / ``None``.

    Asked after prefill, so a refusal here has already cost the prefill -- which is
    what a surface that could answer *not yet* would avoid, and cannot today.
    """

    plan: "Plan"


@dataclass(frozen=True)
class PrefillFinished:
    """*Prefill really finished at this clock -- what is the queue tail now?*

    Both a report and a question, which is why it is a demand and not a fact:
    ``schedule`` reserved the instance until a *predicted* completion, the executed
    cost can differ, and the data plane needs the corrected tail back to apply it to
    a coupled instance's decode timeline.
    """

    inst: str
    now: float


@dataclass(frozen=True)
class ComputeBusy:
    """A decode step occupied a **coupled** instance's compute until ``until``."""

    inst: str
    until: float


@dataclass(frozen=True)
class DecodeState:
    """``inst``'s live decode batch, as one estimated finish time per request.

    Its length is the occupancy and its values answer "still decoding at ``t``?".
    Reported whenever the batch changes, which is the only time either answer moves.
    """

    inst: str
    finishes: Tuple[float, ...]


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


class _Base(Policy, Coordinator):
    """Shared state + prediction/commit helpers for both schedulers.

    **This class does both control-plane jobs, and says so in its bases.** It is a
    :class:`~proposed.policy.Policy`, so the run installs it in the directory and
    the controller consults it there (:meth:`select`); and it is a
    :class:`~proposed.coordinator.Coordinator`, so the run also fronts it with a
    :class:`~realsim.seams.coordinator_handle.CoordinatorHandle` and a serving host
    reaches it as its own service. The two are one object on purpose: the peer it
    prices a pull against is the peer it later tells the directory to serve, with
    nothing threaded through the data plane to carry that between them.

    Args:
        view: a :class:`~kvcache_sim.control._view.KVView` -- the only way this
            object sees the world.
        block_tokens: tokens per KV block.
        profile / model: the cost constants prediction is priced against.
        decode_pool / prefill_pool: instance subsets (default: all).
        slo_ttft / slo_tbt: admission gates.
        simulate_decode: whether the run models batched decode at all.
        max_batch: VRAM cap the data plane's decode batch will use; control needs
            it only to reason about admission.
        early_rejection: ``"off"`` | ``"early"`` | ``"predict"``.
        source_policy: the :class:`~proposed.policy.Policy` that ranks peers for a
            prefix pull (default :class:`~kvcache_sim.control._source.LongestPrefixPolicy`).
    """

    def __init__(
        self,
        *,
        block_tokens: int,
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
        self.B = block_tokens
        self.profile = profile
        self.model = model
        self.source_policy = (
            source_policy if source_policy is not None else LongestPrefixPolicy()
        )
        self._prefill_pool = prefill_pool
        self._decode_pool = decode_pool
        self.slo_ttft = slo_ttft
        self.slo_tbt = slo_tbt
        self.early_rejection = early_rejection
        self.max_batch = max_batch
        self.tbt_enabled = simulate_decode
        # Filled by attach(); a coordinator cannot model servers it has not been
        # told about, and the run knows them only once its stack exists.
        self.view: Any = None
        self.transfer_cost: Optional[TransferCost] = None
        self.topo: Dict[str, Endpoint] = {}
        self.ids: List[str] = []
        self.busy_until: Dict[str, float] = {}
        self.prefill_ids: List[str] = []
        self.decode_ids: List[str] = []
        self._decode_finishes: Dict[str, List[float]] = {}
        self._inflight: List[Tuple[float, str, int]] = []
        # Pulls this coordinator has already routed and priced, oldest first:
        # ``(requester, gap keys, chosen peer)``. Read back by :meth:`select` when
        # the directory asks who should serve those keys -- see its docstring.
        self._routed: List[Tuple[str, Tuple[str, ...], str]] = []

    # -- the stack hands over its ports ----------------------------------- #
    def attach(self, view, transfer_cost: TransferCost) -> None:
        """Receive the ports this coordinator senses and prices through.

        Two-phase on purpose: a capability writes ``MyControl(knobs)`` and the run
        hands it the stack, exactly as the controller hands an installed policy its
        view. It is what lets a control plane be an *object* a scenario declares
        rather than a factory the harness has to call at the right moment.

        The view is upgraded to a :class:`~kvcache_sim.control._view.KVView` here:
        prefix runs are this capability's notion, and deriving them is its job,
        not the store's.
        """
        self.view = KVView(view.directory, view.topology)
        # Priced through the protocol, never a simulator function: the scheduler
        # is written against an estimate, not against one cost model. A simulated
        # run passes one priced off the same model the transport charges, a
        # deployment its measured numbers.
        self.transfer_cost = transfer_cost
        self.topo = dict(view.topology)
        self.ids = sorted(self.topo)
        # PREDICTED prefill queue tail over ALL instances (the disaggregated-
        # prefill pool may be a subset; decode may be a disjoint or overlapping
        # subset). This is control's own model of the servers, corrected by the
        # data plane's observations -- see the module docstring.
        self.busy_until = {i: 0.0 for i in self.ids}
        self.prefill_ids = (
            sorted(self._prefill_pool) if self._prefill_pool else self.ids
        )
        self.decode_ids = (
            sorted(self._decode_pool) if self._decode_pool else self.ids
        )
        # Control's model of the decode side: instance -> one estimated finish
        # time per request decoding or queued there. Empty until the data plane
        # reports, which it does whenever a batch changes.
        self._decode_finishes = {i: [] for i in self.ids}
        # demand type -> the method that answers it, and fact type -> the method
        # that learns it. Bound here, so they resolve through this instance's MRO:
        # overriding an answer in a subclass is all a subclass has to do.
        self._answers = {
            Route: self._decide_route,
            AdmitDecode: self._decide_admit_decode,
            PrefillFinished: self._decide_prefill_finished,
        }
        self._facts = {
            ComputeBusy: self._observe_compute_busy,
            DecodeState: self._observe_decode_state,
        }

    # -- proposed.Coordinator: two members, one per kind of interaction ---- #
    async def decide(self, demand: Any) -> Optional[Any]:
        """:class:`~proposed.coordinator.Coordinator` -- answer ``demand``.

        Dispatch is the demand's own type, looked up in a table bound at
        construction, so the answer a subclass overrides *is* the answer that runs.
        ``functools.singledispatchmethod`` cannot do that: it captures the function
        registered on this class, and a subclass redefining one is silently ignored.
        """
        answer = self._answers.get(type(demand))
        if answer is None:
            raise TypeError(
                f"{type(self).__name__} does not answer {type(demand).__name__}: "
                f"this application's demands are "
                f"{', '.join(sorted(d.__name__ for d in self._answers))}"
            )
        return await answer(demand)

    def observe(self, fact: Any) -> None:
        """:class:`~proposed.coordinator.Coordinator` -- learn that ``fact`` happened."""
        learn = self._facts.get(type(fact))
        if learn is None:
            raise TypeError(
                f"{type(self).__name__} is not told {type(fact).__name__}: this "
                f"application's facts are "
                f"{', '.join(sorted(f.__name__ for f in self._facts))}"
            )
        learn(fact)

    async def _decide_route(self, demand: Route) -> Optional["Plan"]:
        """Where should this request run? Answered by whichever scheduler this is."""
        raise NotImplementedError(
            f"{type(self).__name__} answers no Route: a scheduler decides where a "
            f"request runs"
        )

    async def _decide_admit_decode(self, demand: AdmitDecode) -> Optional[bool]:
        """May this accepted request enter its decode batch now?

        ``None`` -- the refusal the surface defines -- when decode cannot honour the
        TBT SLO. In ``off`` mode this is the only TBT gate, so a refusal here means
        the prefill was already spent: a *wasted* prefill. The data plane performs
        (or skips) the admission; this only decides.
        """
        plan = demand.plan
        if not self.tbt_enabled:
            return True
        if self.early_rejection == "off":
            pred = decode_step_time(
                self._occupancy(plan.decode) + 1, self.profile, self.model
            )
            if pred > self.slo_tbt:
                return None  # late reject -> wasted prefill
        return True

    async def _decide_prefill_finished(self, demand: PrefillFinished) -> float:
        """Correct the predicted queue with the clock the real ops reached.

        Routing reserved this instance until the *predicted* ``done_time``; the
        executed cost can differ (the pull may find fewer blocks still resident, may
        be served by a different peer than the one priced, or may be slowed by
        contention), and only now is the true completion time known.

        Answers with the corrected queue tail. The data plane needs it on a coupled
        instance -- prefill just occupied the timeline decode steps run on -- and
        answering is what keeps that a reply rather than a field it reads.
        """
        if demand.now > self.busy_until[demand.inst]:
            self.busy_until[demand.inst] = demand.now
        return self.busy_until[demand.inst]

    def _observe_compute_busy(self, fact: ComputeBusy) -> None:
        """A decode step on a **coupled** instance occupied its compute.

        Only the data plane knows whether prefill and decode share a timeline, so
        only it reports this; when they are disaggregated it never does, and the
        predicted prefill queue is untouched by decode.
        """
        self.busy_until[fact.inst] = fact.until

    def _observe_decode_state(self, fact: DecodeState) -> None:
        """Replace control's model of ``inst``'s decode batch."""
        self._decode_finishes[fact.inst] = list(fact.finishes)

    # -- the store's questions, answered from what we already decided ----- #


    async def select(
        self, view: Any, keys: Sequence[str], requester: str
    ) -> Selection:
        """:class:`~proposed.policy.Policy` -- who serves ``keys`` for ``requester``.

        This coordinator is *also* the policy the directory consults, which is the
        whole point: it priced this pull against a specific peer's locality tier
        before asking for the bytes, so being asked again is a chance to say what
        it already decided rather than to decide twice. Re-deriving would not even
        agree -- routing ranks over a request's whole block chain, while the fetch
        names only the gap -- and a directory answer that named a different holder
        would charge a cross-node read for a same-node prediction.

        Matching is by requester plus a *superset* of the keys, because a block may
        have been evicted between routing and fetching, and the read-through asks
        only for what is still present. Oldest routed pull first, so two requests
        in flight to one instance resolve in a fixed order.
        """
        wanted = set(keys)
        for i, (inst, gap, peer) in enumerate(self._routed):
            if inst == requester and wanted <= set(gap):
                del self._routed[i]
                return Selection.of([peer])
        # Nothing routed for this caller: whoever is asking chose nothing, so the
        # ranking is the honest answer (and an empty one lets the directory speak).
        return await self.source_policy.select(self.view or view, list(keys), requester)

    # -- reading the model the facts above maintain ----------------------- #
    def _occupancy(self, inst: str) -> int:
        """Requests currently decoding or queued on ``inst``."""
        return len(self._decode_finishes.get(inst, ()))

    def _predict_occupancy(self, inst: str, at_t: float) -> int:
        """How many of those are estimated to still be decoding at ``at_t``."""
        return sum(1 for f in self._decode_finishes.get(inst, ()) if f > at_t)

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
        if not self.tbt_enabled:
            return 0
        if self.early_rejection == "predict":
            n = self._predict_occupancy(d, done_time)
            for pd_done, dec_id, out in self._inflight:
                if (
                    dec_id == d
                    and pd_done <= done_time
                    and pd_done + max(0, out - 1) * decode_step_time(1, self.profile, self.model)
                    > done_time
                ):
                    n += 1
            return n
        return self._occupancy(d)

    def _select_decode(self, done_time: float) -> Tuple[str, int]:
        """Pick the decode instance with the smallest predicted batch (id tie-break)."""
        if not self.tbt_enabled:
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

    # -- commit ----------------------------------------------------------- #
    def _commit(self, plan: Plan) -> Plan:
        """Reserve the prefill server for an accepted plan (decode admitted later)."""
        # Remember the peer this pull was priced against, for the moment the
        # directory asks (see :meth:`select`). Recorded at commit, so a plan that
        # was considered and dropped leaves nothing behind.
        if plan.reuse_source is not None and plan.pull_keys:
            self._routed.append(
                (plan.prefill, tuple(plan.pull_keys), plan.reuse_source)
            )
        self.busy_until[plan.prefill] = plan.done_time
        if self.early_rejection == "predict" and self.tbt_enabled:
            now = self.view.now()
            self._inflight = [e for e in self._inflight if e[0] >= now]
            self._inflight.append(
                (plan.done_time, plan.decode, plan.request.output_tokens)
            )
        return plan


class LoadBalanceScheduler(_Base):
    """Baseline: route to the least-loaded instance; local-only cache reuse."""

    async def _decide_route(self, demand: Route) -> Optional[Plan]:
        request = demand.request
        now = self.view.now()
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

    async def _decide_route(self, demand: Route) -> Optional[Plan]:
        request = demand.request
        now = self.view.now()
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
                xt = self.transfer_cost.get_time(src_inst, inst, xbytes)
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
