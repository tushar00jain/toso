"""Schedulers under test: ``LoadBalanceScheduler`` and ``CacheAwareScheduler``.

Both implement :class:`~proposed.coordinator.Coordinator`, whose two members --
``decide`` and ``observe`` -- are the whole surface the data plane may touch. This
application's demands and facts::

    await decide(Route(request))        -> Plan | None   # None == rejected (SLO)
    await decide(AdmitDecode(plan))     -> True | None   # may it enter a decode batch
    await decide(PrefillFinished(i, t)) -> float         # the corrected queue tail
    observe(ComputeBusy(i, until))                       # a decode step's actual end
    observe(DecodeState(i, finishes))                    # who is decoding, until when

All of them are about *compute*: where work runs, whether it may run, and what the
machines are doing. Data placement is not asked here -- the serving host already
knows which blocks it computed, and a volume that runs out of room drops its own
coldest keys and tells the directory afterwards (:mod:`realsim.seams._retention`).
Every argument and return is a value; nothing here holds a handle to a data-plane
object.

* ``LoadBalanceScheduler`` (baseline, ~vLLM): route to the least-loaded instance;
  reuse only that instance's **local** cache; never pull a remote prefix.
* ``CacheAwareScheduler``: route to minimize predicted TTFT using the **global**
  prefix-match directory, optionally pulling a remote prefix (whose *source* is
  chosen by :class:`~kvcache_sim.control._source.LongestPrefixPolicy`) under a
  balance threshold.

A routing decision runs atomically: the directory read completes without suspending
the loop, so the whole decision sees one consistent snapshot
(:class:`~kvcache_sim.control._view.PinnedKVView` makes that explicit).

Control's model of the cluster
------------------------------
Nothing here executes -- no client, no volume, no deployment, no decode engine. What
the scheduler knows about the running cluster is a *model* corrected by
observations, never a live read:

* the **prefill queue** (:attr:`_Base.busy_until`) is predicted -- routing reserves
  an instance until the TTFT it predicted -- and corrected by
  :class:`PrefillFinished`. The data plane measures the real wait independently, and
  the two are recorded side by side
  (:attr:`kvcache_sim.report.metrics.RequestResult.queue_wait` against
  ``predicted_queue_wait``). They diverge by construction:
  :meth:`_Base._candidate` prices a candidate as ``queue -> transfer -> prefill``
  and reserves the instance for all three, so a remote pull is charged to a device
  that is idle while the fabric works. On a **coupled** instance prefill and decode
  share one accelerator, so the data plane also mirrors each decode step back as
  :class:`ComputeBusy`;
* **decode occupancy** (:attr:`_Base._decode_finishes`) is a per-instance list of
  estimated finish times, replaced wholesale by :class:`DecodeState` whenever a
  batch changes.

The TTFT the metrics record is this prediction rather than a measurement -- a
deliberate choice, spelled out in the README. Prefill cost is deterministic, so on
the default path the prediction is also the actual completion time; a peer that
evicted the planned blocks, a pull served by a volume other than the one priced, or
``contention`` can each move the executed cost off it.
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

from ._pending import Reservations, RoutedPulls
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
# Three demands and two facts. Frozen values, so they cross a process boundary
# unchanged and cannot be edited after they are handed over.


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

    The only thing that tells this coordinator its model of an instance's prefill
    queue was wrong. ``now`` is an independent measurement (the host's accelerator
    serialises its own passes) and is routinely earlier than the ``done_time``
    routing reserved.

    A demand rather than a fact so the caller can rely on the *await* for ordering:
    the decode admission it asks next must be decided by a coordinator that has
    already recorded this completion.
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
    Reported whenever the batch changes.
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

    @property
    def local_blocks(self) -> int:
        """Blocks the prefill host already held: the match, minus what it pulls.

        Derived rather than a field: the data plane needs it three times over (the
        reuse to report, the suffix to publish, the prefix to fall back on when a
        planned pull turns out to be gone) and all three have to agree.
        """
        return self.match_blocks - len(self.pull_keys)


class _Base(Policy, Coordinator):
    """Shared state + prediction/commit helpers for both schedulers.

    Both control-plane jobs in one object. As a :class:`~proposed.policy.Policy` the
    run installs it in the directory and the controller consults it there
    (:meth:`select`); as a :class:`~proposed.coordinator.Coordinator` a serving host
    reaches it as a service. One object on purpose: the peer it prices a pull
    against is the peer it later tells the directory to serve, with nothing threaded
    through the data plane to carry that between them.

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
        # Filled by attach(): the run knows its servers only once its stack exists.
        self.view: Any = None
        self.transfer_cost: Optional[TransferCost] = None
        self.topo: Dict[str, Endpoint] = {}
        self.ids: List[str] = []
        self.busy_until: Dict[str, float] = {}
        self.prefill_ids: List[str] = []
        self.decode_ids: List[str] = []
        self._decode_finishes: Dict[str, List[float]] = {}
        # Decided but not yet carried out: prefills promised, and pulls priced
        # against a peer. Both self-expire (:mod:`kvcache_sim.control._pending`).
        self._reserved = Reservations()
        self._routed = RoutedPulls()

    # -- the stack hands over its ports ----------------------------------- #
    def attach(self, view, transfer_cost: TransferCost) -> None:
        """Receive the ports this coordinator senses and prices through.

        Two-phase so a scenario can declare a control plane as an object
        (``MyControl(knobs)``) and let the run hand it the stack afterwards.

        The view is upgraded to a :class:`~kvcache_sim.control._view.KVView` here:
        prefix runs are this capability's notion, not the store's.
        """
        self.view = KVView(view.directory, view.topology)
        # Priced through the protocol rather than a simulator function: a simulated
        # run passes one priced off the same model the transport charges, a
        # deployment its measured numbers.
        self.transfer_cost = transfer_cost
        self.topo = dict(view.topology)
        self.ids = sorted(self.topo)
        # PREDICTED prefill queue tail over ALL instances (the prefill and decode
        # pools may each be a subset). Corrected by the data plane's observations.
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
        # demand/fact type -> the bound method that handles it, so dispatch
        # resolves through this instance's MRO and a subclass need only override.
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

        Dispatch is on the demand's type, through the table bound in :meth:`attach`.
        (Not ``functools.singledispatchmethod``: it captures the function registered
        on this class, so a subclass redefining one is silently ignored.)
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

        ``None`` when decode cannot honour the TBT SLO. In ``off`` mode this is the
        only TBT gate, so a refusal here means the prefill was already spent: a
        *wasted* prefill.
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
        executed time differs when the pull finds fewer blocks resident, is served
        by a different peer, is slowed by contention, or when the prediction's own
        arithmetic about the wait was simply wrong.

        Raises the tail and never lowers it. An early completion leaves the instance
        looking busier than it is until the next request is routed against it,
        whereas lowering on one report would under-count the prefills this
        coordinator has promised and not yet seen finish. Answers with the tail.
        """
        if demand.now > self.busy_until[demand.inst]:
            self.busy_until[demand.inst] = demand.now
        return self.busy_until[demand.inst]

    def _observe_compute_busy(self, fact: ComputeBusy) -> None:
        """A decode step on a **coupled** instance occupied its compute.

        Only the data plane knows whether prefill and decode share a timeline. A
        disaggregated host never reports this, so decode never touches its
        predicted prefill queue.
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

        This coordinator priced the pull against a specific peer's locality tier
        before asking for the bytes, so it answers with that decision rather than
        deciding twice: re-deriving would not even agree (routing ranks over the
        request's whole block chain, the fetch names only the gap), and naming a
        different holder would charge a cross-node read for a same-node prediction.

        Falls through to the ranking when this caller has no routed pull.
        """
        peer = self._routed.claim(requester, keys)
        if peer is not None:
            return Selection.of([peer])
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

    def _candidate(
        self,
        request: Request,
        inst: str,
        now: float,
        *,
        match: int,
        source: Optional[str] = None,
        pull_keys: Sequence[str] = (),
    ) -> Plan:
        """Price prefilling ``request`` on ``inst`` reusing ``match`` blocks.

        What the reused prefix saves, what the suffix costs to compute, what pulling
        ``pull_keys`` from ``source`` costs, and where all of that lands in
        ``inst``'s predicted queue. Which candidate wins is the caller's, and is the
        only part the two schedulers disagree on.

        Reserves nothing and mutates nothing, so a losing candidate leaves no trace
        (:meth:`_commit` records a decision actually taken).
        """
        cached = min(match * self.B, request.prompt_tokens)
        uncached = request.prompt_tokens - cached
        prefill_t = prefill_time(uncached, self.profile, self.model)
        if source is not None and pull_keys:
            xbytes = self.model.block_bytes(len(pull_keys), self.B)
            # Same cost model the transport charges, so this prediction equals what
            # the real pull will cost.
            transfer_t = self.transfer_cost.get_time(source, inst, xbytes)
        else:
            source, xbytes, transfer_t = None, 0, 0.0
        queue_wait, ttft, done = self._predict(inst, now, transfer_t, prefill_t)
        plan = Plan(
            request, inst, "", match, cached, uncached, source, xbytes,
            queue_wait, ttft, done, 0.0,
        )
        plan.prefill_t = prefill_t
        plan.transfer_t = transfer_t
        plan.pull_keys = list(pull_keys)
        return plan

    # -- decode-side TBT prediction / admission -------------------------- #
    def _predicted_batch(self, d: str, done_time: float) -> int:
        """Predicted decode batch size on ``d`` seen by a request admitted at
        ``done_time`` (its prefill completion). Drives TBT prediction."""
        if not self.tbt_enabled:
            return 0
        if self.early_rejection == "predict":
            n = self._predict_occupancy(d, done_time)
            # Requests whose prefill has not landed are invisible to the observed
            # decode state; the outstanding reservations stand in for them.
            for res in self._reserved.pending(self.view.now()):
                if (
                    res.decode_id == d
                    and res.prefill_done <= done_time
                    and res.prefill_done
                    + max(0, res.output_tokens - 1)
                    * decode_step_time(1, self.profile, self.model)
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

    def _admit(self, plan: Plan) -> Optional[Plan]:
        """Give a won candidate its decode instance, then apply the gates.

        The decode side is chosen once, against the winning candidate's predicted
        prefill completion -- not per candidate in the loop.
        """
        plan.decode, plan.pred_batch = self._select_decode(plan.done_time)
        plan.pred_tbt = (
            decode_step_time(plan.pred_batch + 1, self.profile, self.model)
            if self.tbt_enabled
            else 0.0
        )
        return self._finalize_admission(plan)

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
        """Reserve the prefill server for an accepted plan (decode admitted later).

        Records only; nothing is swept or expired here -- each record expires when
        it is read (:mod:`kvcache_sim.control._pending`).
        """
        # The peer this pull was priced against, for when the directory asks (see
        # :meth:`select`). Recorded at commit, so a dropped plan leaves nothing.
        if plan.reuse_source is not None and plan.pull_keys:
            self._routed.route(plan.prefill, plan.pull_keys, plan.reuse_source)
        self.busy_until[plan.prefill] = plan.done_time
        if self.early_rejection == "predict" and self.tbt_enabled:
            self._reserved.reserve(
                plan.done_time, plan.decode, plan.request.output_tokens
            )
        return plan


class LoadBalanceScheduler(_Base):
    """Baseline: route to the least-loaded instance; local-only cache reuse."""

    async def _decide_route(self, demand: Route) -> Optional[Plan]:
        request = demand.request
        # Per-instance prefix presence from the real directory. The baseline reuses
        # only the instance it routes to, so it never asks the source policy.
        counts = await self.view.prefix_lengths(list(request.block_keys))
        pick = min(self.prefill_ids, key=lambda i: (self.busy_until[i], i))
        return self._admit(
            self._candidate(
                request, pick, self.view.now(), match=counts.get(pick, 0)
            )
        )


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
        # One directory snapshot for the whole decision: the candidate loop's
        # local matches and every source query below read the same state.
        view = self.view.pin(keys)
        prefix_counts = await view.prefix_lengths()

        best: Optional[Plan] = None
        for inst in self.prefill_ids:
            local_len = prefix_counts.get(inst, 0)
            # The one store question in this decision: which peer would serve the
            # gap? Pull-vs-recompute and placement are ours.
            ranked = await self.source_policy.select(view, keys, inst)
            src_inst = ranked.sources[0] if ranked.sources else None
            src_len = prefix_counts.get(src_inst, 0) if src_inst is not None else 0
            if (
                self.replicate
                and src_inst is not None
                and src_inst != inst
                and src_len > local_len * self.balance_threshold
            ):
                # Worth pulling: reuse the peer's whole prefix and fetch the gap.
                cand = self._candidate(
                    request, inst, now, match=src_len,
                    source=src_inst, pull_keys=keys[local_len:src_len],
                )
            else:
                # Not worth it: reuse only what is already here and recompute.
                cand = self._candidate(request, inst, now, match=local_len)
            if best is None or (cand.ttft, cand.prefill) < (best.ttft, best.prefill):
                best = cand

        assert best is not None
        return self._admit(best)
