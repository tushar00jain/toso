"""The serving loop: turning one routing decision into real store calls.

:class:`ServingHost` is one serving instance: its cache, its decode batch, its
compute. A deployment runs one per host, and a host reaches exactly two things --
the store and its control plane. Never another host.

How does a request reach the host that should serve it?
-------------------------------------------------------
Which instance holds the longest reusable prefix is a cluster-wide fact, so the
host a request lands on is rarely the host that should serve it.
:meth:`ServingHost.prefill` asks control, and an answer naming another host comes
back as an address rather than a call to it::

    client -> A          "serve this" (a prompt)
    A      -> client     "not here: B"   (a reroute; A ran nothing, and booked the
                                          request onto B as it answered)
    client -> B          B asks, is told what A was told, and prefills
    B      -> client     the first token, and "decode is C"
    client -> C          C fetches that KV back out of the store and decodes
    C      -> client     the remaining tokens, after the last one is emitted

A request is passed on **once**. The ask that moves it books it
(:class:`~kvcache_sim.control._sensor.Committed`) and records where it sent it, so
the host it named is told the same answer and prices nothing. A second pricing would read the cluster
that booking has already moved, find the host just chosen busier than it was, and
be entitled to send the request on again -- hop after hop, converging on nothing.

TTFT is the time to the first token, sampled from the prefill's last position, so
:meth:`prefill` answers with it and :meth:`decode` answers with the rest. The
client sees both halves and is the only participant that can time the request.

Consequences of answering with an address:

* a host's whole outward surface is the store's client and the two control ports;
* reporting does not travel. Each host writes its own half of a request's row into
  the run's ledger, keyed by request id, and the join happens at the collector;
* the handoff is a real transfer. Under disaggregation the decode host has none of
  the request's KV, so :meth:`ServingHost.decode` drives a real ``get_batch`` over
  the whole block chain, priced by the same cost model as every other fetch. A
  dedicated decode pool buys isolation from prefill and pays for it in KV transfer.

:meth:`prefill` carries a :func:`~proposed.routed.routed` declaration naming where in
its answer the address is (:attr:`Prefilled.elsewhere`).

**Missing:** which host a request *arrives* at is a load balancer's answer -- a
client SDK, an ingress proxy, DNS -- and is not modelled. The run's wiring stands
in for it (:mod:`kvcache_sim.workload._serving`).

Why does a decode host publish?
-------------------------------
KV that lands on a host is registered on that host's volume. A prefill host
publishes the prefix it pulled and the suffix it computed; a decode host publishes
the chain it pulled in and the KV it generated, through the same
:meth:`~kvcache_sim.data._store.KVStore.publish` -- evictable by the same LRU,
refusable by the same bounded volume. Otherwise a decode host has unbounded free
memory and no capacity, eviction or hit-rate number ever feels it.

Two consequences, both intended. The decode host becomes a **replica**: the
directory maps the chain to it as well as to the prefill host, so a later request
can be routed to, or pull from, a host that only ever decoded that prefix. And
decode **competes** for the volume: a decode pool holding every chain it has served
will evict the prefixes prefill is trying to reuse.

Where does control's forecast diverge from what happens?
--------------------------------------------------------
Control prices ``queue -> transfer -> prefill`` and reserves the instance for all
three, but a forward pass cannot be submitted before its inputs have arrived, so
here the queue wait sits *behind* the fetch, and the fetch runs on the fabric while
the device belongs to somebody else. The gap is widest for requests that pull.

Simplification (documented in SPEC): a block becomes reusable at prefill
*completion*, not while in flight, so two requests racing for the same brand-new
prefix may both compute it. Arrivals are typically spaced enough that this is rare.

What may this host touch on the control side?
---------------------------------------------
Two ports, split between asking and telling:

* :class:`~proposed.plane.ControlPlane` -- the **questions**. ``decide`` (here)
  answers a :class:`~kvcache_sim.control.scheduler.Response`: a value naming both
  of the request's hosts and what prefilling on the first was priced at, which the
  named host passes on in its own answer (:class:`Prefilled`) as far as the host
  that decodes. ``sources`` (asked by :mod:`kvcache_sim.data._store`, where the
  fetch is) names which peer serves a fetch;
* :class:`~proposed.dispatch.Dispatcher` -- the **facts**: this host's decode batch,
  its busy compute, the clock its prefill really reached. Nothing comes back, and
  the reply is awaited anyway, because the next question has to be decided against
  sensors that have already folded the action.

Those two are the *only* things this module may touch on the control side.
``check_structure.py`` rule 6 fails the build on a field read, a subscript or a
``getattr`` through either, since none of those survive the planes being in
different processes. So the decode engine's callbacks come back to their owner on
this host, and this plane decides what to send on as a
:class:`~kvcache_sim.control.scheduler.DecodeState` fact.

Whether prefill and decode contend for this host's compute is a fact about the
deployment, so the host owns it: it hands both engines one accelerator or two, and
reports each decode step's end as a
:class:`~kvcache_sim.control.scheduler.ComputeBusy` fact so control's *predicted*
prefill queue tracks the device decode is actually using. A disaggregated host
reports nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

import torch

from proposed import ControlPlane, DataPlane, Deployment, endpoint, routed

from ..control.scheduler import (
    ComputeBusy, DecodeState, Plan, PrefillFinished, Response,
)
from ..report.metrics import Metrics, RequestResult
from ..control.request import Request
from ._decode import DecodeEngine
from ._prefill import PrefillEngine
from ._store import KVStore

__all__ = ["Prefilled", "ServingHost"]


@dataclass(frozen=True)
class Prefilled:
    """What :meth:`ServingHost.prefill` answers: an address, or a prefilled request.

    An address or the other three, never both.
    """

    #: The host the request belongs on. Set means nothing ran here.
    elsewhere: Optional[str] = None
    #: The decision this request was served by; :meth:`ServingHost.decode` reads
    #: the request's two hosts off it.
    response: Optional[Response] = None
    #: The first token, produced by this host's prefill.
    token: Optional[torch.Tensor] = None
    #: The host that generates the rest, ``None`` in a run with no second half.
    decode: Optional[str] = None


class ServingHost(DataPlane):
    """One serving instance: its cache, its decode batch, its compute.

    The running loop's ``time()`` is the only clock (virtual under simulation).
    :meth:`prefill` and :meth:`decode` are the whole surface a client reaches.

    Args:
        me: this host's instance id -- the only one this object holds. A plan may
            name others, but they are addresses to hand back to a client.
        trace: the run's shared trace.
        metrics: the run's :class:`~kvcache_sim.report.metrics.Metrics` ledger.
        prefill: this host's :class:`~kvcache_sim.data._prefill.PrefillEngine`,
            or ``None`` if it does not prefill.
        decode: this host's :class:`~kvcache_sim.data._decode.DecodeEngine`, or
            ``None`` if it does not decode. Whether the two were handed the *same*
            :class:`~kvcache_sim.data._compute.Accelerator` is whether they contend,
            and so whether this host reports its compute timeline at all.
        models_decode: whether this *run* models the request's second half at all.
            Not the same question as whether this host decodes: a prefill-only host
            in a disaggregated run has no decode engine, and its requests still go
            on to decode somewhere else.
    """

    def __init__(
        self,
        me: str,
        *,
        trace,
        metrics: Metrics,
        prefill: Optional[PrefillEngine] = None,
        decode: Optional[DecodeEngine] = None,
        models_decode: bool = False,
    ) -> None:
        self.me = me
        # Filled by attach(): neither exists before the deployment does.
        self.deployment: Optional[Deployment] = None
        self.store: Optional[KVStore] = None
        self.trace = trace
        self.metrics = metrics
        self.prefill_engine = prefill
        self.decode_engine = decode
        self.models_decode = models_decode
        if decode is not None:
            decode.on_finish = self._decode_done
            decode.on_state = self._decode_state
            # Coupled: one accelerator, so a prefill here delays a decode step here.
            if prefill is not None and prefill.compute is decode.compute:
                decode.on_compute_busy = self._compute_busy

    # -- what this host reaches: the store, and the two control ports -------- #
    def attach(self, deployment: Deployment) -> None:
        """Take the deployment this host reaches its control plane and store through."""
        self.deployment = deployment
        self.store = KVStore(deployment)

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    # -- what this host dispatches about its decode side -------------------- #
    async def _decode_state(self, finishes: List[float]) -> None:
        """Forward a changed decode batch. The engine reports here, not there."""
        await self.deployment.dispatcher_handle.dispatch.call_one(
            DecodeState(self.me, tuple(finishes))
        )

    async def _compute_busy(self, until: float) -> None:
        """Forward this host's occupied compute timeline (coupled only)."""
        await self.deployment.dispatcher_handle.dispatch.call_one(
            ComputeBusy(self.me, until)
        )

    # -- the request's prefill: decide where it belongs, or serve it -------- #
    @endpoint
    @routed(at=lambda prefilled: prefilled.elsewhere)
    async def prefill(self, request: Request) -> Optional["Prefilled"]:
        """Ask control where ``request`` belongs; serve it here, or answer with there.

        An answer naming another host goes back out as an address
        (:attr:`Prefilled.elsewhere`) with nothing run and nothing held here. A
        refusal is ``None``, and is recorded here: no other host will hear of this
        request.

        The first token comes back whenever this host prefilled, even in a run that
        names no decode host.
        """
        control: ControlPlane = self.deployment.control_plane_handle
        response = await control.decide.call_one(request, self.me)
        if response is None:
            self.trace.record(
                self._now(), "REJECT", f"{request.id} rejected (SLO/overload)"
            )
            self.metrics.add(
                RequestResult(
                    id=request.id, accepted=False, prompt_tokens=request.prompt_tokens
                )
            )
            return None
        if response.prefill != self.me:
            # Traced only when the answer moves the request; the serving host's own
            # ROUTE line already records where it landed.
            self.trace.record(
                self._now(), "REDIR", f"{request.id} {self.me} -> {response.prefill}"
            )
            return Prefilled(elsewhere=response.prefill)
        # Serve it here. Nothing is reserved: the pass books the device when it is
        # submitted, behind whatever already has it.
        plan = response.plan
        self._trace_route(request, response)
        row = self._make_accepted(request, response)
        # Added before the work below; amended in place as the facts land.
        self.metrics.add(row)

        # (1) A local prefix is a read the store never sees, so tell it: the volume
        # evicts on what it has observed.
        if plan.local_blocks:
            await self.store.reuse(
                self.me, list(request.block_keys[:plan.local_blocks])
            )
        # Pull the remote prefix: a real get_batch, real fabric cost. The KV goes into
        # the forward pass below and out again in what that pass publishes.
        uncached = plan.uncached_tokens
        pulled: List[torch.Tensor] = []
        if plan.reuse_source is not None and plan.pull_keys:
            try:
                pulled = await self.store.fetch(self.me, plan.pull_keys)
            except KeyError:
                # Evicted on the peer since routing. A pull is all-or-nothing, so the
                # whole planned reuse is recomputed rather than half a prefix used.
                uncached = self._recompute(request, plan, row)
                pulled = []
        # (2) Submit the forward pass for the uncached suffix. The engine is told the
        # work, not a duration: it waits here for the device behind whatever pass or
        # decode step has it, and answers with the KV (prefix in, suffix computed)
        # and the first token. The id tags the submission, breaking same-instant ties
        # in the accelerator's service order.
        #
        # Sliced by *offset*: ``prompt[-uncached:]`` would hand a fully-cached
        # request its entire prompt back.
        prompt = request.prompt[request.prompt_tokens - uncached:]
        submitted_at = self._now()
        kv, first_token = await self.prefill_engine.run(
            prompt, pulled, tag=request.id
        )
        # Whatever that took beyond the pass itself was queueing for the device: the
        # *measured* wait, beside control's prediction of it already on the row.
        row.queue_wait = (
            self._now() - submitted_at - self.prefill_engine.cost(uncached)
        )

        # (3) Publish everything past what was already local: the prefix pulled plus
        # the suffix computed. Under disaggregation this is also how *this* request's
        # KV reaches its decode host.
        fresh = list(request.block_keys[plan.local_blocks:])
        # A refused fill costs the request nothing, but "cached" and "had no room"
        # are the two outcomes a capacity sweep exists to tell apart, so record it.
        row.published = await self.store.publish(self.me, fresh, kv)
        # (4) Dispatch the clock the real ops reached: the first news control gets
        # that its queue forecast was wrong. Awaited for the ordering, not the answer
        # -- the next request routed against this host must be priced against a
        # sensor that has already folded this completion.
        await self.deployment.dispatcher_handle.dispatch.call_one(
            PrefillFinished(self.me, self._now())
        )
        return Prefilled(
            response=response,
            token=first_token,
            decode=await self._prefill_done(request, response, row),
        )

    def _recompute(self, request: Request, plan: Plan, row: RequestResult) -> int:
        """Re-price this prefill without the reuse that vanished; answer what is left.

        Only what this host already held is still cached. The row is corrected too:
        a hit rate counting the plan rather than the outcome would flatter the cache
        that dropped the blocks.
        """
        cached = min(
            plan.local_blocks * self.prefill_engine.block_tokens,
            request.prompt_tokens,
        )
        uncached = request.prompt_tokens - cached
        row.cached_tokens = cached
        row.uncached_tokens = uncached
        row.transfer_bytes = 0
        row.ttft += self.prefill_engine.cost(uncached) - plan.prefill_t
        self.trace.record(
            self._now(),
            "RESTALE",
            f"{request.id} lost {len(plan.pull_keys)}blk of reuse on "
            f"{plan.reuse_source} (evicted); recomputing on {self.me}",
        )
        return uncached

    # -- outcome bookkeeping ---------------------------------------------- #
    def _make_accepted(
        self, request: Request, response: Response
    ) -> RequestResult:
        plan = response.plan
        return RequestResult(
            id=request.id,
            accepted=True,
            ttft=plan.ttft,
            prompt_tokens=request.prompt_tokens,
            cached_tokens=plan.cached_tokens,
            uncached_tokens=plan.uncached_tokens,
            transfer_bytes=plan.transfer_bytes,
            prefill=response.prefill,
            reuse_source=plan.reuse_source,
            predicted_queue_wait=plan.queue_wait,
        )

    def _trace_route(self, request: Request, response: Response) -> None:
        plan = response.plan
        if plan.reuse_source is not None:
            src = f" pull {plan.match_blocks}blk from {plan.reuse_source}"
        elif plan.match_blocks:
            src = f" local hit {plan.match_blocks}blk"
        else:
            src = " cold (no reuse)"
        self.trace.record(
            self._now(),
            "ROUTE",
            f"{request.id} -> {response.prefill}"
            f" (match {plan.match_blocks}blk,{src}, "
            f"compute {plan.uncached_tokens}tok, ttft {plan.ttft:.3f})",
        )

    async def _prefill_done(
        self, request: Request, response: Response, row: RequestResult
    ) -> Optional[str]:
        """Close out the prefill, and answer with the next address (or none)."""
        note = "" if row.published else " -- NOT cached, no room on the volume"
        # What was published: the blocks pulled plus the suffix computed, not the
        # request's whole chain.
        stored = len(request.block_keys) - response.plan.local_blocks
        if not self.models_decode:
            self.trace.record(
                self._now(),
                "DONE",
                f"{request.id} prefill done on {self.me}"
                f" (published {stored}blk){note}",
            )
            return None
        # Decode is modelled: nothing is asked here, the decision already named the
        # host and refused this request or did not, before its prefill ran.
        self.trace.record(
            self._now(),
            "DONE",
            f"{request.id} prefill done on {self.me}"
            f" (published {stored}blk){note}"
            f"; decoding on {response.decode}",
        )
        # The address, not the act: the client goes there next.
        return response.decode

    # -- decode, on a host that has to go and get the KV -------------------- #
    # No declaration: this answer is tokens, and a host that has them is the last one
    # the request visits.
    @endpoint
    async def decode(
        self, request: Request, response: Response
    ) -> List[torch.Tensor]:
        """The client brought a prefilled request here. Fetch its KV, decode, finish.

        Answers with the tokens **this host** generated -- the output minus the first
        -- and only once the last one is emitted. The fetch below lands in neither
        headline column (not TTFT, predicted before any of it happens; not TBT,
        measured between decode tokens, and this finishes before the first), so an
        early return would charge the dominant cost of disaggregation on the clock and
        report it nowhere (:mod:`kvcache_sim.workload._serving`).

        Waiting cannot deadlock: a refused request never reaches here, one with <= 1
        output token retires inside
        :meth:`~kvcache_sim.data._decode.DecodeEngine.admit` on the instant it
        arrived, a host with no engine returns without waiting, and a queued request
        enters the batch as a slot frees -- slots always free, because every member's
        ``remaining`` falls by one per step.

        The fetch covers the **whole block chain**, not just what the prefill host
        published: a decode step attends over every token of the prompt. It completes
        before the batch admits the request, as DistServe and Mooncake also put KV
        migration in TTFT; as an inter-token gap it takes both the disaggregated and
        the coupled column to 0.0% attainment against a five-step target. **Unless
        this host prefilled the request**, which the decision says outright: the chain
        is here and the local-hit rule charges nothing. Not a rare corner -- with
        prefill and decode drawn from one pool it was 73% of this scenario's reported
        handoff bytes.

        A missing block is not fatal. ``get_batch`` is all-or-nothing, so a failed
        publish or a since-evicted block surfaces as a ``KeyError`` over the whole
        batch; the request decodes anyway, and
        :attr:`~kvcache_sim.report.metrics.Metrics.handoff_misses` tells a run its
        decode pool is fed by a cache too small for the handoff.

        Both what arrives and what the batch generates are published here
        (:meth:`_reside`, the second under keys continuing the prompt's chain), and
        the generated publish waits for the last token: the loop is one coroutine
        driving one accelerator, so publishing between two steps would stall every
        other member of the batch, a TBT effect the hardware does not have.

        **Missing:** a host that did not prefill may still hold part of the chain,
        which is not asked, so a cross-host handoff over-charges by whatever it
        shared. Deferring the generated publish under-charges intra-generation
        residency, which nothing here runs long enough to feel. Preemption is not
        modelled.
        """
        keys = list(request.block_keys)
        if response.decode == response.prefill:
            # Ours already. Tell the volume it was read, so its eviction ranking
            # sees a hit the store would otherwise never observe, and record a
            # handoff of nothing -- there was no transfer to make.
            await self.store.reuse(self.me, keys)
            self.metrics.handed_off(request.id, self.me, 0)
        else:
            try:
                kv = await self.store.fetch(self.me, keys)
            except KeyError:
                self.trace.record(
                    self._now(),
                    "NOKV",
                    f"{request.id} handoff from {response.prefill} to {self.me} "
                    f"found no {len(keys)}blk chain in the store (evicted or never "
                    f"cached); decoding without charging a transfer",
                )
                self.metrics.handed_off(request.id, self.me, 0, missed=True)
            else:
                # Measured off the tensors the transport just charged for, so the
                # reported handoff cannot drift from the charged one.
                nbytes = sum(b.numel() * b.element_size() for b in kv)
                self.trace.record(
                    self._now(),
                    "HANDOFF",
                    f"{request.id} {response.prefill} -> {self.me} "
                    f"({len(keys)}blk, {nbytes}B of KV)",
                )
                self.metrics.handed_off(request.id, self.me, nbytes)
                # Published *after* the handoff is recorded, so the reported transfer
                # stays the bytes that crossed the fabric.
                await self._reside(keys, kv, request.id, "chain")
        # No engine: a run that does not model decode. The only path that answers
        # without the request having finished, since no last token is coming.
        if self.decode_engine is None:
            return []
        generated = await self.decode_engine.admit(request)
        # What the batch produced, under the chain continued past the prompt.
        await self._reside(
            request.continuation_keys(len(generated.kv)),
            generated.kv,
            request.id,
            "generated",
        )
        return generated.tokens

    async def _reside(
        self, keys: List[str], blocks: List[torch.Tensor], request_id: str, why: str
    ) -> None:
        """Register KV this host now holds on this host's volume.

        ``why`` names which publish it was, for the trace only; nothing branches on it.
        A refusal is not fatal -- the request is served off KV held in hand either way
        -- and is flagged on the row rather than counted into it
        (:attr:`~kvcache_sim.report.metrics.Metrics.decode_unpublished`), since blocks
        the volume threw back occupy nothing.
        """
        if not blocks:
            return
        ok = await self.store.publish(self.me, keys, blocks)
        self.metrics.decode_resident(request_id, len(blocks), published=ok)
        nbytes = sum(b.numel() * b.element_size() for b in blocks)
        self.trace.record(
            self._now(),
            "RESIDE" if ok else "NOROOM",
            f"{request_id} {why} {len(blocks)}blk ({nbytes}B) "
            f"{'now resident on' if ok else 'did not fit'} {self.me}",
        )

    def _decode_done(self, request: Request, tbt: float) -> None:
        """Record this batch's inter-token gaps into the row the prefill host opened."""
        self.metrics.decoded(request.id, tbt)
        self.trace.record(
            self._now(), "DECODE", f"{request.id} decode done (tbt {tbt:.3f})"
        )
