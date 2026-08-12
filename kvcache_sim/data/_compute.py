"""The accelerator an engine runs on, as the engine sees it: :class:`Accelerator`.

Two things a serving host does are not store calls: it runs a forward pass, and it
runs decode steps. Everything else in ``data/`` is an ordinary torchstore call and
lifts unchanged, and for a long time these two did not -- the engines slept the
duration a cost model predicted, which is the one thing in here that could only
ever be a simulation. A deployment does not sleep. It computes.

So the compute is a port. An engine says *what* work it has; the thing behind this
protocol says how long that takes and makes it take that long. Under simulation
that is a roofline and an ``asyncio.sleep``
(:class:`kvcache_sim.workload._accelerator.SimulatedAccelerator`, which lives with
the run's wiring because it is the simulation); in a deployment it is the model's
forward pass, and the durations are measured rather than predicted.

It is the compute counterpart of :class:`proposed.cost.TransferCost`, and it needs
one thing that port does not: fabric cost is *incurred* by the real client call the
data plane already makes, so predicting it is enough. Nothing here makes a GPU busy
on its own, so this both predicts and incurs.

A forward pass produces KV, so this port hands it back
--------------------------------------------------------
:meth:`Accelerator.prefill` answers with the KV blocks the pass produced, and
:attr:`Accelerator.block_tokens` says how many tokens one of them holds. Both are
here rather than beside the store because the thing that knows what a forward pass
*costs* is the thing that knows what it *produces*: the block size is the engine's
own cache layout, and the tensors are the engine's own output. The store used to be
told both -- it was constructed with a "carrier" and a token count and manufactured
one identical stand-in per key -- which put the byte count that every fetch is
priced against in the one object that computes nothing.

So the serving host publishes what prefill handed it, and the store publishes what
it is handed. A simulated accelerator returns ``device="meta"`` tensors: real
``torch.Tensor`` objects of the right dtype and byte count with no storage behind
them (:class:`kvcache_sim.workload._accelerator.SimulatedAccelerator`); a deployment
returns the KV its attention kernels just wrote. Neither is a shape this port has to
know about, which is the point of handing it back rather than describing it.

One object is one accelerator
-----------------------------
A host has one, and hands it to whichever engines run on it. Whether its prefill
engine and its decode engine were handed the **same** object is the whole of what
"coupled" means: two engines on one accelerator collide, so a long prefill delays
the next decode step. A run that models them as not colliding hands out two, which
is a simplification rather than a deployment and is therefore the wiring's to
choose, not a flag an engine carries.

...and one accelerator runs one thing at a time
-----------------------------------------------
Which is why the occupancy is here and not in an engine. A decode step books it
(:meth:`claim_step`) and a forward pass runs on it (:meth:`prefill`), so the two
compose without either engine knowing the other exists: whichever asks second
waits for the first. An engine never *reads* the occupancy -- it asks for work to
be done and is told when it was done -- which is why there is no member here for
"when are you free?" and no way for one engine to schedule around the other except
by actually queueing behind it.

There used to be a third writer: a ``reserve(until)`` the serving host called with
the control plane's *predicted* completion, so that a decode step would not be
scheduled through a prefill the accelerator itself knew nothing about. It knows
now, so the member is gone rather than kept and ignored. Its removal is the whole
of what "one owner of busy_until" means here, and the alternative -- a prediction
and a measurement writing one field, the first before the work starts and the
second after it ends -- is the kind of thing that is right until the two disagree,
which is exactly the case this model exists to study.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import torch

__all__ = ["Accelerator"]


class Accelerator(ABC):
    """What an engine may ask of the compute it runs on.

    Split into asking what work costs and doing it, because both are needed and
    only one of them is a prediction. A deployment answers the cost members by
    measuring what it has already run; a simulation answers them from a roofline
    and then sleeps exactly that long, which is what makes a simulated run's
    charged time equal its predicted time.

    A base class rather than a ``Protocol``, for the reason
    :class:`proposed.coordinator.Coordinator` gives: the protocols in this repo are
    surfaces a caller *reaches* -- a directory, a volume, a coordinator, each behind
    a handle whose real implementation is somebody else's class and cannot inherit
    from ours. This is a surface somebody *writes*, in this process, and an
    implementation should say so in its bases and be told at construction when it
    has missed a member.

    Every member is abstract: there is no sensible default for what a forward pass
    costs, and a stub that answered zero would make a run that forgot to supply one
    look merely fast.
    """

    @property
    @abstractmethod
    def block_tokens(self) -> int:
        """Tokens one KV block holds -- this engine's cache-page size.

        A fact about how the engine lays its KV cache out, and therefore what
        :meth:`prefill` returns one tensor per. Read off the accelerator rather
        than configured a second time next to the store, because two numbers that
        have to agree and are set in two places eventually do not: the store's copy
        used to be what a request's recomputed prefix was measured with while this
        one was what the KV was actually cut into.
        """

    @abstractmethod
    def prefill_cost(self, tokens: int) -> float:
        """What prefilling ``tokens`` uncached tokens costs here."""

    @abstractmethod
    def step_cost(self, batch_size: int) -> float:
        """What one decode step over ``batch_size`` requests costs here."""

    @abstractmethod
    async def prefill(
        self, tokens: int, cached: Sequence[torch.Tensor] = (), *, tag: str = ""
    ) -> List[torch.Tensor]:
        """Submit a forward pass over ``tokens`` uncached tokens; answer with its KV.

        **Runs when this accelerator is free to run it**, which is the difference
        between submitting work and being told how long to sleep. The pass is
        booked on the same occupancy every other kind of work here books on, so it
        queues behind a prefill already running and behind a decode step in
        progress (:meth:`claim_step`), and the next decode step queues behind it --
        which is what makes "coupled" cost something rather than merely mean two
        engines were handed one object. How long a request waited is therefore
        something the caller can measure and nothing here has to be told.

        This used to be paired with the caller sleeping a queue wait the *control
        plane predicted*, and only then calling this to sleep the pass. Two things
        were wrong with that and both are the same thing: the wait was an input
        rather than an output, so no run could ever contradict the scheduler's
        model of its own queue; and it put the wait in front of the caller's KV
        fetch, where control's arithmetic has it, rather than behind it, where a
        forward pass that cannot start before its inputs arrive actually has it.

        ``cached`` is the KV this host pulled for the prefix in front of those
        tokens -- blocks another host computed and this one fetched out of the store.
        It is passed in rather than fetched here because getting it is a store call
        and this port makes none; what an engine does with it is load it into the
        cache the forward pass then attends over, which is why it comes back out in
        the answer.

        ``tag`` is a stable name for this submission -- the request's id, in this
        application. It is not needed to run the pass; it is needed to *order* it.
        An implementation serves its queue by ``(submitted at, tag)``, so two
        passes handed in at the same virtual instant run in an order the workload
        fixes rather than one the event loop's ready queue happens to produce. A
        queue ordered by whichever coroutine resumed first is a queue that can
        reorder when an unrelated ``await`` is added upstream of one of its
        callers, which is a determinism bug that hides for months.

        Answers with **one tensor per KV block this host now holds and did not
        before**, in prompt order: the pulled prefix first, then the suffix this
        pass computed. That is exactly the set the caller publishes, which is the
        reason for the shape -- the alternative, returning only the newly computed
        blocks and leaving the caller to splice the pulled ones back in front, puts
        the ordering of a request's KV in the serving loop, where a silent
        off-by-one would publish real bytes under the wrong keys.

        A real engine holds its KV as one contiguous per-layer region and would
        slice it; per-block is what the store wants and what a paged engine already
        has, so it is what this port asks for. Returning the region plus a block
        table was considered and rejected: it is a second description of the same
        layout that only the caller would use, and only to cut it up again.
        """

    @abstractmethod
    def claim_step(self, batch_size: int) -> float:
        """Book the next decode step and answer when it will end.

        Two members rather than one ``await decode_step(...)`` because the caller
        has to announce the end *before* waiting for it: an engine whose prefill
        shares this accelerator reports each step's end onward, and a report that
        arrived after the step would be telling the control plane about a
        collision it could no longer avoid.
        """

    @abstractmethod
    async def wait_until(self, when: float) -> None:
        """Wait until ``when``, the moment :meth:`claim_step` answered with."""
