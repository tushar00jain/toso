"""1x-fabric dedup routing, as a :class:`proposed.policy.Policy`.

The question a synchronized read burst asks the store is exactly the one the
policy interface answers: *given this key and this requester, which volume serves
it, and when is that volume usable?* Answering it is the whole capability.

How 1x falls out
----------------
With no routing, ``m`` readers of one key all ``locate_volumes`` before anyone
finishes, so all ``m`` are told "the origin" and each pulls from it -- ``m x``
fabric.

``DedupPolicy`` answers differently. Readers arrive at the controller in order;
the first is routed to the volume that already holds the key (the single fabric
hop), and every later one is routed to a **peer** -- a reader that is *about to*
hold it. Because that peer has not registered yet, the selection carries a
readiness gate, and the controller withholds its answer until the peer's
read-through put lands (:meth:`DedupPolicy.notice`). Peers are handed out FIFO
under a fan-out cap, so cap 1 builds a chain and cap >= 2 a shallow tree.

Whether a peer holds the key is asked of the directory every time an answer is
formed (:meth:`DedupPolicy._registered`), never remembered: volumes evict, so a
peer that registered the key and later dropped it for a newer version is a peer
the next requester has to wait for again.

Which raises the question of when waiting is safe at all, because a gate that
nothing will ever open hangs the requester behind it. The answer is the debt the
policy is already tracking: a requester it routes is going to read the key
through, so from the moment it asks it *owes* that registration, and waiting for
it is bounded. A source that owes nothing has to hold the key already, and one
that holds nothing and owes nothing is no longer a source at all -- it is retired
and the directory answers for that requester (:meth:`DedupPolicy._retire`).

Exactly one reader is ever routed to a pre-existing holder, so exactly one
transfer's source is an origin: ``origin_bytes`` is the 1x union whatever the cap.

What is *not* here
------------------
There is no burst loop, no reader list, and no knowledge of how many readers
there will be. The tree is assigned one requester at a time as they ask, and the
chain executes because each reader's read-through put releases the next reader's
withheld answer -- an emergent property of the data plane's registration, not a
schedule this module runs. That is what makes the scenario ordinary user code:
a gather of ``client.get(K)``, with no idea a policy exists.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Hashable, List, Optional, Sequence, Set, Tuple

from proposed import DecisionLog, Policy, Selection

from ._readiness import Readiness

__all__ = ["DedupPolicy"]


class DedupPolicy(Policy):
    """Route each requester to a peer, not the origin (a real ``Policy``).

    Args:
        fanout_cap: how many peers one source may be planned to feed (1 = a
            chain, >= 2 = a shallow tree). The fabric stays 1x for any cap; the
            cap only trades wallclock against tree depth.
        trace: optional :class:`~proposed.policy.DecisionLog` to record each
            routing decision into. Changes no metric; the policy behaves
            identically with none attached.
    """

    name = "dedup"

    def __init__(
        self, *, fanout_cap: int = 1, trace: Optional[DecisionLog] = None
    ) -> None:
        self.cap = fanout_cap
        self.trace = trace
        # requester -> the source it was routed to (decided once, then reused).
        self._route: Dict[str, str] = {}
        # One entry per peer a source may still be planned to feed, oldest first: a
        # requester joins with ``cap`` slots and each assignment consumes one. The
        # cap is the queue's own shape rather than a tally compared against it,
        # because assignment is a read-modify-write with no lock -- one popleft
        # cannot leave a half-applied cap behind the way an increment, a comparison
        # and a conditional pop could. See :meth:`_assign`.
        self._avail: Deque[str] = deque()
        # Requesters already offered their slots -- once each, however many times
        # they are assigned (see :meth:`_offer`).
        self._offered: Set[str] = set()
        # The (volume, key) publications planned and not yet seen to land: a routed
        # requester reads the key through into its own volume, so from the moment it
        # is routed it OWES that registration. The only thing that makes waiting for
        # a source safe -- see :meth:`select`.
        self._promised: Set[Tuple[str, str]] = set()
        # Waiting for the (volume, key) pairs the real directory has not registered
        # yet. The concurrency lives there, as does the rule that the directory --
        # not a memory of past registrations -- says which are true, since a volume
        # that evicts makes one false again.
        self._ready = Readiness()

    # -- decide -------------------------------------------------------------- #
    async def select(self, keys: Sequence[str], requester: str) -> Selection:
        """Route ``requester`` to a peer (or, if it is first, to a holder)."""
        # Asking is the promise: this requester is about to fetch these keys and the
        # read-through plane publishes what it fetched, so it now owes the directory
        # that registration. Recorded before any source is handed out, which is what
        # makes the invariant below hold -- a peer is only ever offered as a source
        # after it has asked, hence after it has promised.
        self._promised.update((requester, key) for key in keys)
        source = self._route.get(requester)
        if source is None:
            source = self._assign(keys, requester)
            if source is None:
                # Nobody holds it and no peer is planned to: let the directory
                # answer for itself (the naive selection).
                return Selection()
            self._route[requester] = source
            if self.trace is not None:
                self.trace.record(
                    self.view.now(), "route", f"{requester} <- {source}"
                )
        # The answer's shape depends on what the source owes:
        facts = [(source, key) for key in keys]
        if all(fact in self._promised for fact in facts):
            # It owes every key, so the wait is bounded by its read-through. Still a
            # directory read, because it may already hold a key it is about to
            # republish -- then there is nothing to wait for.
            async def registered(wanted: Sequence[Hashable]) -> List[Hashable]:
                # A readiness probe is awaited because the truth it reads may live
                # somewhere that travels. This one is the local directory.
                return self._registered(wanted)

            ready = await self._ready.gate(facts, registered)
            return Selection.of([source], ready=ready)
        # It owes nothing, so a gate here could outlive the run -- nothing would
        # ever record it. Usable only if it holds every key right now, which is the
        # ordinary case: this is how a requester routed to a pre-existing holder is
        # answered.
        if len(self._registered(facts)) == len(facts):
            return Selection.of([source])
        # Holds nothing, owes nothing: a peer that published and has since evicted.
        # Waiting would hang and naming it would route a reader to a volume with
        # nothing to serve, so it stops being a source.
        return self._retire(requester, source)

    def _retire(self, requester: str, source: str) -> Selection:
        """Drop a source nothing is coming from, and answer naively this once.

        The requester keeps no route to it, so its next ask is assigned afresh --
        to a peer that is actually going to have the key.
        """
        self._avail = deque(peer for peer in self._avail if peer != source)
        self._route.pop(requester, None)
        if self.trace is not None:
            self.trace.record(self.view.now(), "retire", f"{source} holds nothing")
        return Selection()

    def _assign(self, keys: Sequence[str], requester: str) -> Optional[str]:
        """Pick this requester's source and fold it into the read-through tree.

        Safe without a lock because it cannot suspend
        (:meth:`~proposed.deployment.Controller.locate_raw`), so it runs to completion
        before the next requester's does -- the serialized mailbox a real controller
        would give it. A directory read that could suspend would make the queue below
        a check-then-act race, and the cap would be exceeded rather than enforced.
        """
        if self._avail:
            # A peer is already planned to hold the key: take the oldest free slot
            # and offer our own.
            source = self._avail.popleft()
            self._offer(requester)
            return source
        # First requester: the closest volume that already holds every key -- the
        # one hop whose source is an origin, and the 1x fabric cost.
        located = self.view.locate(keys)
        holders = set(self.view.holders(located, keys[0]))
        for key in keys[1:]:
            holders &= set(self.view.holders(located, key))
        holders.discard(requester)
        if not holders:
            return None
        self._offer(requester)
        return self.view.nearest(sorted(holders), requester)

    def _offer(self, requester: str) -> None:
        """Offer ``requester`` as a source for up to ``cap`` later peers.

        Once per requester, however many times it is assigned. A requester whose
        source is retired is assigned afresh, and offering it again would hand it a
        second full batch of slots, so one whose first batch was already consumed
        would go on to feed ``2 x cap`` peers. Tracked separately from the queue
        because the queue only remembers the slots that are *left*: an exhausted
        requester is absent from it and would otherwise look like one never offered.
        """
        if requester in self._offered:
            return
        self._offered.add(requester)
        self._avail.extend([requester] * self.cap)

    # -- readiness ----------------------------------------------------------- #
    def _registered(self, facts: Sequence[Hashable]) -> List[Hashable]:
        """Which of these ``(volume, key)`` pairs the directory holds *now*.

        The truth a readiness gate is opened against, read rather than remembered: a
        source that registered a key and later evicted it does not hold it, and a
        requester routed there waits for the read-through that brings it back.
        """
        located = self.view.locate([key for _volume, key in facts])
        return [
            (volume, key)
            for volume, key in facts
            if volume in self.view.holders(located, key)
        ]

    def notice(self, volume_id: str, keys: Sequence[str]) -> None:
        """The real directory just registered ``keys`` on ``volume_id``.

        Releases any requester whose answer was withheld pending that volume, and
        settles the debt: the publication happened, so from here on the directory
        is what says whether that volume still holds the key.
        """
        for key in keys:
            self._promised.discard((volume_id, key))
            self._ready.record((volume_id, key))
