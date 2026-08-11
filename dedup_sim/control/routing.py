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

import asyncio
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Sequence, Set, Tuple

from proposed import DecisionLog, Policy, Selection

__all__ = ["DedupPolicy"]


class DedupPolicy(Policy):
    """Route each requester to a peer, not the origin (a real ``Policy``).

    Args:
        fanout_cap: how many peers one source may be planned to feed (1 = a
            chain, >= 2 = a shallow tree). The fabric stays 1x for any cap; the
            cap only trades wallclock against tree depth.
        trace: optional :class:`~proposed.policy.DecisionLog` to record each
            routing decision into. Control explaining itself costs nothing and
            makes the demo readable; it changes no metric, and the policy behaves
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
        # source -> peers it has been planned to feed (the tree-shaping tally).
        self._planned: Dict[str, int] = defaultdict(int)
        # sources still under the fan-out cap, oldest first.
        self._avail: Deque[str] = deque()
        # (volume, key) pairs the real directory has registered, plus the gates
        # waiting on the ones it has not.
        self._registered: Set[Tuple[str, str]] = set()
        self._gates: Dict[Tuple[str, str], asyncio.Event] = {}

    # -- decide -------------------------------------------------------------- #
    async def select(
        self,
        view: Any,
        keys: Sequence[str],
        requester: str,
        chosen: Optional[str] = None,
    ) -> Selection:
        """Route ``requester`` to a peer (or, if it is first, to a holder).

        ``chosen`` is ignored: dedupe's callers are plain ``client.get(K)`` users
        that pick nothing, which is the whole point -- the fan-out tree is a
        consequence of the controller's answers, not of anything a reader decided.
        """
        source = self._route.get(requester)
        if source is None:
            source = await self._assign(view, keys, requester)
            if source is None:
                # Nobody holds it and no peer is planned to: let the directory
                # answer for itself (the naive selection).
                return Selection()
            self._route[requester] = source
            if self.trace is not None:
                self.trace.record(
                    view.now(), "route", f"{requester} <- {source}"
                )
        return Selection.of([source], ready=self._gate_for(source, keys))

    async def _assign(
        self, view: Any, keys: Sequence[str], requester: str
    ) -> Optional[str]:
        """Pick this requester's source and fold it into the read-through tree."""
        if self._avail:
            # A peer is already planned to hold the key: attach to the oldest one
            # still under the cap, and become a source ourselves.
            source = self._avail[0]
            self._planned[source] += 1
            if self._planned[source] >= self.cap:
                self._avail.popleft()
            self._avail.append(requester)
            return source
        # First requester: the closest volume that already holds every key. This
        # is the one hop whose source is an origin -- the 1x fabric cost.
        located = await view.locate(keys)
        holders = set(view.holders(located, keys[0]))
        for key in keys[1:]:
            holders &= set(view.holders(located, key))
        holders.discard(requester)
        if not holders:
            return None
        self._avail.append(requester)
        return view.nearest(sorted(holders), requester)

    # -- readiness ----------------------------------------------------------- #
    def notice(self, volume_id: str, keys: Sequence[str]) -> None:
        """The real directory just registered ``keys`` on ``volume_id``.

        Releases any requester whose answer was withheld pending that volume.
        """
        for key in keys:
            slot = (volume_id, key)
            self._registered.add(slot)
            gate = self._gates.get(slot)
            if gate is not None:
                gate.set()

    def _gate_for(self, source: str, keys: Sequence[str]):
        """A gate that opens once ``source`` holds every key (``None`` if it does)."""
        pending = [(source, k) for k in keys if (source, k) not in self._registered]
        if not pending:
            return None
        events = [self._gates.setdefault(slot, asyncio.Event()) for slot in pending]

        async def ready() -> None:
            for event in events:
                await event.wait()

        return ready
