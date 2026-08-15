"""Every volume that could serve a read, priced in one pass: :class:`Candidates`.

With no routing, ``m`` readers of one key all ``locate_volumes`` before anyone finishes,
so all ``m`` are told "the origin" and each pulls from it -- ``m x`` fabric. Readers ask
this ranking instead, and it prices one pool: the volumes that hold the key **and** the
readers already routed to fetch it, which will hold it shortly. A peer is not a
different kind of source, only one whose copy has not arrived yet::

    score = wait + hop + fabric * hop        (seconds; lower is better)

    wait = seconds until that source holds the key -- zero for a holder, and for a peer
           the sum of the real link times up its own branch
    hop  = what the transfer to the requester costs over the link between them

Both terms are seconds off the run's own cost model
(:meth:`~proposed.view.View.transfer_cost`), so nothing here weighs a distance against a
delay: a hop is expensive because the link is slow, and a chain is expensive because
each of its links takes what it takes. No tier arithmetic, and no units to reconcile.

``fabric`` is the one weight, and it is dimensionless: what a second of the link this
read occupies is worth against a second of the requester's own waiting.

* :data:`CHAIN` -- link time worth many times the requester's wait. A hop off a holder
  is charged for the fabric it burns, so a near peer outprices a far holder, the burst
  folds into a chain or a tree, and the holders are read once: dedup's 1x. It holds
  while the branch's accumulated link time stays under that charge -- past that a fresh
  hop off a holder really is the better answer, and the depth where that happens comes
  out of the machine's own bandwidth ratio rather than a constant chosen here.
* :data:`SPREAD` -- nothing: the fabric is nobody's but mine, so the soonest source
  wins and a holder that has the key beats a peer still fetching it.

No queue is charged in the score. How many readers are already behind a peer is read
here, but only as the cap that stops offering it at all; what a busy source *costs* is
the plane's trade to make over this ranking
(:mod:`dedup_sim.control.routing`, :class:`~proposed.selector.Balance`).

The ranking only prices. Ordering the pool and doing something with the winner -- the
route recorded, the answer withheld until that source is usable -- is the plane's
(:mod:`dedup_sim.control.routing`, :func:`~dedup_sim.control._answer.committed`), which
is what lets a combinator re-key this from above. Nothing here suspends, so a decision
runs to completion before the next requester's starts; the rest of that argument is on
:mod:`dedup_sim.control._answer`.

There is no burst loop, no reader list and no knowledge of how many readers there will
be. The tree is one requester at a time, and it *executes* because each reader's
read-through commits an action whose boundary releases the next reader's withheld
answer -- an emergent property of what the data plane commits, not a schedule this
module runs.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from proposed import Key, KeySelector, Selection, VolumeId

from ._answer import holders
from ._view import FanoutView

__all__ = ["Candidates", "CHAIN", "SPREAD"]

#: What a second of the link a read occupies is worth against a second of the
#: requester's own waiting -- the whole of the difference between folding a burst into a
#: tree and fanning it across the replicas. :data:`CHAIN` is measured: on the default
#: profile it keeps a cap-1 chain 1x seven readers deep and a cap-2 tree 1x well past
#: sixty (``dedup_sim/tests/test_dedup.py``).
CHAIN = 10.0
SPREAD = 0.0


class Candidates(KeySelector):
    """Holders and planned peers as one pool, each priced in seconds.

    Args:
        fabric: what a second of somebody else's link time is worth against a second of
            the requester's own wait (:data:`CHAIN`, :data:`SPREAD`).
        payload_bytes: bytes to price a hop for. The directory reports what a volume
            holds and not how big it is, so this is a nominal byte by default and a hop
            is priced at little more than its link's latency; a run that knows the size
            can say so, and the bandwidth term then sharpens the ratio between tiers.

    A peer is offered only while it still **owes** the key: from the moment it asks
    until its read-through lands (:meth:`~dedup_sim.control._sensor.FanoutSensor.owes`).
    One that published and has since evicted owes nothing and holds nothing, so it
    simply is not a candidate -- there is no source to retire from a ranking that never
    named it.
    """

    name = "candidates"
    sensors = (FanoutView,)

    def __init__(self, fabric: float = CHAIN, payload_bytes: int = 1) -> None:
        self.fabric = fabric
        self.payload_bytes = payload_bytes

    async def select(self, keys: Sequence[Key], requester: str) -> Selection:
        """Everything that could serve every one of ``keys``, scored; else abstain.

        The score is the whole of the key
        (:meth:`~proposed.selector.Selection.priced`): what a source costs *is* what
        orders it here. Ordering is the plane's, one fold at the end
        (:meth:`~dedup_sim.control.routing.Dedup._decide`).
        """
        fanout = self.view.fanout
        fanout.promise(requester, keys)
        located = self.view.locate(keys)
        holds = set(holders(located, keys[0]))
        for key in keys[1:]:
            holds &= set(holders(located, key))
        routes, queued = fanout.routes(), fanout.named()
        # The requester's own route, so a source it is already behind is not counted
        # as full *by it*: a second ask costs no slot, and a reader re-asking after an
        # eviction would otherwise be shut out by its own place in the queue.
        mine = fanout.planned(requester)
        priced: List[Tuple[VolumeId, float]] = []
        # One entry per volume the walk below reaches, so the branch is walked once per
        # decision and not once per candidate. The pool itself needs no order: what
        # orders it is the score, and the fold that reads it is total (the id its last
        # tie-break).
        waits: Dict[VolumeId, Optional[float]] = {}
        for volume in holds | routes.keys():
            if volume == requester:
                continue
            wait = self._wait(volume, keys, holds, routes, waits)
            if wait is None:
                continue
            behind = queued.get(volume, 0) - (volume == mine)
            if wait and behind >= fanout.cap:
                # A peer at its cap: the ceiling on how wide the tree may fan out, and
                # the only thing the cap does. A holder has no such limit -- what keeps
                # readers off one is its price, and whatever the plane weighs against
                # it.
                continue
            hop = self.view.transfer_cost(volume, requester, self.payload_bytes)
            priced.append((volume, wait + hop + self.fabric * hop))
        if not priced:
            return Selection.of([])
        return Selection.priced(priced)

    def _wait(
        self,
        volume: VolumeId,
        keys: Sequence[Key],
        holds: Set[VolumeId],
        routes: Mapping[VolumeId, VolumeId],
        waits: Dict[VolumeId, Optional[float]],
    ) -> Optional[float]:
        """Seconds until ``volume`` holds ``keys``; ``0`` if it does, ``None`` if never.

        A volume holding them now waits for nothing. Otherwise it is a peer, and what it
        waits for is its own source's wait plus the link between the two -- so a branch
        costs what its links cost rather than a hop count, and a peer two cheap hops
        away can be sooner than one over a slow link. It counts only while every step
        up that branch is still coming: a volume that owes nothing and holds nothing has
        no copy on the way, and so is no source. Bounded by the number of edges, since
        a walk that cannot terminate is a cycle this would otherwise follow forever.

        Elapsed time is not subtracted: a fetch already in flight is priced as if it
        started now, which reads a peer as slightly further off than it is. Exact for a
        synchronized burst, where every decision is made at one instant; the correction
        is a predicted landing time the fan-out does not keep.

        Every volume the walk passes is answered on the way back down (``waits``), so
        one decision costs one walk of the tree however many candidates hang off it --
        a chain of ``m`` readers is ``m`` steps in total, not ``m`` per reader.
        """
        fanout = self.view.fanout
        branch: List[VolumeId] = []
        while True:
            if volume in waits:
                wait = waits[volume]
                break
            if volume in holds:
                wait = waits[volume] = 0.0
                break
            if len(branch) > len(routes) or not fanout.owes(
                [(volume, key) for key in keys]
            ):
                wait = waits[volume] = None
                break
            branch.append(volume)
            source = routes.get(volume)
            if source is None:
                wait = waits[volume] = None
                break
            volume = source
        for node in reversed(branch):
            if wait is None:
                waits[node] = None
                continue
            wait += self.view.transfer_cost(volume, node, self.payload_bytes)
            waits[node] = wait
            volume = node
        return waits[branch[0]] if branch else wait
