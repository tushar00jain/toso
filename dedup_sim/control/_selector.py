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
(:meth:`~proposed.view.View.transfer_cost`): no tier arithmetic, and no units to
reconcile.

``fabric`` is the one weight, and it is dimensionless: what a second of the link this
read occupies is worth against a second of the requester's own waiting.

* :data:`CHAIN` -- link time worth many times the requester's wait, so a near peer
  outprices a far holder, the burst folds into a chain or a tree, and the holders are
  read once: dedup's 1x. That holds while the branch's accumulated link time stays
  under the fabric charge on a fresh hop; past that depth, which falls out of the
  machine's bandwidth ratio, a hop off a holder is the better answer.
* :data:`SPREAD` -- nothing, so the soonest source wins and a holder beats a peer still
  fetching.

No queue is charged in the score. How many readers are already behind a peer is read
here only as the cap that stops offering it at all; what a busy source *costs* is the
plane's trade to make over this ranking (:mod:`dedup_sim.control.routing`).

There is no burst loop, no reader list and no knowledge of how many readers there will
be: the tree is one requester at a time, and each reader's read-through is what
releases the next reader's withheld answer (:mod:`dedup_sim.data`).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from proposed import Key, KeySelector, Selection, VolumeId

from ._answer import holders
from ._view import FanoutView

__all__ = ["Candidates", "CHAIN", "SPREAD"]

#: Measured: on the default profile :data:`CHAIN` keeps a cap-1 chain 1x seven readers
#: deep and a cap-2 tree 1x well past sixty (``dedup_sim/tests/test_dedup.py``).
CHAIN = 10.0
SPREAD = 0.0


class Candidates(KeySelector):
    """Holders and planned peers as one pool, each priced in seconds.

    Args:
        fabric: what a second of somebody else's link time is worth against a second of
            the requester's own wait (:data:`CHAIN`, :data:`SPREAD`).
        payload_bytes: bytes to price a hop for. The directory reports what a volume
            holds and not how big it is, so the default nominal byte prices a hop at
            little more than its link's latency; a run that knows the size can say so,
            and the bandwidth term then sharpens the ratio between tiers.

    A peer is offered only while it still **owes** the key: from the ask that promises
    it (:class:`~dedup_sim.control._sensor.Asked`) until its read-through lands. One
    that published and has since evicted owes nothing and holds nothing, so it is no
    candidate.
    """

    name = "candidates"
    sensors = (FanoutView,)

    def __init__(self, fabric: float = CHAIN, payload_bytes: int = 1) -> None:
        self.fabric = fabric
        self.payload_bytes = payload_bytes

    async def select(self, keys: Sequence[Key], requester: str) -> Selection:
        """Everything that could serve every one of ``keys``, scored; else abstain."""
        fanout = self.view.fanout
        located = self.view.locate(keys)
        holds = set(holders(located, keys[0]))
        for key in keys[1:]:
            holds &= set(holders(located, key))
        routes, queued = fanout.routes(), fanout.named()
        # Discounted from the cap below: a second ask costs no slot, so a reader
        # re-asking after an eviction is not shut out by its own place in the queue.
        mine = fanout.planned(requester)
        priced: List[Tuple[VolumeId, float]] = []
        # Memo for the walk below: one entry per volume reached, so a branch is walked
        # once per decision and not once per candidate. The pool needs no order of its
        # own -- the score orders it, and the fold reading it is total (id the last
        # tie-break).
        waits: Dict[VolumeId, Optional[float]] = {}
        for volume in holds | routes.keys():
            # Never a source for itself.
            if volume == requester:
                continue
            wait = self._wait(volume, keys, holds, routes, waits)
            if wait is None:
                continue
            behind = queued.get(volume, 0) - (volume == mine)
            if wait and behind >= fanout.cap:
                # A peer at its cap: the ceiling on how wide the tree fans out, and the
                # only thing the cap does. A holder has no such limit; its price is
                # what keeps readers off it.
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

        A peer's wait is its own source's wait plus the link between the two, so a
        branch costs what its links cost rather than a hop count, and a peer two cheap
        hops away can be sooner than one over a slow link. Every step up the branch must
        still be coming: a volume that owes nothing and holds nothing has no copy on the
        way.

        Missing: elapsed time is not subtracted, so a fetch already in flight is priced
        as if it started now and its peer reads as slightly further off than it is.
        Exact for a synchronized burst, where every decision is made at one instant.

        Every volume the walk passes is answered on the way back down (``waits``), so a
        chain of ``m`` readers costs ``m`` steps in total, not ``m`` per reader.
        """
        fanout = self.view.fanout
        branch: List[VolumeId] = []
        while True:
            # Priced by an earlier candidate's walk.
            if volume in waits:
                wait = waits[volume]
                break
            # A holder waits for nothing.
            if volume in holds:
                wait = waits[volume] = 0.0
                break
            # Owes nothing, or more steps than there are edges -- a walk that cannot
            # terminate is a cycle, and this is what stops following it.
            if len(branch) > len(routes) or not fanout.owes(
                [(volume, key) for key in keys]
            ):
                wait = waits[volume] = None
                break
            branch.append(volume)
            source = routes.get(volume)
            # Owed the key, but nothing is planned to feed it: no copy on the way.
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
