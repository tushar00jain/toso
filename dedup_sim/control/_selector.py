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
(:meth:`~proposed.environment.Environment.read_time`): no tier arithmetic, and no units to
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

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Unpack

from proposed import Key, KeySelector, Selection, VolumeId

from ._sensor import DedupDirectorySensor, FanoutSensor

__all__ = ["Candidates", "Holders", "CHAIN", "SPREAD"]

#: Measured: on the default profile :data:`CHAIN` keeps a cap-1 chain 1x seven readers
#: deep and a cap-2 tree 1x well past sixty (``dedup_sim/tests/test_dedup.py``).
CHAIN = 10.0
SPREAD = 0.0


class Holders(KeySelector[Unpack[Tuple[()]]]):
    """Every live holder, once, in directory order."""

    sensors = (DedupDirectorySensor,)

    def select(
        self, keys: Sequence[Key], requester: str
    ) -> Selection[Unpack[Tuple[()]]]:
        located = self.sensor(DedupDirectorySensor).locate(keys)
        return Selection.of(
            tuple(
                dict.fromkeys(
                    source for by_source in located.values() for source in by_source
                )
            )
        )


class Candidates(KeySelector[float]):
    """Holders and planned peers as one pool, each priced in seconds.

    Args:
        fabric: what a second of somebody else's link time is worth against a second of
            the requester's own wait (:data:`CHAIN`, :data:`SPREAD`).
        payload_bytes: bytes to price a hop for. The directory reports what a volume
            holds and not how big it is, so the default nominal byte prices a hop at
            little more than its link's latency; a run that knows the size can say so,
            and the bandwidth term then sharpens the ratio between tiers.

    A peer is offered only while its promised requests cover this request: from the ask
    that promises them (:class:`~dedup_sim.control._sensor.Asked`) until its
    read-through lands. One that published and has since evicted owes nothing and holds
    nothing, so it is no candidate.
    """

    sensors = (DedupDirectorySensor, FanoutSensor)

    def __init__(self, fabric: float = CHAIN, payload_bytes: int = 1) -> None:
        self.fabric = fabric
        self.payload_bytes = payload_bytes

    def select(self, keys: Sequence[Key], requester: str) -> Selection[float]:
        """Every source with a relevant key region, scored; else abstain."""
        fanout = self.sensor(FanoutSensor)
        directory = self.sensor(DedupDirectorySensor)
        requested = tuple(directory.plan(requester).values())
        located = directory.locate(keys)
        candidates, pending = directory.serving_sources(requested)
        queued = fanout.named()
        located_by_key: Dict[str, Mapping[VolumeId, Any]] = dict(located)
        # Discounted from the cap below: a second ask costs no slot, so a reader
        # re-asking after an eviction is not shut out by its own place in the queue.
        mine = set(fanout.planned(requester))
        priced: List[Tuple[VolumeId, float]] = []
        waits: Dict[VolumeId, Optional[float]] = {}
        in_flight = directory.in_flight()
        live = candidates - pending
        for volume in candidates:
            # Never a source for itself.
            if volume == requester:
                continue
            wait = (
                0.0
                if volume in live
                else self._wait(
                    volume,
                    fanout,
                    in_flight,
                    directory,
                    located_by_key,
                    waits,
                    set(),
                )
            )
            if wait is None:
                continue
            behind = queued.get(volume, 0) - (volume in mine)
            if volume in pending and behind >= fanout.cap:
                # A peer at its cap: the ceiling on how wide the tree fans out, and the
                # only thing the cap does. A holder has no such limit; its price is
                # what keeps readers off it.
                continue
            hop = self.env.read_time(volume, requester, self.payload_bytes)
            priced.append((volume, wait + hop + self.fabric * hop))
        if not priced:
            return Selection.abstain()
        return Selection.priced(priced)

    def _wait(
        self,
        volume: VolumeId,
        fanout: FanoutSensor,
        in_flight: Set[VolumeId],
        directory: DedupDirectorySensor,
        located: Dict[Key, Mapping[VolumeId, Any]],
        waits: Dict[VolumeId, Optional[float]],
        visiting: Set[VolumeId],
    ) -> Optional[float]:
        if volume in waits:
            return waits[volume]
        if volume not in in_flight:
            waits[volume] = None
            return None
        if volume in visiting:
            waits[volume] = None
            return None
        required = fanout.route_required(volume)
        if not required:
            waits[volume] = None
            return None
        visiting.add(volume)
        pending = fanout.route_pending(volume)
        requests = tuple(directory.plan(volume).values())
        arrivals: List[float] = []
        for source, expected in required.items():
            required_keys = {key for key, _region in expected}
            for key in required_keys:
                if key not in located:
                    located.update(directory.locate_live([key]))
            ready = directory.covers(requests, {source: expected}, located)
            if ready:
                wait = 0.0
            elif source in pending and source in in_flight:
                wait = self._wait(
                    source,
                    fanout,
                    in_flight,
                    directory,
                    located,
                    waits,
                    visiting,
                )
            else:
                wait = None
            if wait is None:
                waits[volume] = None
                visiting.remove(volume)
                return None
            arrivals.append(
                wait + self.env.read_time(source, volume, self.payload_bytes)
            )
        visiting.remove(volume)
        waits[volume] = max(arrivals)
        return waits[volume]
