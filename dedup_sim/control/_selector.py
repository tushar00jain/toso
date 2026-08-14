"""The links that fold a reader into the read-through tree, as key selectors.

With no routing, ``m`` readers of one key all ``locate_volumes`` before anyone
finishes, so all ``m`` are told "the origin" and each pulls from it -- ``m x`` fabric.
Readers ask this chain in order instead: :class:`PlannedPeer` hands out the peers
already planned to hold the key, :class:`HolderRanking` opens a tree at the volumes
that hold it now, and the :class:`~proposed.selector.NaiveKeySelector` behind them
leaves the answer to the directory (:class:`~dedup_sim.control.routing.Dedup`).

Both links call :func:`_once_usable` for the answer's shape rather than inheriting it:
what a source is worth waiting for follows from what it owes, not from which link
picked it.

No link suspends while it decides. The directory read cannot
(:meth:`~proposed.deployment.Controller.locate_raw`) and neither does building a gate,
so a decision runs to completion before the next requester's starts -- the serialized
mailbox a real controller would give it, and what makes the sensor's read-modify-writes
safe without a lock. A read that could suspend would turn its slot queue into a
check-then-act race, and the fan-out cap would be exceeded rather than enforced.

There is no burst loop, no reader list and no knowledge of how many readers there will
be. The tree is assigned one requester at a time as they ask, and the chain *executes*
because each reader's read-through put releases the next reader's withheld answer -- an
emergent property of the data plane's registration, not a schedule this module runs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import (
    Any, Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar,
)

from proposed import DecisionLog, Key, KeySelector, locality, Selection, VolumeId

from ._sensor import FanoutSensor
from ._view import FanoutView

__all__ = ["PlannedPeer", "HolderRanking"]

#: The price a link here answers in, left open on the one that quotes none
#: (:class:`PlannedPeer`, whose payload is empty). Every link of one chain answers in
#: the same terms, and an empty payload is a payload in any of them.
_P = TypeVar("_P")


def _holders(located: Dict[str, Dict[str, Any]], key: str) -> List[str]:
    """Volumes holding ``key`` in a :meth:`~proposed.view.View.locate` answer.

    In directory order, and empty when nobody holds it -- a missing key is absent
    from the answer rather than an error in it. Local because reading a located map
    is this chain's arithmetic, not a member the store owes anyone.
    """
    return list(located.get(key, {}))


def _registered(view: FanoutView, facts: Sequence[Hashable]) -> List[Hashable]:
    """Which of these ``(volume, key)`` pairs the directory holds *now*.

    The truth a readiness gate is opened against, read rather than remembered, and read
    afresh every time an answer is formed: volumes evict, so a peer that registered the
    key and later dropped it for a newer version is a peer the next requester has to
    wait for again. Hence the live read
    (:meth:`~proposed.view.View.locate_live`): a gate is correct only against the
    directory *now*, and one opened against a directory read taken before the
    registration landed would park its requester forever.
    """
    located = view.locate_live([key for _volume, key in facts])
    return [
        (volume, key)
        for volume, key in facts
        if volume in _holders(located, key)
    ]


def _fold(
    view: FanoutView,
    fanout: FanoutSensor,
    requester: str,
    source: str,
    trace: Optional[DecisionLog],
) -> None:
    """Record the decision: ``requester`` reads from ``source`` from now on."""
    fanout.route(requester, source)
    if trace is not None:
        trace.record(view.now(), "route", f"{requester} <- {source}")


async def _once_usable(
    view: FanoutView,
    fanout: FanoutSensor,
    keys: Sequence[Key],
    requester: str,
    ranking: Selection[_P],
    trace: Optional[DecisionLog],
) -> Selection[_P]:
    """``ranking`` -- which must name a head -- gated until that head is usable.

    Which of three shapes an answer takes is the whole of when waiting is safe, because
    a gate nothing will ever open hangs the requester behind it. What bounds one is the
    debt the sensor tracks: a routed requester is going to read the key through, so from
    the moment it asks it *owes* that registration. A source that owes nothing has to
    hold the key already, and one that holds nothing and owes nothing is no longer a
    source at all.

    The head is what the gate is opened on and what a caller preferring this ranking
    reads from; whatever is ranked behind it rides through with its prices.
    """
    source = ranking.head
    facts = [(source, key) for key in keys]
    if fanout.owes(facts):
        # It owes every key, so the wait is bounded by its read-through. Still a
        # directory read, because it may already hold a key it is about to republish --
        # then there is nothing to wait for.
        async def registered(wanted: Sequence[Hashable]) -> List[Hashable]:
            # A readiness probe is awaited because the truth it reads may live
            # somewhere that travels. This one is the local directory.
            return _registered(view, wanted)

        return replace(ranking, ready=await fanout.gate(facts, registered))
    if len(_registered(view, facts)) == len(facts):
        # Owes nothing, so a gate here could outlive the run -- nothing would ever
        # record it. Usable because it holds every key right now, which is the ordinary
        # case: this is how a requester routed to a pre-existing holder is answered.
        return ranking
    # Holds nothing, owes nothing: a peer that published and has since evicted. Waiting
    # would hang and naming it would route a reader to a volume with nothing to serve,
    # so it stops being a source and this requester gets the directory's own answer,
    # once.
    fanout.retire(requester, source)
    if trace is not None:
        trace.record(view.now(), "retire", f"{source} holds nothing")
    return Selection()


class PlannedPeer(KeySelector[_P]):
    """The peer this requester is folded in behind, or the oldest one with a slot.

    A **peer** is a reader that is *about to* hold the key rather than one that does,
    which is why every requester but the first can be answered without a fabric hop.
    Slots are handed out FIFO under the sensor's fan-out cap, so cap 1 builds a chain
    and cap >= 2 a shallow tree, and a requester that already has a route keeps it --
    a second ask costs no slot.

    One source and no price: the requester is folded in behind that one peer, and the
    alternative to reading from it is the origin hop this exists to avoid.

    Answering **spends** a slot
    (:meth:`~dedup_sim.control._sensor.FanoutSensor.claim_slot`), so this belongs at the
    head of a :class:`~proposed.selector.FirstMatch` chain and under no combinator that
    can drop the answer or rank it down (:class:`~proposed.selector.Discount`). There
    spending and using coincide: a link that answers wins the chain, and an abstention
    claimed nothing. Under one that could reject the peer, the slot would be gone and
    that peer would feed one reader fewer than the cap allows.

    Args:
        trace: optional :class:`~proposed.selector.DecisionLog` to record each routing
            decision into. Changes no metric; the link behaves identically with none.
    """

    name = "planned-peer"

    def __init__(self, *, trace: Optional[DecisionLog] = None) -> None:
        self.trace = trace

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[_P]:
        """The peer ``requester`` reads from, once it is usable; else abstain."""
        fanout = self.view.fanout
        fanout.promise(requester, keys)
        source = fanout.planned(requester)
        if source is None:
            source = fanout.claim_slot()
            if source is None:
                return Selection.of([])
            _fold(self.view, fanout, requester, source, self.trace)
        return await _once_usable(
            self.view, fanout, keys, requester, Selection.of([source]), self.trace
        )


class HolderRanking(KeySelector[int]):
    """The volumes that already hold every key, nearest first: where a tree starts.

    Reached only when no peer is planned to hold the key. Every other requester is
    folded in behind a peer, so a reader pulling from a pre-existing holder is one this
    link routed -- which is what keeps ``origin_bytes`` the 1x union whatever the cap.

    Ranked by the key :func:`~proposed.topology.nearest` minimises -- locality tier,
    then volume id -- so the head is the nearest holder and the order is total, hence
    reproducible. The head is the single fabric hop, and the one the answer is gated on;
    the sources behind it are alternatives for a caller that re-ranks, and cost nothing
    to carry because a whole-key read takes the first volume it is offered.

    Priced at the **negated locality tier**, since a price is better when it is higher:
    ``0`` same host, ``-1`` same node, ``-2`` cross-node. One unit is one tier, so a
    :class:`~proposed.selector.Discount` over this with ``max_discount=1`` says load may
    cost a source one tier and no more.

    Abstains when nobody holds every key -- there is nothing to fold a requester in
    behind, and the chain falls through to the directory's own answer.

    Args:
        trace: optional :class:`~proposed.selector.DecisionLog` to record each routing
            decision into. Changes no metric; the link behaves identically with none.
    """

    name = "holder"

    def __init__(self, *, trace: Optional[DecisionLog] = None) -> None:
        self.trace = trace

    async def select(self, keys: Sequence[Key], requester: str) -> Selection[int]:
        """The holders of every one of ``keys``, nearest first; else abstain."""
        fanout = self.view.fanout
        fanout.promise(requester, keys)
        located = self.view.locate(keys)
        candidates = set(_holders(located, keys[0]))
        for key in keys[1:]:
            candidates &= set(_holders(located, key))
        candidates.discard(requester)
        if not candidates:
            return Selection.of([])
        ranked = self._by_locality(candidates, requester)
        _fold(self.view, fanout, requester, ranked[0][0], self.trace)
        return await _once_usable(
            self.view, fanout, keys, requester, Selection.priced(ranked), self.trace
        )

    def _by_locality(
        self, candidates: Sequence[str], requester: str
    ) -> List[Tuple[VolumeId, int]]:
        """``(volume, -tier)`` per candidate, nearest first, volume id breaking the tie.

        Distance is arithmetic on endpoints and needs no directory read, so it comes off
        the topology map (:func:`~proposed.topology.locality`).
        """
        topology = self.view.topology
        here = topology[requester]
        priced = [
            (volume, -int(locality(topology[volume], here))) for volume in candidates
        ]
        return sorted(priced, key=lambda c: (-c[1], c[0]))
