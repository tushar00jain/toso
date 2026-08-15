"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

One member, and it is everything a reader asks: *which volumes serve this key for me,
and when are they usable* (:meth:`Dedup.sources`). A reader asks, reads from what it
was told, and puts. That the put landed is not a second question and not a report to
this plane: it is one action a reader commits (:class:`proposed.dispatch.Stored`), and
this plane's own state is what folds it (:attr:`Dedup.dispatcher`).

The deciding is the source chain this plane holds
(:mod:`dedup_sim.control._selector`); what a decision is *made of* once that chain has
named a head is :mod:`dedup_sim.control._answer`. What is left here is the service
boundary and the order of the chain.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from proposed import ControlPlane, DecisionLog, Dispatcher, Key, Selection
from proposed.selector import Balance, Dims, FirstMatch, Fold, NaiveKeySelector

from ._answer import committed
from ._selector import Candidates, CHAIN, SPREAD
from ._sensor import FanoutSensor
from ._view import DedupView

__all__ = ["Dedup"]


def _soonest(dims: Dims) -> float:
    """Spread's fold: every reader already routed at a source costs one more read.

    The score of a source the fabric is charged nothing for is what reading it costs
    once, so a reader waiting its turn behind ``queued`` others waits that many times
    over -- which is what sends the second reader of a key to a second replica instead
    of queueing it behind the first.
    """
    score, queued = dims
    return score * (1 + queued)


class Dedup(ControlPlane):
    """Dedup's whole control plane: one member over one source chain.

    Args:
        fanout_cap: peers one source may be planned to feed -- 1 a chain, >= 2 a
            shallow tree, 1x fabric either way. The sensor's knob
            (:class:`~dedup_sim.control._sensor.FanoutSensor`), spent in
            :meth:`attach`.
        spread: price the fabric a read occupies at nothing
            (:data:`~dedup_sim.control._selector.SPREAD`) and fold the queue at a
            source into the score instead (:func:`_soonest`), so a reader takes
            whatever is soonest for it. Off by default, and a real trade rather than a
            refinement: it is what sends two readers of one key to two replicas of it
            instead of chaining the second behind the first, which costs a second hop
            off the holders and buys the wallclock of that chain hop back.
        trace: optional :class:`~proposed.selector.DecisionLog` for each routing
            decision. Records only; no metric turns on it.
    """

    name = "dedup"

    def __init__(
        self,
        *,
        fanout_cap: int = 1,
        spread: bool = False,
        trace: Optional[DecisionLog] = None,
    ) -> None:
        self._cap = fanout_cap
        self._fabric = SPREAD if spread else CHAIN
        #: How the chain's key is read, stamped on its answers by the
        #: :class:`~proposed.selector.Balance` that appends the last dimension of it,
        #: so :meth:`_decide` folds without naming one. ``None`` -- the chain preset --
        #: compares the dimensions as they stand.
        self._fold: Optional[Fold] = _soonest if spread else None
        self._trace = trace
        # All built in attach(), where the ports the chain senses through arrive.
        self.view: Optional[DedupView] = None
        self.dispatcher: Optional[Dispatcher] = None
        self._chain: Optional[FirstMatch[float]] = None

    def attach(self, view: Any) -> None:
        """Compose this plane's one sensor, and attach the chain that senses it.

        The sensor is built here and never accepted from a caller, because one held by
        two planes would have each answering for the other's decisions: a requester
        would be handed a peer the other plane planned and then wait on a put only the
        other plane is told about. It is composed under both of the names a decision
        reads it by (:class:`~dedup_sim.control._view.DedupView`), and the links reach
        it through the view each declares, so
        :meth:`~proposed.selector.FirstMatch.attach` is the whole of the wiring and no
        link is handed a sensor.

        The dispatcher is built here for the same reason and composed with that sensor
        as its reducer: what a landed put moves is this plane's own state, so this is
        what knows to fold it.

        The chain is two links and the second is a tail: one ranking over every volume
        that could serve the read (:class:`~dedup_sim.control._selector.Candidates`)
        under a :class:`~proposed.selector.Balance` that appends the readers already
        routed at each source, and a :class:`~proposed.selector.NaiveKeySelector`
        behind it so an unroutable ask is the directory's own answer rather than a
        hole. Holders and peers are priced together rather than asked in turn, so which
        one wins is arithmetic a caller can read off the score instead of an order it
        has to know.
        """
        sensor = FanoutSensor(fanout_cap=self._cap)
        self.view = view.derived(DedupView, fanout=sensor, load=sensor)
        self.dispatcher = Dispatcher()
        self.dispatcher.compose(sensor)
        self._chain = FirstMatch([
            Balance(Candidates(self._fabric), self._fold),
            NaiveKeySelector(),
        ]).attach(self.view)

    # -- what a reader asks -------------------------------------------------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection[float]:
        """Which volumes serve ``keys`` for ``requester``, once they are usable.

        Three things, because a caller reaches this plane as a service: the ranking, the
        decision made out of it (:func:`~dedup_sim.control._answer.committed` -- the
        route recorded and the gate hung on whichever head survived the chain), and the
        wait. A gate is a closure, so it cannot travel to whoever asked: it is spent
        here (:meth:`~proposed.selector.Selection.settled`) and what goes back is the
        ranking. Which is also what a caller wants: it is about to read from these
        sources, so an answer released before they hold the key would send it to a
        volume with nothing to serve.
        """
        return await (await self._decide(keys, requester)).settled()

    async def _decide(self, keys: Sequence[Key], requester: str) -> Selection[float]:
        """The whole decision, gate unspent: the chain's scores folded into an order,
        then what is committed out of it.

        The fold is one call, and it is the only ordering in the decision: the chain
        this plane built in :meth:`attach` keys each source ``(score, queued)`` -- the
        seconds :class:`~dedup_sim.control._selector.Candidates` priced it at, then the
        readers :class:`~proposed.selector.Balance` found already routed to it -- and
        neither link sorts. Compared as they stand (the chain preset), the fabric
        decides and the queue only settles a tie the score cannot: queueing behind a
        peer is already in that peer's own wait, so charging it again would price one
        delay twice, while two replicas of one key keep alternating rather than
        reverting to id order. ``spread`` blends the two instead (:func:`_soonest`).

        Ordered rather than cut to the winner
        (:meth:`~proposed.selector.Selection.max`), because what a reader is handed is a
        preference and it reads down that list: a source the ranking placed behind the
        head is what serves the read if the head has evicted the key by the time the
        reader gets there (:func:`~proposed.selector.prefer`).

        Separate from :meth:`sources` only because the gate is the answer's last step
        and not part of forming it: this is what a caller inspecting a decision it is
        not going to read from can await without parking on a peer's read-through.
        """
        keys = list(keys)
        ranking = (await self._chain.select(keys, requester)).sort()
        return committed(
            self.view, self.dispatcher, keys, requester, ranking, self._trace
        )
