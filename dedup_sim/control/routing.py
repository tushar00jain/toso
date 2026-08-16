"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

One member, and it is everything a reader asks: *which volumes serve this key for me,
and when are they usable* (:meth:`Dedup.sources`). That the put landed is not a second
question and not a report to this plane: it is one action a reader commits
(:class:`proposed.dispatch.Stored`), folded by this plane's own state
(:attr:`Dedup.dispatcher`).

The ranking is :mod:`dedup_sim.control._selector`; what a decision is made of once the
chain has named a head is :mod:`dedup_sim.control._answer`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from proposed import ControlPlane, DecisionLog, Dispatcher, Key, Selection
from proposed.selector import (
    Balance, Dims, FirstMatch, Folded, NaiveKeySelector, Selector, Sort,
)

from ._answer import committed
from ._selector import Candidates, CHAIN, SPREAD
from ._sensor import Asked, FanoutSensor
from ._view import DedupView

__all__ = ["Dedup"]


def _soonest(dims: Dims) -> float:
    """Spread's fold: with the fabric free the score is one read, so a reader queued
    behind ``queued`` others waits that many times over."""
    score, queued = dims
    return score * (1 + queued)


class Dedup(ControlPlane):
    """Dedup's whole control plane: one member over one source chain.

    Args:
        fanout_cap: peers one source may be planned to feed -- 1 a chain, >= 2 a
            shallow tree, 1x fabric either way.
        spread: price the fabric at nothing
            (:data:`~dedup_sim.control._selector.SPREAD`) and fold the queue at a
            source into the score instead (:func:`_soonest`). Off by default, and a
            trade: two readers of one key go to two replicas rather than chaining the
            second behind the first, which costs a second hop off the holders and buys
            back the wallclock of that chain hop.
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
        self._spread = spread
        self._trace = trace
        # Built in attach().
        self.view: Optional[DedupView] = None
        self.dispatcher: Optional[Dispatcher] = None
        self._chain: Optional[Selector[Sequence[Key]]] = None

    def attach(self, view: Any) -> None:
        """Compose this plane's one sensor, and attach the chain that senses it.

        The sensor is built here, never accepted from a caller: two planes sharing one
        would each answer for the other's decisions -- a requester handed a peer the
        other plane planned, then waiting on a put only the other plane hears about.

        Sorted rather than cut to the winner: a reader reads its preference down, so a
        source ranked behind the head still serves the read if the head evicted the key
        before the reader got there.
        """
        sensor = FanoutSensor(fanout_cap=self._cap)
        self.view = view.derived(DedupView, fanout=sensor, load=sensor)
        self.dispatcher = Dispatcher()
        self.dispatcher.compose(sensor)
        # Which volumes serve a read: every holder of the key and every peer already
        # planned to hold it, priced together in seconds, so which one wins is
        # arithmetic off the score rather than an order the caller has to know.
        self._chain = Sort(FirstMatch([
            Folded(
                Balance(Candidates(SPREAD if self._spread else CHAIN)),
                _soonest if self._spread else None,
            ),
            # Tail: an unroutable ask gets the directory's own answer, not a hole.
            NaiveKeySelector(),
        ])).attach(self.view)

    # -- what a reader asks -------------------------------------------------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection:
        """Which volumes serve ``keys`` for ``requester``, once they are usable."""
        # The wait is spent here, not handed back: a caller that read before these
        # sources held the key would go to a volume with nothing to serve.
        return await (await self._decide(keys, requester)).settled()

    async def _decide(self, keys: Sequence[Key], requester: str) -> Selection:
        """The whole decision with the gate unspent, awaitable without parking.

        The chain keys each source ``(score, queued)``: the seconds
        :class:`~dedup_sim.control._selector.Candidates` priced it at, then the readers
        :class:`~proposed.selector.Balance` found already routed to it. Compared as they
        stand, the fabric decides and the queue only settles a tie the score cannot:
        queueing behind a peer is already in that peer's own wait, so charging it again
        would price one delay twice. The tie-break keeps two replicas of one key
        alternating rather than falling back to id order.
        """
        keys = list(keys)
        # Asking is what makes this requester a peer, so its debt is dispatched before
        # the ranking is consulted -- that debt is what bounds the wait on whichever
        # source is named. Dispatched without suspending, so the debt and the decision
        # priced against it are one turn.
        self.dispatcher.dispatch_sync(Asked(requester, tuple(keys)))
        ranking = self._chain.select(keys, requester)
        return committed(
            self.view, self.dispatcher, keys, requester, ranking, self._trace
        )
