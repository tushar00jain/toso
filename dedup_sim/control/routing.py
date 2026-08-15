"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

One member, and it is everything a reader asks: *which volumes serve this key for me,
and when are they usable* (:meth:`Dedup.sources`). A reader asks, reads from what it
was told, and puts. That the put landed is not a second question and not a report to
this plane: it is one action a reader commits (:class:`proposed.dispatch.Stored`), and
this plane's own state is what folds it (:attr:`Dedup.dispatcher`).

The deciding is the source chain this plane holds
(:mod:`dedup_sim.control._selector`); what is left here is the service boundary.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from proposed import ControlPlane, DecisionLog, Dispatcher, Key, Selection
from proposed.selector import FirstMatch, NaiveKeySelector

from ._selector import HolderRanking, PlannedPeer
from ._sensor import FanoutSensor
from ._view import FanoutView

__all__ = ["Dedup"]


class Dedup(ControlPlane):
    """Dedup's whole control plane: one member over one source chain.

    Args:
        fanout_cap: peers one source may be planned to feed -- 1 a chain, >= 2 a
            shallow tree, 1x fabric either way. The sensor's knob
            (:class:`~dedup_sim.control._sensor.FanoutSensor`), spent in
            :meth:`attach`.
        trace: optional :class:`~proposed.selector.DecisionLog` for each routing
            decision. Records only; no metric turns on it.
    """

    name = "dedup"

    def __init__(
        self, *, fanout_cap: int = 1, trace: Optional[DecisionLog] = None
    ) -> None:
        self._cap = fanout_cap
        self._trace = trace
        # All built in attach(), where the ports the chain senses through arrive.
        self.view: Optional[FanoutView] = None
        self.dispatcher: Optional[Dispatcher] = None
        self._chain: Optional[FirstMatch[int]] = None

    def attach(self, view: Any) -> None:
        """Compose this plane's one sensor, and attach the chain that senses it.

        The sensor is built here and never accepted from a caller, because one held by
        two planes would have each answering for the other's decisions: a requester
        would be handed a slot the other plane planned and then wait on a put only the
        other plane is told about. The links reach it through the view each declares
        (:class:`~dedup_sim.control._view.FanoutView`), so
        :meth:`~proposed.selector.FirstMatch.attach` is the whole of the wiring and no
        link is handed a sensor.

        The dispatcher is built here for the same reason and composed with that sensor
        as its reducer: what a landed put moves is this plane's own state, so this is
        what knows to fold it. The links get the dispatcher too, because a withheld
        answer waits on its commit and on nothing else.

        The order is load-bearing: :class:`~dedup_sim.control._selector.PlannedPeer`
        spends a slot when it answers, so it goes at the head, and the
        :class:`~proposed.selector.NaiveKeySelector` tail is what makes an unroutable
        ask the directory's own answer rather than a hole.
        """
        self.view = view.derived(
            FanoutView, fanout=FanoutSensor(fanout_cap=self._cap)
        )
        self.dispatcher = Dispatcher()
        self.dispatcher.compose(self.view.fanout)
        self._chain = FirstMatch([
            PlannedPeer(self.dispatcher, trace=self._trace),
            HolderRanking(self.dispatcher, trace=self._trace),
            NaiveKeySelector(),
        ]).attach(self.view)

    # -- what a reader asks -------------------------------------------------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection[int]:
        """Which volumes serve ``keys`` for ``requester``, once they are usable.

        Two things, because a caller reaches this plane as a service: the decision and
        the wait. A decision that routes a requester to a peer still fetching carries a
        gate, and a gate is a closure -- it cannot travel to whoever asked, so it is
        spent here (:meth:`~proposed.selector.Selection.settled`) and what goes back is
        the ranking. Which is also what a caller wants: it is about to read from these
        sources, so an answer released before they hold the key would send it to a
        volume with nothing to serve.
        """
        selection = await self._chain.select(list(keys), requester)
        return await selection.settled()
