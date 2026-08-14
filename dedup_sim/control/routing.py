"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

Two members, and between them everything a reader needs: *which volumes serve this
key for me, and when are they usable* (:meth:`Dedup.sources`), and *my read-through
has landed* (:meth:`Dedup.published`). A reader asks, reads from what it was told,
puts, and says so. Nothing else is decided here and nothing else is told.

The deciding is the source chain this plane holds
(:mod:`dedup_sim.control._selector`); what is left here is the service boundary.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from proposed import ControlPlane, DecisionLog, Key, Selection
from proposed.selector import FirstMatch, NaiveKeySelector

from ._selector import HolderRanking, PlannedPeer
from ._sensor import FanoutSensor
from ._view import FanoutView

__all__ = ["Dedup"]


class Dedup(ControlPlane):
    """Dedup's whole control plane: two members over one source chain.

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
        # Both built in attach(), where the ports the chain senses through arrive.
        self.view: Optional[FanoutView] = None
        self._chain: Optional[FirstMatch[int]] = None

    def attach(self, view: Any) -> None:
        """Compose this plane's one sensor, and attach the chain that senses it.

        The sensor is built here and never accepted from a caller, because one held by
        two planes would have each answering for the other's decisions: a requester
        would be handed a slot the other plane planned and then wait on a put only the
        other plane is told about. The links reach it through the view
        (:class:`~dedup_sim.control._view.FanoutView`), so
        :meth:`~proposed.selector.FirstMatch.attach` is the whole of the wiring and no
        link is handed a sensor.

        The order is load-bearing: :class:`~dedup_sim.control._selector.PlannedPeer`
        spends a slot when it answers, so it goes at the head, and the
        :class:`~proposed.selector.NaiveKeySelector` tail is what makes an unroutable
        ask the directory's own answer rather than a hole.
        """
        self.view = view.derived(
            FanoutView, fanout=FanoutSensor(fanout_cap=self._cap)
        )
        self._chain = FirstMatch([
            PlannedPeer(trace=self._trace),
            HolderRanking(trace=self._trace),
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

    async def published(self, requester: str, keys: Sequence[Key]) -> None:
        """``requester``'s read-through has landed: ``keys`` are on its volume now.

        The other half of this plane's surface, and the reason it can gate at all:
        the data plane reports its own put (:mod:`dedup_sim.data.read_through`), so
        this hears a registration without the directory having to know that anything
        is listening. Folded into the sensor the chain senses, which is where the
        debt it settles is kept.

        Reported *after* the put, so the directory already lists ``requester`` when a
        waiter released here re-reads it. Awaited for the ordering rather than an
        answer: the reporter's next ask must be decided against a plane that has
        folded this one in.
        """
        self.view.fanout.published(requester, keys)
