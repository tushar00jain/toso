"""1x-fabric dedup routing: :class:`Dedup`, the capability's whole control plane.

Two members, and between them everything a reader needs: *which volumes serve this
key for me, and when are they usable* (:meth:`Dedup.sources`), and *my read-through
has landed* (:meth:`Dedup.published`). A reader asks, reads from what it was told,
puts, and says so. Nothing else is decided here and nothing else is told.

The deciding is the ranking this plane holds
(:class:`~dedup_sim.control._source.PeerKeySelector`); what is left here is the
service boundary.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from proposed import ControlPlane, DecisionLog, Key, Selection

from ._source import PeerKeySelector

__all__ = ["Dedup"]


class Dedup(ControlPlane):
    """Dedup's whole control plane: two members over one ranking.

    Args:
        fanout_cap: peers one source may be planned to feed -- 1 a chain, >= 2 a
            shallow tree, 1x fabric either way. The ranking's knob
            (:class:`~dedup_sim.control._source.PeerKeySelector`), passed down.
        trace: optional :class:`~proposed.selector.DecisionLog` for each routing
            decision. Records only; no metric turns on it.
    """

    name = "dedup"

    def __init__(
        self, *, fanout_cap: int = 1, trace: Optional[DecisionLog] = None
    ) -> None:
        #: The ranking behind both members, built from this plane's own knobs.
        self.selector = PeerKeySelector(fanout_cap=fanout_cap, trace=trace)

    def attach(self, view: Any) -> None:
        """Hand the view down to the ranking; nothing on this side senses.

        It ranks nothing but holders, so the transfer-cost half of the port goes
        unused: which peer is nearest is a locality question the view answers.
        """
        self.selector.attach(view)

    # -- what a reader asks -------------------------------------------------- #
    async def sources(self, keys: Sequence[Key], requester: str) -> Selection[None]:
        """Which volumes serve ``keys`` for ``requester``, once they are usable.

        Two things, because a caller reaches this plane as a service: the decision and
        the wait. A decision that routes a requester to a peer still fetching carries a
        gate, and a gate is a closure -- it cannot travel to whoever asked, so it is
        spent here (:meth:`~proposed.selector.Selection.settled`) and what goes back is
        the ranking. Which is also what a caller wants: it is about to read from these
        sources, so an answer released before they hold the key would send it to a
        volume with nothing to serve.
        """
        selection = await self.selector.select(list(keys), requester)
        return await selection.settled()

    async def published(self, requester: str, keys: Sequence[Key]) -> None:
        """``requester``'s read-through has landed: ``keys`` are on its volume now.

        The other half of this plane's surface, and the reason it can gate at all:
        the data plane reports its own put (:mod:`dedup_sim.data.read_through`), so
        this hears a registration without the directory having to know that anything
        is listening.

        Reported *after* the put, so the directory already lists ``requester`` when a
        waiter released here re-reads it. Awaited for the ordering rather than an
        answer: the reporter's next ask must be decided against a plane that has
        folded this one in.
        """
        self.selector.published(requester, keys)
