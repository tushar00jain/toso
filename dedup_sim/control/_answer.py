"""What a dedup decision answers with once a head is named: :func:`committed`.

A link ranks; this is what the plane does with the ranking it gets back
(:meth:`dedup_sim.control.routing.Dedup.sources`) -- record the route, and withhold the
answer until that head is usable. It belongs to the plane and not to the link because a
ranking is in no order until this plane folds it and a stage above the link may have
added a dimension first (:class:`~proposed.selector.Balance`): a link that recorded its
own head would record a source the requester never reads from, and gate on it.

Nothing here suspends, which is where the chain's own claim stops being one: the
directory read cannot (:meth:`~proposed.deployment.Controller.locate_raw`) and neither
does building a gate (:meth:`~proposed.dispatch.Dispatcher.gate`), so a whole decision
-- the chain and this -- runs to completion before the next requester's starts. That is
the serialized mailbox a real controller would give it, and what makes the sensor's
read-modify-writes safe without a lock.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Hashable, List, Optional, Sequence

from proposed import DecisionLog, Dispatcher, Key, Selection

from ._view import FanoutView

__all__ = ["committed", "holders"]


def holders(located: Dict[str, Dict[str, Any]], key: str) -> List[str]:
    """Volumes holding ``key`` in a :meth:`~proposed.view.View.locate` answer.

    In directory order, and empty when nobody holds it -- a missing key is absent
    from the answer rather than an error in it. Here because reading a located map is
    this plane's arithmetic, not a member the store owes anyone.
    """
    return list(located.get(key, {}))


def committed(
    view: FanoutView,
    commits: Dispatcher,
    keys: Sequence[Key],
    requester: str,
    ranking: Selection[Any],
    trace: Optional[DecisionLog] = None,
) -> Selection[Any]:
    """``ranking``, routed to its head and gated until that head is usable.

    Which of three shapes an answer takes is the whole of when waiting is safe, because
    a gate nothing will ever open hangs the requester behind it. What bounds one is the
    debt the sensor tracks: a routed requester is going to read the key through, so from
    the moment it asks it *owes* that registration. A source that owes nothing has to
    hold the key already, and one that holds nothing and owes nothing is no longer a
    source at all.

    The head is what the route is recorded against, what the gate is opened on, and what
    a caller preferring this ranking reads from; whatever is ranked behind it rides
    through with its prices. A ranking naming no head in particular -- an abstention, or
    the directory's own answer -- comes back untouched: there is nothing to route to.
    """
    source = ranking.head
    if source is None:
        return ranking
    fanout = view.fanout
    if fanout.planned(requester) != source:
        # Re-asked and unmoved is not a decision: recording it again would trace a
        # route nothing changed.
        fanout.route(requester, source)
        if trace is not None:
            trace.record(view.now(), "route", f"{requester} <- {source}")
    facts = [(source, key) for key in keys]
    if fanout.owes(facts):
        # It owes every key, so the wait is bounded by its read-through. The gate is
        # still opened on a directory read, because it may already hold a key it is
        # about to republish -- then there is nothing to wait for.
        return replace(ranking, ready=commits.gate(
            lambda: len(_registered(view, facts)) == len(facts)
        ))
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


def _registered(view: FanoutView, facts: Sequence[Hashable]) -> List[Hashable]:
    """Which of these ``(volume, key)`` pairs the directory holds *now*.

    The truth a readiness gate is opened against, read rather than remembered, and read
    afresh every time an answer is formed *and* at every commit a parked requester wakes
    on (:meth:`~proposed.dispatch.Dispatcher.gate`): volumes evict, so a peer that
    registered the key and later dropped it for a newer version is a peer the next
    requester has to wait for again. Hence the live read
    (:meth:`~proposed.view.View.locate_live`): a gate is correct only against the
    directory *now*, and one opened against a directory read taken before the
    registration landed would park its requester forever.

    The cross-slice read, and the only one: what the fan-out owes is one reducer's
    state and what the directory holds is another's, and a decision is where the two are
    read together (:mod:`proposed.dispatch`).
    """
    located = view.locate_live([key for _volume, key in facts])
    return [
        (volume, key)
        for volume, key in facts
        if volume in holders(located, key)
    ]
