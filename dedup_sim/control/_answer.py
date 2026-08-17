"""What a dedup decision answers with once a head is named: :func:`committed`.

The route is recorded by the plane, not by the link that ranked: a stage above the link
can still reorder the ranking, so a link recording its own head would record a source
the requester never reads from, and gate on it.

What the fan-out owes is one reducer's state and what the directory holds is another's;
:func:`committed` is the only place the two are read together.

Nothing here suspends: neither the directory read
(:meth:`~proposed.deployment.Controller.locate_raw`) nor building a gate
(:meth:`~proposed.dispatch.Dispatcher.gate`). A whole decision therefore runs to
completion before the next requester's starts, which is what makes the sensor's
read-modify-writes safe without a lock.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Hashable, List, Optional, Sequence

from proposed import DecisionLog, DirectoryDefault, Dispatcher, Key, Selection

from ._view import FanoutView

__all__ = ["committed", "holders"]


def holders(located: Dict[str, Dict[str, Any]], key: str) -> List[str]:
    """Volumes holding ``key``, in directory order; empty when nobody holds it."""
    return list(located.get(key, {}))


def committed(
    view: FanoutView,
    commits: Dispatcher,
    keys: Sequence[Key],
    requester: str,
    ranking: Selection,
    trace: Optional[DecisionLog] = None,
) -> Selection:
    """``ranking``, routed to its head and gated until that head is usable.

    An answer takes one of three shapes, and picking the wrong one hangs the requester
    behind a gate nothing will open. The debt the sensor tracks is what bounds a wait:
    a routed requester is going to read the key through, so from the moment it asks it
    *owes* that registration. A source that owes nothing must hold the key already, and
    one that holds nothing and owes nothing is no source at all.

    The head is what the route is recorded against and what the gate is opened on;
    whatever is ranked behind it rides through as it was ranked. A ranking naming no
    head -- an abstention, or the directory's own answer -- comes back untouched.
    """
    source = ranking.head
    if source is None:
        return ranking
    fanout = view.fanout
    if fanout.planned(requester) != source:
        # Skipped when re-asked and unmoved: no route changed, nothing to trace.
        fanout.route(requester, source)
        if trace is not None:
            trace.record(view.now(), "route", f"{requester} <- {source}")
    facts = [(source, key) for key in keys]
    if fanout.owes(facts):
        # Owes every key: the wait is bounded by its read-through. Still gated on a
        # directory read, since it may already hold a key it is about to republish.
        return replace(ranking, ready=commits.gate(
            lambda: len(_registered(view, facts)) == len(facts)
        ))
    if len(_registered(view, facts)) == len(facts):
        # A pre-existing holder, the ordinary case: usable now, and owing nothing it
        # would never record the facts a gate here waited on.
        return ranking
    # Published, then evicted: holds nothing and owes nothing. A gate would hang and
    # naming it would route a reader to a volume with nothing to serve, so it stops
    # being a source and this requester gets the directory's own answer.
    fanout.retire(requester, source)
    if trace is not None:
        trace.record(view.now(), "retire", f"{source} holds nothing")
    return DirectoryDefault()


def _registered(view: FanoutView, facts: Sequence[Hashable]) -> List[Hashable]:
    """Which of these ``(volume, key)`` pairs the directory holds *now*.

    Read live, and re-read at every commit a parked requester wakes on: volumes evict,
    so a peer that registered the key and later dropped it is one the next requester
    waits for again. A gate opened against a read taken before the registration landed
    parks forever."""
    located = view.locate_live([key for _volume, key in facts])
    return [
        (volume, key)
        for volume, key in facts
        if volume in holders(located, key)
    ]
