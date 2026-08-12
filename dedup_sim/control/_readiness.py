"""Waiting for a fact that has not happened yet: :class:`Readiness`.

A policy that routes a requester to a source which does not hold the key *yet* has
to withhold its answer until it does. That is a gate per fact still outstanding,
and getting it right is concurrency work, not routing work: the gate must never be
created for something that has already happened, or the waiter is never woken.

So it lives here, in one object, and :mod:`dedup_sim.control.routing` is left with
the tree it is actually shaping. A *fact* is any hashable the caller invents; this
module never interprets one, and it never decides whether one is true either --
that answer is read from wherever the caller says the truth lives, which for dedup
is the real directory (``(volume_id, key)``: "that volume holds that key").

Register interest, then read the truth
--------------------------------------
The obvious order -- ask whether the fact is true, and park on an event if it is
not -- loses a wakeup: the fact can land in the window between the two. The classic
fix is to remember every fact ever recorded, so the "is it true?" question can be
answered from memory with no window at all. That is what this used to do, and it is
wrong for a store whose volumes **evict**: a remembered registration is not a fact
about the present (a new model version displaces the old one, and the volume that
registered the key no longer holds it), and the memory grows with every key
version the run touches, for the life of the run.

So the order is inverted instead. :meth:`gate` creates the event **first** and only
then reads the truth, which means the window has nowhere to hide a wakeup: anything
recorded during the read sets an event that already exists. Nothing is remembered,
because nothing needs to be -- the directory is the record of what is true, and it
is the one record eviction updates.

Why it is safe
--------------
* **No lost wakeup.** Interest is registered before the truth is read, so a
  :meth:`record` landing in between sets the event rather than being missed. The
  gate is only returned for facts that were false *after* the event existed.
* **One fact, one event, however many waiters.** Requesters gating on the same
  fact share it, and one :meth:`record` wakes all of them -- there is one
  registration coming, not one per waiter.
* **Released gates are dropped.** :meth:`record` pops the event as it sets it, so
  the map holds one entry per fact somebody is waiting on right now and nothing
  else -- it does not grow with the run, and a fact recorded with no waiter costs
  nothing. Dropping is safe: every waiter already inside :meth:`gate`'s closure
  holds the event object itself, not a lookup.
* **No stale truth.** A fact is never answered from memory, so a registration that
  has since been evicted cannot release a waiter -- the next :meth:`gate` asks the
  directory again and waits for the key to come back.
* **Release order is deterministic.** One event per fact, awaited in the order the
  caller listed them, and every wakeup goes through the loop's FIFO ready queue --
  so two requesters released by the same fact resume in the order they parked.

What is *not* here: whether waiting for a fact is a good idea. A gate nothing will
ever record hangs the requester behind it forever, and only the caller knows which
facts are coming -- for dedup, the registration a routed peer owes. That check is
in :mod:`dedup_sim.control.routing`, next to the state that answers it.

It is folder-private because dedup is the only capability that waits today. If a
second one needs it, this is what would move into ``proposed`` -- it is the
mechanism behind ``Selection.ready``, and nothing in it knows about dedup.
"""

from __future__ import annotations

import asyncio
from typing import (
    Awaitable, Callable, Dict, Hashable, Iterable, List, Optional, Sequence,
)

__all__ = ["Observed", "Readiness"]

#: Reads which of ``facts`` are true *now*, from wherever the truth lives (for
#: dedup, the real directory). Awaited, because that is a read of live state.
Observed = Callable[[Sequence[Hashable]], Awaitable[Iterable[Hashable]]]


class Readiness:
    """The gates waiting on facts that are not true yet.

    Holds no record of what has happened -- only of who is waiting.
    """

    def __init__(self) -> None:
        # One event per fact somebody is currently waiting on, dropped as soon as
        # it is released. Nothing else is kept: see the module docstring.
        self._gates: Dict[Hashable, asyncio.Event] = {}

    def record(self, fact: Hashable) -> None:
        """``fact`` is now true: release anything waiting on it."""
        gate = self._gates.pop(fact, None)
        if gate is not None:
            gate.set()

    async def gate(
        self, facts: Iterable[Hashable], observed: Observed
    ) -> Optional[Callable[[], Awaitable[None]]]:
        """An awaitable that returns once every one of ``facts`` is true.

        ``observed`` is asked which of them already are -- *after* interest in
        every one of them has been registered, which is what makes the answer
        safe to act on however long it takes to arrive.

        ``None`` when they are all already true -- which the caller should read as
        "no need to wait at all", not as an empty wait.
        """
        wanted = list(facts)
        if not wanted:
            return None
        # Register interest first. From here on a record() for any of these facts
        # has an event to set, so the read below cannot straddle one.
        events = {fact: self._gates.setdefault(fact, asyncio.Event()) for fact in wanted}
        for fact in await observed(wanted):
            self.record(fact)
        pending: List[asyncio.Event] = [
            event for event in events.values() if not event.is_set()
        ]
        if not pending:
            return None

        async def ready() -> None:
            for event in pending:
                await event.wait()

        return ready
