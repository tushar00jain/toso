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
Asking whether a fact is true and *then* parking on an event loses a wakeup: the
fact can land in the window between the two. Remembering every fact ever recorded
closes that window, but it is wrong for a store whose volumes **evict** -- a
remembered registration is not a fact about the present, and the memory grows with
every key version the run touches.

So :meth:`gate` inverts the order: it creates the event **first** and only then
reads the truth, leaving the window nowhere to hide a wakeup, since anything
recorded during the read sets an event that already exists. Nothing is remembered,
because the directory is the record of what is true and the one record eviction
updates.

Why it is safe
--------------
* **No lost wakeup.** Interest is registered before the truth is read, so a
  :meth:`record` landing in between sets the event rather than being missed.
* **One fact, one event, however many waiters.** Requesters gating on the same
  fact share it, and one :meth:`record` wakes all of them.
* **Released gates are dropped.** :meth:`record` pops the event as it sets it, so
  the map holds one entry per fact somebody is waiting on right now. Safe because
  every waiter inside :meth:`gate`'s closure holds the event object, not a lookup.
* **No stale truth.** A fact is never answered from memory, so a registration since
  evicted cannot release a waiter -- the next :meth:`gate` asks the directory again.
* **Release order is deterministic.** One event per fact, awaited in the order the
  caller listed them, and every wakeup goes through the loop's FIFO ready queue, so
  two requesters released by the same fact resume in the order they parked.

What is *not* here: whether waiting for a fact is a good idea. A gate nothing will
ever record hangs the requester behind it forever, and only the caller knows which
facts are coming -- for dedup, the registration a routed peer owes. That check is
in :mod:`dedup_sim.control.routing`, next to the state that answers it.

Folder-private because dedup is the only capability that waits today; it is the
mechanism behind ``Selection.ready`` and knows nothing about dedup, so it is what
would move into ``proposed`` if a second one needed it.
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

        ``observed`` is asked which of them already are -- *after* interest in every
        one has been registered, which is what makes the answer safe to act on
        however long it takes to arrive.

        ``None`` when they are all already true: "no need to wait at all", not an
        empty wait.
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
