"""Waiting for a fact that has not happened yet: :class:`Readiness`.

A policy that routes a requester to a source which does not hold the key *yet* has
to withhold its answer until it does. That is two pieces of state -- the facts
observed so far, and a gate per fact still outstanding -- and getting them right is
concurrency work, not routing work: the whole reason the fact set exists is that a
gate must never be created for something that already happened.

So it lives here, in one object, and :mod:`dedup_sim.control.routing` is left with
the tree it is actually shaping. A *fact* is any hashable the caller invents; this
module never interprets one. Dedup's is ``(volume_id, key)``: "that volume holds
that key".

It is folder-private because dedup is the only capability that waits today. If a
second one needs it, this is what would move into ``proposed`` -- it is the
mechanism behind ``Selection.ready``, and nothing in it knows about dedup.

Why it is safe
--------------
* **The gate latches.** ``record`` before ``gate`` means no gate at all; ``record``
  after means the waiter is already parked on an ``Event`` that gets set. There is no
  window between checking and waiting, because :meth:`gate` does both without
  awaiting.
* **Released gates are dropped.** Once a fact is recorded it is in :attr:`_facts`
  forever, so a later :meth:`gate` call sees it as satisfied and never consults the
  event again -- keeping the released event would grow the map for the life of the
  run. Dropping it is safe: a waiter already inside :meth:`gate`'s closure holds the
  object itself, not a lookup.
* **Release order is deterministic.** One event per fact, awaited in the order the
  caller listed them, and every wakeup goes through the loop's FIFO ready queue --
  so two requesters released by the same fact resume in the order they parked.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, Hashable, Iterable, Optional, Set

__all__ = ["Readiness"]


class Readiness:
    """Facts observed, and the gates waiting on facts that are not true yet."""

    def __init__(self) -> None:
        # Facts that have happened. Never pruned: it is what makes a gate for an
        # already-true fact impossible, which is the lost-wakeup guard.
        self._facts: Set[Hashable] = set()
        # One event per outstanding fact, dropped as soon as it is released.
        self._gates: Dict[Hashable, asyncio.Event] = {}

    def record(self, fact: Hashable) -> None:
        """``fact`` is now true: release anything waiting on it."""
        self._facts.add(fact)
        gate = self._gates.pop(fact, None)
        if gate is not None:
            gate.set()

    def gate(
        self, facts: Iterable[Hashable]
    ) -> Optional[Callable[[], Awaitable[None]]]:
        """An awaitable that returns once every one of ``facts`` is true.

        ``None`` when they all already are -- which the caller should read as "no
        need to wait at all", not as an empty wait.
        """
        pending = [f for f in facts if f not in self._facts]
        if not pending:
            return None
        events = [self._gates.setdefault(f, asyncio.Event()) for f in pending]

        async def ready() -> None:
            for event in events:
                await event.wait()

        return ready
