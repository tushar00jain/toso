"""One action, one commit: :class:`Dispatcher`, and the reducers it calls.

One fact can move several owners' state at once: an admitted request moves the
scheduler's reservation, the routed record and the pending map
(``kvcache_sim.control._sensor``), and written one at a time their order is a caller's
problem. Combined from one :class:`Action` and committed together, it is nobody's. A
capability with one owner per fact gets the commit and the wake instead --
``dedup_sim`` folds :class:`Stored` in one place and parks its readers on the commit.

**Nothing is stored here.** A dispatcher holds registrations -- an action type, and the
reducers that fold it -- and no state at all. Every reducer goes on owning exactly what
it owned before, and is handed no way to read a neighbour's. What is shared is the
transaction: one entry point, one boundary after every reducer for that action has run,
one payload-free notification at it.

Which is the rule that makes the order they run in irrelevant, stated here and nowhere
else: **a reducer writes its own state and reads nothing else.** A fold reaching for
another's would be deciding, and deciding belongs in the plane that senses them all at
once (:class:`proposed.view.View`).

The vocabulary is Redux's, minus one word: a *selector* here ranks sources
(:mod:`proposed.selector`), so a read of committed state is called a read.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

from proposed.deployment import Key, VolumeId
from proposed.selector import Ready

__all__ = ["Action", "Dispatcher", "Fold", "Probe", "Reducer", "Stored"]

#: Folds one action into one reducer's own state, and reads nothing else.
Fold = Callable[[Any], None]

#: Reads whether what a waiter is waiting for is true *now*, from wherever that truth
#: lives (for a registration, the directory). Not a coroutine: it is called between two
#: commits, and a read that could suspend would answer for another moment.
Probe = Callable[[], bool]


class Action:
    """Something that happened, as a value: the one thing a caller dispatches.

    A base rather than a protocol, for :class:`proposed.deployment.Sensor`'s reason: a
    member-less protocol would declare nothing, every object satisfying it.
    """


@dataclass(frozen=True)
class Stored(Action):
    """``host`` holds ``key`` from now on -- one reader's landed put.

    The put registered itself before it returned, so the directory needs nothing from
    this: what it carries is who stored what, for whoever keeps state about the debt
    that put discharged.
    """

    host: VolumeId
    key: Key


class Reducer(Protocol):
    """State something owns, and which actions it folds into it."""

    @property
    def folds(self) -> Mapping[type, Fold]:
        """``action type -> the write that folds it``, so it is called for those and no
        others. Fixed for the reducer's life."""


class Dispatcher:
    """Who folds which action, and the commit that follows them.

    Empty until reducers are composed onto it (:meth:`compose`), because they come up at
    different times -- a plane's own sensor once it is attached, the directory once the
    deployment exists.

    What a run fronts with a service (:attr:`~proposed.plane.ControlPlane.dispatcher`),
    so this is the whole of where a host's facts arrive and the only place an action is
    folded.

    **Two ways in, one fold**, the same pair as
    :meth:`proposed.deployment.Controller.locate_volumes` /
    :meth:`~proposed.deployment.Controller.locate_raw` and for the same reason:
    :meth:`dispatch` is the seam a reporter at any distance reaches, and
    :meth:`dispatch_sync` is what a caller in this process calls when it must not
    suspend. A caller chooses between them on the distance, never on the fold.
    """

    def __init__(self) -> None:
        # action type -> the folds to run for it, in composition order. The order is
        # fixed so a run is reproducible; nothing rests on which it is (see above).
        self._folds: Dict[type, List[Fold]] = {}
        # The event the next commit sets, replaced as it is set: a waiter holds the
        # object, so it is woken exactly once, by the commit it captured.
        self._commit = asyncio.Event()

    def compose(self, reducer: Reducer) -> None:
        """Register ``reducer``'s folds."""
        for action, fold in reducer.folds.items():
            self._folds.setdefault(action, []).append(fold)

    async def dispatch(self, action: Action) -> None:
        """Fold ``action`` in, as a reporter at any distance reaches it: the seam.

        Awaited, like :meth:`proposed.deployment.Controller.notify_put_batch`: the
        reply carries nothing and is the ordering a reporter needs, since it comes back
        after the commit, so the question it asks next is decided against state that has
        folded this action. Sending it one-way would order it only at the sender, and
        over any distance at all the question would arrive first.
        """
        self.dispatch_sync(action)

    def dispatch_sync(self, action: Action) -> None:
        """Fold ``action`` into every reducer that folds its type, then commit.

        **Not a coroutine, and that is load-bearing**, as
        :meth:`proposed.deployment.Controller.locate_raw` is not one: no decision can
        interleave with a commit, so whoever the commit wakes reads state that has
        folded all of ``action`` rather than part of it. A control plane moving several
        sensors as one decision rests its atomicity on that
        (``kvcache_sim.control.scheduler``'s admission), and an ``await`` anywhere in
        this path -- a fold, or the wake at the commit -- would let a second decision
        interleave.

        Raises:
            TypeError: nothing composed onto this dispatcher folds ``action``'s type.
                The wiring gap and the hang are the same bug -- a fact nothing folds is
                a waiter nothing wakes.
        """
        folds = self._folds.get(type(action))
        if folds is None:
            raise TypeError(
                f"nothing here folds {type(action).__name__}: this run's actions are "
                f"{', '.join(sorted(a.__name__ for a in self._folds)) or 'none'}"
            )
        for fold in folds:
            fold(action)
        # The commit boundary: every reducer for this action has written, so everything
        # parked is woken -- with nothing, because what it needs is what it can read.
        woken, self._commit = self._commit, asyncio.Event()
        woken.set()

    def gate(self, holds: Probe) -> Optional[Ready]:
        """An awaitable that returns once ``holds()`` is true, or ``None`` if it is.

        ``None`` means "no need to wait at all", not an empty wait. Whether what
        ``holds`` reads is *coming* is the caller's question: a gate on something
        nothing will ever commit parks its waiter for the rest of the run.

        Three things make it safe. No lost wakeup: each pass captures the next commit
        *before* reading, so one landing in between sets an event the waiter already
        holds. Nothing remembered: ``holds`` is asked again at every commit, so a fact
        that was true and has since stopped being one -- a volume that evicted what it
        registered -- parks its next waiter rather than releasing it. And nothing kept
        per waiter or per fact, so what wakes anybody is a commit and never an action.
        """
        # TODO: wake only the waiters an action concerns. Every commit wakes every
        # parked waiter, which re-probes and re-parks: O(parked) per commit, so O(m^2)
        # over a burst of m readers. The remedy is cheap, linear and not speculative --
        # the interest -> event map a per-fact readiness gate keeps, so a commit sets
        # only what it touched. What is not cheap is the vocabulary: only an application
        # knows which interests its own action satisfies, so the action would declare
        # what it touches and this would take the keys to park under. The default stays
        # "wake everyone" whichever way that goes -- an under-reported touch is a waiter
        # that never wakes, where this is only slow. Nothing here is wrong today: the
        # probe decides, and a wake, a probe and a re-park cost no simulated time.
        if holds():
            return None

        async def ready() -> None:
            while True:
                commit = self._commit
                if holds():
                    return
                await commit.wait()

        return ready
