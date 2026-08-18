"""One action, one commit: :class:`Dispatcher`, and the reducers it calls.

One fact can move several owners' state at once: an admitted request moves the
scheduler's reservation, the routed record and the pending map
(``kvcache_sim.control._sensor``), and written one at a time their order is a caller's
problem. Folded from one :class:`Action` and committed together, it is nobody's. A
capability with one owner per fact gets the commit and the wake instead --
``dedup_sim`` folds its completion in one place and parks its readers on the commit.

**No application state is stored here.** Every reducer goes on owning exactly what it
owned before, and is handed no way to read a neighbour's. The dispatcher holds its
wiring and currently parked waiters: one entry point, one boundary after every reducer
for that action has run, and payload-free gate updates for that action.

Which is the rule that makes the order they run in irrelevant, stated here and nowhere
else: **a reducer writes its own state and reads nothing else.** A fold reaching for
another's would be deciding, and deciding belongs in the plane that holds the sensors.

The vocabulary is Redux's, minus one word: a *selector* here ranks sources
(:mod:`proposed.selector`), so a read of committed state is called a read.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Set

from proposed.selector import Ready

__all__ = ["Action", "Dispatcher", "Fold", "Probe", "Reducer"]

#: Folds one action into one reducer's own state, and reads nothing else.
Fold = Callable[[Any], None]


class Action:
    """Something that happened, as a value: the one thing a caller dispatches.

    A base rather than a protocol, for :class:`proposed.deployment.Sensor`'s reason: a
    member-less protocol would declare nothing, every object satisfying it.
    """


#: Reads whether what a waiter is waiting for is true now.
Probe = Callable[[], bool]


class Reducer(Protocol):
    """State something owns, and which actions it folds into it."""

    @property
    def folds(self) -> Mapping[type, Fold]:
        """``action type -> the write that folds it``, so it is called for those and no
        others. Fixed for the reducer's life."""


class _Waiter:
    """One gate, released after every missing action commits."""

    def __init__(self, actions: Iterable[Action]) -> None:
        self.missing: Set[Action] = set(actions)
        self.ready = asyncio.Event()


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
        # action -> the gates still missing it, in parking order.
        self._waiters: Dict[Action, Dict[_Waiter, None]] = {}

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
        # Every fold is visible before this commit satisfies a gate.
        for waiter in self._waiters.pop(action, {}):
            waiter.missing.discard(action)
            if not waiter.missing:
                waiter.ready.set()

    def gate(
        self, holds: Probe, actions: Iterable[Action]
    ) -> Optional[Ready]:
        """An awaitable for every ``action``, or ``None`` if ``holds()`` now.

        A gate on an action that will never commit parks its waiter for the rest of the
        run. Callers naming several actions must know they are all coming when the
        probe is false.

        The gate parks before probing, so a commit during that read cannot be lost.
        Each commit removes one missing action; only the last wakes the task.
        """
        waiter = _Waiter(actions)
        if not waiter.missing:
            if holds():
                return None
            raise ValueError("a false gate must name at least one action")
        for action in waiter.missing:
            self._waiters.setdefault(action, {})[waiter] = None
        if holds():
            for action in waiter.missing:
                waiters = self._waiters.get(action)
                if waiters is None:
                    continue
                waiters.pop(waiter, None)
                if not waiters:
                    del self._waiters[action]
            return None

        async def ready() -> None:
            try:
                await waiter.ready.wait()
            finally:
                for action in waiter.missing:
                    waiters = self._waiters.get(action)
                    if waiters is None:
                        continue
                    waiters.pop(waiter, None)
                    if not waiters:
                        del self._waiters[action]

        return ready
