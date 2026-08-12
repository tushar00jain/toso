"""Release work on the virtual clock: :class:`Runner`.

Both capabilities used to carry a private driver -- one shaped around a
synchronized burst, one around an arrival stream -- and both drivers did the same
four things:

1. order the work by ``(release_time, id)`` so a run replays identically;
2. install the mesh's shared transport factory **once** for the whole run;
3. release each item at its time and gather them, then drain whatever keeps
   running after the last item's coroutine returns (a batched decode loop, say);
4. record one outcome row per item.

That is this class. Everything capability-specific reaches it through
:class:`ItemDispatch`.

Why the shape a run needs lives here
------------------------------------
It used to live in :class:`~proposed.plane.DataPlane`: ``execute(item)``,
``drain()`` and ``writes_own_outcomes``, alongside the one member a capability
actually declares. Every one of those three is phrased in terms of a **work item**,
and a work item exists because *this loop* releases work onto a clock -- a
deployment has none. So a package forbidden from importing the simulator was
declaring an interface only the simulator could implement.

They are this loop's contract, so they are here. :class:`ItemDispatch` is that
contract, and it takes functions: a capability satisfies it without inheriting
anything, and run wiring never has to declare itself a capability's executing half
in order to be driven.

That also gives a capability whose executing half is *many* objects somewhere to
stand. ``kvcache_sim`` is one serving host per instance, so what it hands this loop
is a dispatcher -- it executes nothing itself, it decides which of the real ones
gets the item.

Determinism: the release order is a total order (``release_time`` then ``id``),
and the engine's ready queue is FIFO, so items released at the same virtual
instant start in id order, every run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from sim_common.report import Ledger, Outcome

__all__ = ["ItemDispatch", "Runner", "WorkItem"]


@dataclass
class WorkItem:
    """One unit of work the runner releases onto the clock.

    Args:
        id: stable identity; also the tie-break among items released together.
        release_time: virtual time at which the item starts.
        run: the item's own ordinary store call, awaited by the default
            :meth:`ItemDispatch.execute`. A run whose dispatch supplies its own
            ``on_item`` may leave this unset.
        payload: whatever the capability's plane needs (a request, a key, ...).
    """

    id: str
    release_time: float = 0.0
    run: Optional[Callable[[], Awaitable[Any]]] = None
    payload: Any = None

    @property
    def order(self):
        """The total order the runner releases items in."""
        return (self.release_time, self.id)


class ItemDispatch:
    """What :class:`Runner` drives: how to run an item, and what follows it.

    Three functions, all optional, each with the plain answer as its default:

    * ``on_item`` -- run one item. Default: the item's own ordinary store call,
      which is what a capability that adds nothing around the transfer wants;
    * ``on_after`` -- a capability's :meth:`proposed.plane.DataPlane.after`, called
      with the *requester* rather than the item, because that is the part of an
      item that means anything outside a harness;
    * ``on_drain`` -- awaited once after every item's coroutine has returned, for
      work that outlives them (every host's decode, say).

    Args:
        on_item: called with each :class:`WorkItem`; its result is the item's.
        on_after: called ``(requester, result)`` once an item has landed.
        on_drain: awaited after the last item. ``None`` -> nothing to wait for,
            which the runner can see without inspecting whether a method was
            overridden.
        writes_own_outcomes: set it when what is behind these functions publishes
            its own outcome rows -- one at several different lifecycle points,
            say -- so the runner does not also write one per item.
    """

    def __init__(
        self,
        on_item: Optional[Callable[[WorkItem], Awaitable[Any]]] = None,
        *,
        on_after: Optional[Callable[[str, Any], Awaitable[None]]] = None,
        on_drain: Optional[Callable[[], Awaitable[None]]] = None,
        writes_own_outcomes: bool = False,
    ) -> None:
        self._on_item = on_item
        self._on_after = on_after
        self.on_drain = on_drain
        self.writes_own_outcomes = writes_own_outcomes

    async def execute(self, item: WorkItem) -> Any:
        """Run ``item``; by default, its own ordinary store call."""
        if self._on_item is None:
            return await item.run()
        return await self._on_item(item)

    async def after(self, item: WorkItem, result: Any) -> None:
        """Hand the capability what landed, named by who fetched it."""
        if self._on_after is not None:
            await self._on_after(item.id, result)


class Runner:
    """Releases items on the virtual clock over one installed mesh.

    Args:
        mesh: the :class:`realsim.mesh.Mesh` whose shared transport factory is
            installed for the run (duck-typed: only ``installed()`` is used).
        dispatch: the run's :class:`ItemDispatch` (default: the plain one -- run
            the item, nothing around it).
        ledger: optional :class:`~sim_common.report.Ledger`. When given, the
            runner records one :class:`~sim_common.report.Outcome` row per item
            and stamps the run's wallclock. A capability whose outcome only
            becomes known at several different lifecycle points (``kvcache_sim``
            publishes a row at rejection, at acceptance, or when the last token
            is emitted) passes ``None`` and records its own richer rows.
    """

    def __init__(
        self,
        mesh: Any,
        *,
        dispatch: Optional[ItemDispatch] = None,
        ledger: Optional[Ledger] = None,
    ) -> None:
        self.mesh = mesh
        self.dispatch = dispatch if dispatch is not None else ItemDispatch()
        self.ledger = ledger

    async def run(self, items: Sequence[WorkItem]) -> Dict[str, Any]:
        """Run every item to completion; return ``item id -> result``."""
        ordered: List[WorkItem] = sorted(items, key=lambda it: it.order)
        if self.ledger is not None:
            self.ledger.items_total += len(ordered)
        with self.mesh.installed():
            results = await asyncio.gather(*(self._run_one(it) for it in ordered))
            if self.dispatch.on_drain is not None:
                await self.dispatch.on_drain()
            if self.ledger is not None:
                self.ledger.wallclock = asyncio.get_running_loop().time()
        return {it.id: res for it, res in zip(ordered, results)}

    async def _run_one(self, item: WorkItem) -> Any:
        """Sleep until the item's release time, execute it, then record it."""
        loop = asyncio.get_running_loop()
        delay = item.release_time - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        result = await self.dispatch.execute(item)
        await self.dispatch.after(item, result)
        if self.ledger is not None:
            self.ledger.add(
                Outcome(
                    id=item.id, released=item.release_time, done=loop.time()
                )
            )
        return result
