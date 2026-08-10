"""Release work on the virtual clock: :class:`Runner`.

Both capabilities used to carry a private driver -- one shaped around a
synchronized burst, one around an arrival stream -- and both drivers did the same
four things:

1. order the work by ``(release_time, id)`` so a run replays identically;
2. install the mesh's shared transport factory **once** for the whole run;
3. release each item at its time and gather them, then drain whatever keeps
   running after the last item's coroutine returns (a batched decode loop, say);
4. record one outcome row per item.

That is this class. Everything capability-specific lives in the
:class:`~proposed.plane.DataPlane` it is given.

Determinism: the release order is a total order (``release_time`` then ``id``),
and the engine's ready queue is FIFO, so items released at the same virtual
instant start in id order, every run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from proposed import DataPlane
from sim_common.report import Ledger, Outcome

__all__ = ["WorkItem", "Workload", "Runner"]


@dataclass
class WorkItem:
    """One unit of work the runner releases onto the clock.

    Args:
        id: stable identity; also the tie-break among items released together.
        release_time: virtual time at which the item starts.
        run: the item's own ordinary store call, awaited by the default
            :meth:`~proposed.plane.DataPlane.execute`. A capability whose plane
            overrides ``execute`` may leave this unset.
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


@dataclass
class Workload:
    """What to run: the items, and any setup that precedes them on the clock.

    ``setup`` is for work that is part of the simulated timeline but is not an
    item -- seeding a key before a read burst. It runs before the first item is
    released, outside the mesh's shared factory, as a single-client drive would.
    """

    items: Sequence[WorkItem]
    setup: Optional[Callable[[], Awaitable[None]]] = None


class Runner:
    """Releases items on the virtual clock over one installed mesh.

    Args:
        mesh: the :class:`realsim.mesh.Mesh` whose shared transport factory is
            installed for the run (duck-typed: only ``installed()`` is used).
        plane: the capability's :class:`~proposed.plane.DataPlane` (default: the
            plain one -- run the item, nothing around it).
        ledger: optional :class:`~sim_common.report.Ledger`. When given, the
            runner records one :class:`~sim_common.report.Outcome` row per item
            and stamps the run's wallclock. A capability whose outcome only
            becomes known at several different lifecycle points (``kvcache_sim``
            publishes a row at rejection, at acceptance, or when the last token
            is emitted) passes ``None`` and records its own richer rows.
        drain: optional coroutine function awaited after every item's coroutine
            has returned, for work that outlives them.
    """

    def __init__(
        self,
        mesh: Any,
        *,
        plane: Optional[DataPlane] = None,
        ledger: Optional[Ledger] = None,
        drain: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self.mesh = mesh
        self.plane = plane if plane is not None else DataPlane()
        self.ledger = ledger
        self.drain = drain

    async def run(self, items: Sequence[WorkItem]) -> Dict[str, Any]:
        """Run every item to completion; return ``item id -> result``."""
        ordered: List[WorkItem] = sorted(items, key=lambda it: it.order)
        if self.ledger is not None:
            self.ledger.items_total += len(ordered)
        with self.mesh.installed():
            results = await asyncio.gather(*(self._run_one(it) for it in ordered))
            if self.drain is not None:
                await self.drain()
            if self.ledger is not None:
                self.ledger.wallclock = asyncio.get_running_loop().time()
        return {it.id: res for it, res in zip(ordered, results)}

    async def _run_one(self, item: WorkItem) -> Any:
        """Sleep until the item's release time, execute it, then record it."""
        loop = asyncio.get_running_loop()
        delay = item.release_time - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        result = await self.plane.execute(item)
        await self.plane.after(item, result)
        if self.ledger is not None:
            self.ledger.add(
                Outcome(
                    id=item.id, released=item.release_time, done=loop.time()
                )
            )
        return result
