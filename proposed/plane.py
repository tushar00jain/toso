"""The execution half of a capability: :class:`DataPlane`.

A capability's control plane returns a decision; something has to turn that
decision into store calls. That something is small and has the same shape in
every capability, so it is one two-method type:

* :meth:`DataPlane.execute` -- the work *around* the transfer. The transfer
  itself is an ordinary client call, so the default adds nothing and simply runs
  the work item's own call;
* :meth:`DataPlane.after` -- registration and eviction once the bytes have
  landed. The default does nothing.

Both defaults are real behaviour, not stubs: a plain fetch takes them unchanged.
``dedup_sim`` overrides only :meth:`after` (the read-through put that makes the
reader a directory source for the next one); ``kvcache_sim`` overrides both (a
serving loop around the pull, then publish + evict). A new capability starts by
overriding one method.

Deliberately absent: any per-operation hook to *shape* a transfer -- chunking,
striping across sources, batching keys, failover. Those are real questions, but
answering them here would put execution back under the control plane's thumb,
which is the coupling this split exists to remove.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DataPlane"]


class DataPlane:
    """What a capability does around, and after, a transfer.

    Two class-level facts let a run be driven without the caller restating them:
    whether the capability publishes its own outcome rows, and whether it has work
    that outlives the items. Both default to the simple answer.
    """

    #: Set by a capability that records its own outcome rows -- one published at
    #: several different lifecycle points, say -- so the harness does not also
    #: write one per item.
    writes_own_outcomes: bool = False

    async def execute(self, item: Any) -> Any:
        """Run the work ``item`` describes; return its result.

        Default: nothing around the transfer -- just the item's own ordinary
        store call.
        """
        return await item.run()

    async def after(self, item: Any, result: Any) -> None:
        """Registration / eviction once ``item``'s bytes have landed.

        Default: nothing.
        """

    async def drain(self) -> None:
        """Await work that outlives the items (kvcache's decode steps).

        Called once, after every item's coroutine has returned. Default: nothing.
        """
