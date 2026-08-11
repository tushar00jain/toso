"""The execution half of a capability: :class:`DataPlane`.

A capability's control plane returns a decision; something has to turn that
decision into store calls. That something is small and has the same shape in
every capability, so it is one two-method type:

* :class:`ControlPlane` -- the deciding half's one lifecycle member: knobs at
  construction, the stack's ports at :meth:`ControlPlane.attach`;
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

__all__ = ["ControlPlane", "DataPlane"]


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


class ControlPlane:
    """What a capability's *deciding* half looks like to a harness.

    The sibling of :class:`DataPlane`, and as capability-agnostic: it says nothing
    about what is decided, only how such an object is brought up. Construct it with
    its knobs, and it is handed the stack's ports once those exist --
    :meth:`attach` -- because a control plane senses through a
    :class:`~proposed.view.View` and prices through a
    :class:`~proposed.cost.TransferCost`, neither of which a caller has before the
    store is assembled.

    That two-phase shape is torchstore's own: ``TorchStoreStrategy`` takes its
    knobs at construction and receives the cluster later through
    ``set_storage_volumes``.

    The default is real behaviour, not a stub: a control plane that decides from
    what it is asked -- a :class:`~proposed.policy.Policy` handed a view per call --
    needs no ports of its own and inherits this no-op.
    """

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Receive the ports this control plane senses and prices through."""
