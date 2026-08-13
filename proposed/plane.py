"""The two halves of a capability: :class:`ControlPlane` and :class:`DataPlane`.

A capability's control plane returns a decision; something has to turn that
decision into store calls. That something is its data plane, and the point of
declaring both is the whole proposal in miniature: a capability is written
*against* torchstore rather than *into* it, so writing one should mean
implementing these and nothing else.

* :class:`ControlPlane` -- the deciding half's lifecycle: knobs at construction,
  the stack's ports at :meth:`ControlPlane.attach`, and the one thing a run
  harvests off it once attached -- the model it decides against
  (:attr:`ControlPlane.cluster`), to put a service in front of. Where a plane is
  *reached* from is its type, not a field: a
  :class:`~proposed.selector.KeySelector` is installed in the directory and a
  the plane an application's own hosts ask is given a service, so a capability
  deciding in both places is two planes rather than one naming the other;
* :class:`DataPlane` -- the executing half's lifecycle, and only that: knobs at
  construction, the deployment at :meth:`DataPlane.attach`. What it *does* with
  those ports is its own members' business, because moving bytes is an ordinary
  client call and there is no shape here for a contract to declare.

Both are lifecycle-only, and for one reason: what a capability is asked and what it
does are the capability's, so a run reaches them by holding the object it was given
rather than by a member this package named in advance.

What is deliberately *not* here
-------------------------------
Anything shaped like a run. This package may not import the simulator (see
:mod:`proposed`), and for a long time :class:`DataPlane` broke that rule without
importing anything: it carried ``execute(item)``, ``drain()`` and
``writes_own_outcomes``, every one of them phrased in terms of a *work item* --
which exists only because a harness releases work onto a clock. ``Any`` hid the
import; it did not hide the dependency, and nobody outside a harness could have
implemented them.

They were never the capability's contract, they were the runner's, so they live
with the runner now (``realsim.runner.ItemDispatch``) -- all but ``drain()``,
which turned out not to be anybody's: the one run with work outliving its items
had a request whose decode leg answered too early, and fixing that left nothing to
drain.

``after(requester, result)`` went the same way, later and for a subtler reason. It
was the *only* verb declared here, and only one capability could implement it: a
post-transfer hook exists because something else owns the transfer, so a plane that
owns its own execution (``kvcache_sim``'s serving host) never implemented it while
the run wired a framework callback for the plane that did. A plane that is handed
the fetch owns the whole sequence instead -- read, then keep what was read -- which
is the same shape both capabilities now have, with no hook between the two halves.

Also absent, and for a different reason: any per-operation hook to *shape* a
transfer -- chunking, striping across sources, batching keys, failover. Those are
real questions, but answering them here would put execution back under the control
plane's thumb, which is the coupling this split exists to remove.
"""

from __future__ import annotations

from typing import Any, Optional

from proposed.deployment import ClusterModel

__all__ = ["ControlPlane", "DataPlane"]


class DataPlane:
    """A capability's executing half, as a harness brings it up.

    No verbs. Moving bytes is an ordinary client call -- ``get``, ``put``,
    ``get_batch`` -- so a capability needs no interface to do it and this declares
    none; what it *does* around those calls it declares itself, and whoever wires
    the run names the member (``realsim.runner.ItemDispatch``). ``dedup_sim``'s
    plane reads through and keeps what it read; ``kvcache_sim``'s host walks three
    legs. Neither is a shape this package could have guessed.

    What it does need is somewhere to *reach*, which is what :meth:`attach` hands
    over -- the store to call and the control plane to ask, both on one object.
    """

    def attach(self, deployment: Any) -> None:
        """Receive the deployment this plane executes against. Default: nothing.

        The sibling of :meth:`ControlPlane.attach`, and two-phase for the same
        reason: a plane is constructed with its knobs, and the store, the control
        plane it asks and the model it reports into do not exist until the
        deployment does. One argument, because they all hang off it
        (:class:`~proposed.deployment.Deployment`) -- a plane that had to be handed
        each port separately would make every caller responsible for knowing which
        ports it wanted.
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
    what it is asked -- a :class:`~proposed.selector.AnySelector` ranking the values in
    its subject -- needs no ports of its own and inherits this no-op.
    """

    #: The picture this control plane decides against, if it keeps one. Read after
    #: :meth:`attach` by whoever assembles the run, and given a service of its own
    #: so the application's hosts can report into it without a question being asked.
    #: ``None`` -- the default -- is a control plane that models nothing between
    #: calls, so there is nothing for a host to correct and no service to stand up.
    cluster: Optional[ClusterModel] = None

    def attach(self, view: Any, transfer_cost: Any) -> None:
        """Receive the ports this control plane senses and prices through."""
