"""The application's own control plane: :class:`Coordinator`.

Some control planes are consulted inside the store -- a
:class:`~proposed.policy.Policy` runs in the controller's ``locate_volumes``, and
every decision it makes is about *data*: which copy serves this read, what leaves to
make room for a new one. This is the other kind: an application's control plane,
which decides about the application's own resources (which instance runs this work)
from facts only the application can see (its queues, its batches, its own
completions).

The store declares it anyway, for one reason: an application that needs such a thing
should not have to invent its shape. This is the shape.

Two members, and only two
-------------------------
* :meth:`decide` -- something is asked; the coordinator answers, or refuses.
* :meth:`observe` -- something is reported; the coordinator learns and answers
  nothing.

That is the whole vocabulary, because those are the only two ways a control plane
that runs as a service can be interacted with: a call that waits for a reply, and a
send that does not. There is no third member for "admit this", "reject that", "here
is what finished" -- those are :meth:`decide` and :meth:`observe` with different
payloads, and the payloads are the *application's* to define.

What deliberately is not here: anything the coordinator would do **unprompted**.
Nothing in this surface runs without a caller, so a control plane that has to notice
that time passed -- a rebalance, a reclaim sweep, an expiry -- has nowhere to put it
yet. That absence is real and better left visible than papered over with a ``tick``
nobody drives.

Why the payloads are the application's
--------------------------------------
A demand, an answer and a fact are all ``Any``, for the reason given in
:mod:`proposed.deployment`: this package cannot import an application. What is
declared is the *structure* -- which interactions exist and which way each one
travels -- so that a KV-cache scheduler, a weight-sync planner and whatever comes
third all reach their control plane the same way, and the boundary between them and
it is built once (:mod:`realsim.seams.coordinator_handle`) instead of per
application.

An implementation should keep the generality in the surface and *not* in an
``if isinstance(...)`` ladder: ``functools.singledispatchmethod`` on the demand type
gives each kind its own typed method behind these two names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from proposed.plane import ControlPlane

__all__ = ["Coordinator"]


class Coordinator(ControlPlane, ABC):
    """An application's deciding half: values in, values out.

    This is the port between the two planes, and it is deliberately the *whole*
    port -- a host that held anything else would be reaching into another service.
    ``check_structure.py`` rule 6 enforces that: a ``data/`` module annotating a
    field with this class may name only these members on it (no other attribute, no
    subscript, no ``getattr``).

    Abstract, and a base class rather than a ``Protocol``, so it is the same kind of
    thing as :class:`~proposed.policy.Policy`: a control plane declares which
    surfaces it implements in its bases, and both answers are read the same way. One
    object may declare both -- ``kvcache_sim``'s scheduler does, because the peer it
    prices a pull against is the peer it later tells the directory to serve.

    Every argument and every return is a value; nothing here takes a handle to a
    data-plane object, and nothing here is a field the data plane reads. That is what
    lets the boundary be inserted: the object and the reference to it have the same
    shape, so a host cannot tell whether the coordinator it is talking to is in this
    process or another one.

    :class:`proposed.deployment.Coordinator` declares these same two members for the
    *caller* -- an endpoint each, because what a caller holds is a reference to a
    running service rather than the object. Two members is what keeps those two
    halves from drifting: there is nothing application-specific in either to keep in
    sync.
    """

    @abstractmethod
    async def decide(self, demand: Any) -> Optional[Any]:
        """Answer ``demand``, or ``None`` to refuse it.

        One member for every question the application asks its control plane: where
        should this work run, may this proceed to the next stage, what became true
        now that this finished. Which questions exist is the application's business;
        that they are asked and answered here is not.

        ``None`` is the refusal, and it is the whole of the vocabulary today -- a
        surface that could also say *not yet* would let a caller be held rather than
        turned away, which is what flow control means and what this cannot yet
        express.

        No clock is passed in. A coordinator reads its own, because a demand arrives
        when it arrives: over a non-zero hop the sender's stamp is already stale, and
        deciding against every instance's queue with it would be reading the cluster
        in the past.
        """

    def observe(self, fact: Any) -> None:
        """Learn that ``fact`` happened. Answers nothing.

        Default: nothing -- real behaviour, not a stub. A coordinator that decides
        from what it is asked, and models nothing between calls, needs no facts and
        inherits this.

        Not awaitable, and that is about the *handler*, not the caller: whoever
        reports pays a hop either way (``observe.broadcast(...)`` on the endpoint it
        holds), while a body that cannot suspend cannot lose an event halfway through
        handling it. A control plane whose model is corrected by these reports
        depends on that.
        """
