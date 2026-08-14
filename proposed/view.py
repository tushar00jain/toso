"""The read-only observation a control plane is handed: :class:`View`.

A :class:`~proposed.selector.KeySelector` never touches a client, a volume or the
mesh -- it is given a ``View`` and returns a decision. The ``View`` is the read-only
half of that contract: *reads* of state that already exists, and no mutation of any
kind.

What the base view offers, and how a capability adds to it
----------------------------------------------------------
* :meth:`locate` -- the real directory answer for a set of keys, read straight
  from the real ``Controller`` body, with no caller's source preference applied to
  it (see :mod:`proposed.selector`): a view must report the directory as it *is*,
  and a selector ranking an answer somebody has already reordered would be reading
  its own output back.
* :meth:`holders` / :meth:`topology` / :meth:`endpoint` / :meth:`locality` --
  who holds a key and how far away they are, the two inputs every source
  decision has needed so far.
* :meth:`now` -- the running loop's clock, because a decision has to be timed
  where it is *made*: a selector is handed a subject and no timestamp, since
  over a non-zero hop the sender's stamp is already stale and comparing it against
  every instance's queue would read the cluster in the past.

Anything more specific stays out, and is *composed* on instead: a capability adds
one subclass of this base per read it senses, and builds a view by naming them
(:meth:`View.derived`), so a selector takes the one view carrying the read it needs,
any one of them alone is already a view, and the class statement is the list of what
that capability's decisions sense. ``kvcache_sim``
composes leading-prefix-run lengths, which are a KV-cache notion (a block key
chain); ``dedup_sim`` wants a fan-out tally, which is the selector's own
bookkeeping, not observed state. Folding either into the base type would make it a
union serving neither caller -- and per-node *load* is the same trap twice over: the
KV-cache scheduler's load signal is its own predicted prefill queue (a control-plane
model, not an observation) and the dedup selector's is its planned tree, so a base
``load()`` would be a stub with two incompatible meanings. It is left out until a
caller can observe one: an application ranking by load exposes its own
:class:`~proposed.deployment.Sensor` through a view of its own.

Construction
------------
A view is built from a :class:`~proposed.deployment.Controller` and a topology,
and reads the first through ``locate_raw`` alone. That surface is *declared* rather
than left as ``Any``, so the proposal states what it needs instead of silently
depending on the shape of whatever the simulator happens to pass. ``realsim``
builds one via ``Mesh.view``; a real controller would build one over itself.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Sequence

from proposed.cost import TransferCost
from proposed.deployment import Controller
from proposed.topology import Endpoint

__all__ = ["View"]


class View:
    """Everything a control plane senses and prices with, and nothing else.

    A container of run-supplied reads: who holds a key, where the volumes are, what
    time it is, what a transfer would cost. This base holds what a plane could not
    otherwise reach and nothing a capability made itself; a capability whose decisions
    also read state it holds itself composes a subclass carrying that
    :class:`~proposed.deployment.Sensor`, over these same ports (:meth:`derived`).

    What it reads are the members: :attr:`directory`, :attr:`topology`. Of the
    five the directory answers, one is safe for a decision to read, and that one is
    what :meth:`locate` calls -- the read spelled out here rather than left to each
    caller, since a decision made against an answer somebody has already reordered
    would be ranking its own output back.

    Args:
        directory: the directory to read -- anything satisfying
            :class:`~proposed.deployment.Controller`. In the simulator that is the
            controller service, in a deployment the controller itself.
        topology: ``volume_id -> Endpoint``; the volume id is the directory
            identity, the endpoint is what locality is priced against.
        cost: a :data:`~proposed.cost.TransferCost`. ``None`` for a run whose
            decisions price nothing, which is every capability but ``kvcache_sim``.
    """

    def __init__(
        self,
        directory: Controller,
        topology: Dict[str, Endpoint],
        cost: Optional[TransferCost] = None,
    ) -> None:
        self._directory = directory
        self._topology = dict(topology)
        self._cost = cost

    def derived(self, cls: type, **sensors: Any) -> "View":
        """A view of type ``cls`` over these same ports, carrying ``sensors``.

        How a capability composes its own reads in: ``cls`` names the views the
        capability adds -- ``kvcache_sim`` adds prefix runs, its model of the cluster,
        and the pulls it has priced -- rather than a read handle it assembles out of
        ports it would have to be handed one by one. Each derives this base, so
        composing a subset of them is a class statement and this base enters the
        result's MRO once.

        ``sensors`` is what the capability *already holds* and wants read through the
        same view as the run's ports, and nothing here supplies any of it (see the load
        discussion above). Passed as keywords, each claimed by the view that named it,
        so this base names none of them -- and one no view claimed reaches this base's
        own ``__init__``, which takes none, and fails there rather than being silently
        absent.
        """
        return cls(self._directory, self._topology, self._cost, **sensors)

    @property
    def directory(self) -> Controller:
        """The directory this view senses. :meth:`locate` is the read to make of it."""
        return self._directory

    def locate(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``key -> {volume_id -> StorageInfo}`` from the REAL directory.

        Missing keys are simply absent (``missing_ok=True``): a read reports
        what is there, it does not raise at the observer. Reads the raw controller
        body, so no caller's preference is folded into what a decision is made
        against -- and it cannot suspend, which is what that decision relies on
        (:meth:`~proposed.deployment.Controller.locate_raw`).
        """
        if not keys:
            return {}
        return self._directory.locate_raw(list(keys), missing_ok=True)

    # -- topology ----------------------------------------------------------- #
    @property
    def topology(self) -> Dict[str, Endpoint]:
        """``volume_id -> Endpoint`` for the whole run."""
        return self._topology

    # -- price -------------------------------------------------------------- #
    def transfer_cost(self, src_id: str, dst_id: str, nbytes: int) -> float:
        """Seconds to move ``nbytes`` from ``src_id`` to ``dst_id``.

        The run's estimate, not this capability's, so a prediction and the charge the
        transport makes cannot diverge. Raises for a run that supplied none: pricing
        a transfer a run cannot price is a scheduler in the wrong harness, not a
        number to invent.
        """
        if self._cost is None:
            raise RuntimeError(
                "this run supplied no transfer cost, so nothing here can be priced"
            )
        return self._cost(src_id, dst_id, nbytes)

    # -- clock -------------------------------------------------------------- #
    def now(self) -> float:
        """The running loop's clock, in seconds.

        Stock ``asyncio``, and the whole of what makes this liftable: under a plain
        loop it is ``time.monotonic()`` (real seconds), under a simulation engine
        whose loop overrides ``time()`` it is that run's virtual seconds. Same line
        either way -- which is why time is read here and never ``time.time()``,
        whose value no loop controls and no run can reproduce.
        """
        return asyncio.get_running_loop().time()
