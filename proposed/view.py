"""The read-only observation a policy is handed: :class:`View`.

A :class:`~proposed.policy.Policy` never touches a client, a volume or the
mesh -- it is given a ``View`` and returns a decision. The ``View`` is the sensor
half of that contract: awaited *reads* of state that already exists, and no
mutation of any kind.

What the base view offers, and why that is all it offers
--------------------------------------------------------
* :meth:`locate` -- the real directory answer for a set of keys, read straight
  from the real ``Controller`` body. It deliberately bypasses the controller's
  routing hook (see :mod:`proposed.policy`): a sensor must report the directory
  as it *is*, and a policy reading its own answer back would recurse.
* :meth:`holders` / :meth:`topology` / :meth:`endpoint` / :meth:`locality` --
  who holds a key and how far away they are, the two inputs every source
  decision has needed so far.
* :meth:`now` -- the running loop's clock, because a decision has to be timed
  where it is *made*: a coordinator is handed a request and no timestamp, since
  over a non-zero hop the sender's stamp is already stale and comparing it against
  every instance's queue would read the cluster in the past.

Anything more specific stays out. ``kvcache_sim`` wants leading-prefix-run
lengths, which are a KV-cache notion (a block key chain), so that derived read is
a subclass in ``kvcache_sim/control/``; ``dedup_sim`` wants a fan-out tally,
which is the policy's own bookkeeping, not observed state. Folding either into
the base type would make it a union serving neither caller -- and per-node
*load* is the same trap twice over: the KV-cache scheduler's load signal is its
own predicted prefill queue (a control-plane model, not an observation) and the
dedup policy's is its planned tree, so a base ``load()`` would be a stub with two
incompatible meanings. It is left out until a caller can observe one.

Construction
------------
A view is built from a :class:`~proposed.deployment.Controller` and a topology. The
controller is *declared* rather than left as ``Any``: the proposal states what it
needs from a controller instead of silently depending on the shape of whatever the
simulator happens to pass. Only one member of that surface is used here --
``locate_raw``, the unrouted read -- and it lives on the controller rather than in a
narrower protocol beside it, because the object that has it is the same object that
answers ``locate_volumes``. ``realsim`` builds one via ``Mesh.view``; a real
controller would build one over itself.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from proposed.deployment import Controller
from proposed.topology import Endpoint, locality, Tier

__all__ = ["View"]


class View:
    """Awaited, read-only observation of the real directory and topology.

    Args:
        directory: the directory service to read -- anything satisfying
            :class:`~proposed.deployment.Controller`. In the simulator that is the
            controller service, in a deployment the controller itself; either way
            only :meth:`~proposed.deployment.Controller.locate_raw` is used.
        topology: ``volume_id -> Endpoint``; the volume id is the directory
            identity, the endpoint is what locality is priced against.
    """

    def __init__(
        self, directory: Controller, topology: Dict[str, Endpoint]
    ) -> None:
        self._directory = directory
        self._topology = dict(topology)

    # -- directory ---------------------------------------------------------- #
    @property
    def directory(self) -> Controller:
        """The directory this view reads, so a derived view can rebuild over it."""
        return self._directory

    async def locate(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``key -> {volume_id -> StorageInfo}`` from the REAL directory.

        Missing keys are simply absent (``missing_ok=True``): a sensor reports
        what is there, it does not raise at the observer. Reads the raw
        controller body, so the routing hook a policy may be installed behind is
        not re-entered.
        """
        if not keys:
            return {}
        return await self._directory.locate_raw(list(keys), missing_ok=True)

    @staticmethod
    def holders(located: Dict[str, Dict[str, Any]], key: str) -> List[str]:
        """Volumes holding ``key``, in directory order (empty if none)."""
        return list(located.get(key, {}))

    # -- topology ----------------------------------------------------------- #
    @property
    def topology(self) -> Dict[str, Endpoint]:
        """``volume_id -> Endpoint`` for the whole run."""
        return self._topology

    def endpoint(self, volume_id: str) -> Endpoint:
        """``volume_id``'s locality endpoint."""
        return self._topology[volume_id]

    def locality(self, src_id: str, dst_id: str) -> Tier:
        """Locality tier between two volumes (cheapest tier compares smallest)."""
        return locality(self._topology[src_id], self._topology[dst_id])

    def nearest(self, candidates: Sequence[str], to: str) -> Optional[str]:
        """The closest of ``candidates`` to ``to`` (locality, id tie-break)."""
        if not candidates:
            return None
        return min(candidates, key=lambda v: (int(self.locality(v, to)), v))

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
