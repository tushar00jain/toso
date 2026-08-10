"""The read-only observation a policy is handed: :class:`View`.

A :class:`~realsim.policy.Policy` never touches a client, a volume or the
mesh -- it is given a ``View`` and returns a decision. The ``View`` is the sensor
half of that contract: awaited *reads* of state that already exists, and no
mutation of any kind.

What the base view offers, and why that is all it offers
--------------------------------------------------------
* :meth:`locate` -- the real directory answer for a set of keys, read straight
  from the real ``Controller`` body. It deliberately bypasses the controller's
  routing hook (see :mod:`realsim.policy`): a sensor must report the directory
  as it *is*, and a policy reading its own answer back would recurse.
* :meth:`holders` / :meth:`topology` / :meth:`endpoint` / :meth:`locality` --
  who holds a key and how far away they are, the two inputs every source
  decision has needed so far.
* :meth:`now` -- the run's virtual clock.

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
:meth:`View.of` takes anything with the mesh's shape (``handle`` + ``topology``);
the type is duck-typed rather than imported so that ``control/`` code may hold a
``View`` without importing the mesh.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from sim_common.topology import Endpoint, locality, Tier


class View:
    """Awaited, read-only observation of the real directory and topology.

    Args:
        handle: the real controller directory handle (a
            :class:`~realsim.seams.controller_handle.FakeControllerHandle`).
        topology: ``volume_id -> Endpoint``; the volume id is the directory
            identity, the endpoint is what locality is priced against.
    """

    def __init__(self, handle: Any, topology: Dict[str, Endpoint]) -> None:
        self._handle = handle
        self._topology = dict(topology)

    @classmethod
    def of(cls, mesh: Any) -> "View":
        """Build the view over a :class:`realsim.mesh.Mesh` (duck-typed)."""
        return cls(mesh.handle, mesh.topology)

    # -- directory ---------------------------------------------------------- #
    async def locate(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``key -> {volume_id -> StorageInfo}`` from the REAL directory.

        Missing keys are simply absent (``missing_ok=True``): a sensor reports
        what is there, it does not raise at the observer. Reads the raw
        controller body, so the routing hook a policy may be installed behind is
        not re-entered.
        """
        if not keys:
            return {}
        return await self._handle.locate_raw(list(keys), missing_ok=True)

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
        """The run's virtual time (the loop's clock -- never a wall clock)."""
        return asyncio.get_running_loop().time()
