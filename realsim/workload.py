"""What a run performs on the clock: :class:`Workload`.

A workload is *the work*, and nothing else. It knows the topology it wants and
how to turn that into work items; it assembles no stack, installs no policy,
builds no data plane and renders nothing. Those are :class:`~realsim.simulation.
Simulation`'s, the capability's and :class:`~realsim.reporting.Report`'s jobs
respectively.

That split is the point. The previous ``Workload`` returned ``(plane, items)``
from a ``build(sim)`` hook, so every workload also constructed schedulers, stores
and planes onto the stack it was handed -- assembly, which is what a
``Simulation`` is for. It additionally carried a ``result()`` hook that copied
its own attributes into a result subclass. Both are gone: a workload yields items.

    class MyBurst(Workload):
        def __init__(self, m):
            super().__init__(_topology(m))
            self.m = m

        def items(self, sim):
            return [WorkItem(id=f"r{i}") for i in range(self.m)]

:meth:`items` takes the assembled ``sim`` because an item's work is ordinary user
code against a real client, and the clients only exist once the stack does.
:meth:`prepare` is for work that precedes the items *on the clock* -- seeding a
key before a read burst -- which is part of the simulated timeline and therefore
part of the workload, not of the stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence, TYPE_CHECKING

from proposed import Endpoint

from realsim.runner import WorkItem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from realsim.simulation import Simulation

__all__ = ["Workload"]


class Workload(ABC):
    """The work a run performs. Assembles nothing, renders nothing.

    Args:
        topology: ``node_id -> Endpoint``. The node id is also its storage-volume
            id in the real directory. Known before the stack exists, because
            :class:`~realsim.simulation.Simulation` is built from it.
    """

    def __init__(self, topology: Dict[str, Endpoint]) -> None:
        self.topology = topology

    @abstractmethod
    def items(self, sim: "Simulation") -> Sequence[WorkItem]:
        """The work items to release, given the assembled stack."""

    async def prepare(self, sim: "Simulation") -> None:
        """Work that precedes the items on the clock. Default: nothing."""
