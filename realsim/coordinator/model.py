"""The new read coordinator (a *model*) driving the real TorchStore directory.

This is the capability ``realsim`` prototypes. It sits in the **read path**:
given ``m`` readers wanting overlapping data, it

1. **consults the real controller directory** -- ``FakeControllerHandle.locate_volumes``,
   which runs the real ``Controller.locate_volumes`` endpoint body over the real
   ``Trie`` state -- to see which volumes hold each key, then
2. **issues the fetches through the real ``LocalClient``** planning core.

Everything under the coordinator is real TorchStore code (client planning +
controller directory + in-memory transport); the coordinator itself is the one
*new* component being designed, so it is the one piece kept as a model.

Naive vs. a dedup/cache-aware policy
------------------------------------
The routing lives entirely in the pluggable :class:`ReadPolicy`, built on the
**real** directory/client:

* :class:`NaivePolicy` (shipped) -- every reader fetches independently. In a
  synchronized burst they all locate the origin before anyone finishes, so each
  reader pulls from the origin volume: ``m x`` fabric, the baseline.
* :class:`ReadPolicy` (the seam) -- override :meth:`ReadPolicy.after_fetch` to
  register a finished reader back into the **real** directory
  (``controller_handle.notify_put_batch``, exactly the real read-through path) so
  later readers' ``locate_volumes`` find a *closer peer* instead of the origin --
  the dedup/cache-aware routing, expressed purely by mutating real directory
  state. ``dedup_sim``'s ``DedupPolicy`` is exactly such an override.

Multi-client transport seam
---------------------------
The wiring a burst runs on -- real volumes, one real ``LocalClient`` per node, the
shared resource registry, and the single ``create_transport_buffer`` substitution
that resolves the calling reader's source endpoint -- is not burst-specific, so it
lives in :class:`realsim.mesh.Mesh`. The coordinator is a burst-shaped *consumer*
of a mesh: it adds the read path, the policy seam, and the fabric accounting, and
nothing else.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from realsim.mesh import Mesh
from realsim.seams import factory
from realsim.seams.transport import Endpoint
from sim_common.cost_model import MachineProfile
from sim_common.resources import ResourceRegistry
from sim_common.trace import Trace


@dataclass
class Reader:
    """One participant in a read burst.

    Attributes:
        id: stable reader id (also its co-located volume id).
        client: the real ``LocalClient`` (from a ``RealClientAdapter``).
        endpoint: this reader's locality :class:`~sim_common.topology.Endpoint`.
        tensor_slice_spec: optional slice to fetch (``None`` = whole tensor).
    """

    id: str
    client: Any
    endpoint: Endpoint
    tensor_slice_spec: Any = None


@dataclass
class BurstMetrics:
    """Fabric/wallclock accounting for one burst (mirrors ``dedup_sim`` Metrics).

    ``fabric_bytes`` counts bytes served by an *origin* volume (a volume that
    held the key before the burst) -- the fabric cost the coordinator exists to
    reduce. ``total_get_bytes`` counts every byte delivered to a reader; for the
    naive policy the two are equal (``m x``), a dedup policy would drive
    ``fabric_bytes`` toward the 1x union while ``total_get_bytes`` stays ``m x``.
    """

    fabric_bytes: int = 0
    total_get_bytes: int = 0
    wallclock: float = 0.0
    readers_total: int = 0
    readers_done: int = 0
    # (src_id, dst_id, key) fetch edges, for sim_common.report.render_tree.
    edges: List[Tuple[str, str, str]] = field(default_factory=list)


class ReadPolicy:
    """Pluggable read-routing policy (the dedup seam).

    The base policy is a pass-through: readers use whatever the real directory
    returns (the origin), and nothing is registered back. Subclass and override
    either seam to change routing:

    * :meth:`after_fetch` -- turn a finished reader into a directory source
      (read-through), so later ``locate_volumes`` calls can route to a peer; and
    * :meth:`run_burst` -- take full control of how the burst executes (the
      default fans every reader out concurrently, so a synchronized burst pulls
      ``m x`` from the origin; a dedup/cache-aware policy overrides this to
      serialize/stage the burst into a 1x-fabric chain/tree). ``dedup_sim``'s
      ``DedupPolicy`` is exactly such an override.
    """

    name = "naive"

    def note_locate(
        self, key: str, located: Dict[str, Dict[str, Any]]
    ) -> str:
        """Return a human-readable summary of a directory lookup for the trace."""
        volumes = sorted(located.get(key, {}))
        return f"locate {key} -> volumes {volumes}"

    async def run_burst(
        self, coordinator: "ReadCoordinator", readers: List[Reader], key: str
    ) -> Dict[str, Any]:
        """Execute a synchronized read burst; return ``reader_id -> payload``.

        Default: the naive concurrent fan-out (every reader locates and fetches
        at once, so in a synchronized burst they all pull from the origin --
        ``m x`` fabric). A routing policy overrides this to consult the real
        directory and stage the burst so each unique byte crosses the fabric
        once (registering read-through peers so later readers pull from a peer).
        """
        return await coordinator._naive_burst(readers, key)

    async def after_fetch(self, controller_handle: Any, reader: Reader, key: str) -> None:
        """Hook after ``reader`` finishes fetching ``key``.

        Naive: no-op. A dedup/cache-aware policy would register ``reader`` as a
        new directory source here (e.g. via ``controller_handle.notify_put_batch``),
        the real read-through path, so subsequent readers locate a closer peer.
        """


class NaivePolicy(ReadPolicy):
    """Baseline: independent fetches, no read-through (the ``m x`` fabric case)."""


class ReadCoordinator:
    """Coordinates a synchronized read burst over the real directory/client.

    The real objects the burst runs on come from ``mesh``; this class adds only
    the burst-specific parts -- the read path, the :class:`ReadPolicy` seam, and
    the fabric accounting in :class:`BurstMetrics`. It claims the mesh's
    :attr:`~realsim.mesh.Mesh.on_transfer` hook for that accounting, so one mesh
    backs at most one coordinator.

    Args:
        mesh: the :class:`realsim.mesh.Mesh` holding the real controller
            directory, per-node volumes/clients, resource registry, and the
            shared transport factory.
        origin_ids: endpoint ids that held data before the burst; transfers whose
            source is one of these count as fabric bytes.
        policy: a :class:`ReadPolicy` (defaults to :class:`NaivePolicy`).
        metrics: shared :class:`BurstMetrics` (created if omitted).
    """

    def __init__(
        self,
        mesh: Mesh,
        *,
        origin_ids: Optional[set] = None,
        policy: Optional[ReadPolicy] = None,
        metrics: Optional[BurstMetrics] = None,
    ) -> None:
        self.mesh = mesh
        self.origin_ids = set(origin_ids) if origin_ids is not None else set()
        self.policy = policy if policy is not None else NaivePolicy()
        self.metrics = metrics if metrics is not None else BurstMetrics()
        mesh.on_transfer = self._on_transfer

    # -- the mesh's shared pieces, surfaced for policies ------------------- #
    @property
    def controller_handle(self) -> Any:
        """The real ``Controller`` directory behind the actor surface."""
        return self.mesh.handle

    @property
    def topology(self) -> Dict[str, Endpoint]:
        """``volume_id -> Endpoint`` for transfer-cost locality."""
        return self.mesh.topology

    @property
    def trace(self) -> Trace:
        """The run's shared :class:`~sim_common.trace.Trace`."""
        return self.mesh.trace

    @property
    def profile(self) -> MachineProfile:
        """The target-machine :class:`~sim_common.cost_model.MachineProfile`."""
        return self.mesh.profile

    @property
    def registry(self) -> ResourceRegistry:
        """The run's shared :class:`~sim_common.resources.ResourceRegistry`."""
        return self.mesh.registry

    def installed(self) -> Iterator[Mesh]:
        """Install the mesh's shared transport factory for the burst.

        A :class:`ReadPolicy` that overrides :meth:`ReadPolicy.run_burst` must
        wrap its execution in this, exactly as :meth:`_naive_burst` does.
        """
        return self.mesh.installed()

    def _on_transfer(self, kind, src_id, dst_id, nbytes, cost) -> None:
        """Structured fabric-byte accounting (see :class:`BurstMetrics`)."""
        if kind != "get":
            return
        self.metrics.total_get_bytes += nbytes
        if src_id in self.origin_ids:
            self.metrics.fabric_bytes += nbytes
        if nbytes > 0:
            self.metrics.edges.append((src_id, dst_id, dst_id))

    # -- the read path ----------------------------------------------------- #
    async def _locate(self, key: str) -> Dict[str, Dict[str, Any]]:
        """Consult the REAL controller directory for ``key``."""
        located = await self.controller_handle.locate_volumes.call_one([key])
        self.trace.record(
            asyncio.get_running_loop().time(), "coord", self.policy.note_locate(key, located)
        )
        return located

    async def _fetch_one(self, reader: Reader, key: str) -> Any:
        """Run one reader's real ``client.get`` with its source endpoint bound."""
        factory.bind_source(reader.endpoint)
        result = await reader.client.get(key, tensor_slice_spec=reader.tensor_slice_spec)
        await self.policy.after_fetch(self.controller_handle, reader, key)
        self.metrics.readers_done += 1
        self.trace.record(
            asyncio.get_running_loop().time(), "coord", f"reader {reader.id} done"
        )
        return result

    async def run_burst(self, readers: List[Reader], key: str) -> Dict[str, Any]:
        """Run a synchronized burst for ``key``, delegating routing to the policy.

        Returns ``reader_id -> fetched payload``. Accounts the readers, then hands
        execution to :meth:`ReadPolicy.run_burst` (the routing seam): the default
        :class:`NaivePolicy` runs the concurrent fan-out in :meth:`_naive_burst`
        (``m x`` fabric), while a dedup/cache-aware policy stages the burst into a
        1x-fabric chain/tree. Both fill :attr:`metrics` and the shared trace.
        """
        self.metrics.readers_total += len(readers)
        return await self.policy.run_burst(self, readers, key)

    async def _naive_burst(self, readers: List[Reader], key: str) -> Dict[str, Any]:
        """The naive concurrent fan-out: locate ``key``, then gather every reader.

        In a synchronized burst all readers locate the origin before anyone
        finishes, so each pulls from the origin volume -- the ``m x`` fabric
        baseline. Records the directory lookup, a per-reader completion, and (via
        the transport) every transfer into the shared trace.
        """
        with self.installed():
            await self._locate(key)
            loop = asyncio.get_running_loop()
            self.trace.record(loop.time(), "coord", f"burst: {len(readers)} readers get {key!r}")
            results = await asyncio.gather(
                *(self._fetch_one(r, key) for r in readers)
            )
            self.metrics.wallclock = loop.time()
        return {r.id: res for r, res in zip(readers, results)}
