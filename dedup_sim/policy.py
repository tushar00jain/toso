"""The dedup read-routing policy: a real :class:`realsim.ReadPolicy`.

``DedupPolicy`` plugs into realsim's :class:`~realsim.coordinator.model.ReadCoordinator`
and turns a synchronized read burst into a **1x-fabric** transfer over the **real**
TorchStore directory -- the dedup capability, expressed entirely by consulting and
mutating real controller state, on real types throughout.

How 1x is achieved on the real directory
----------------------------------------
The baseline :class:`~realsim.coordinator.model.NaivePolicy` fans every reader out
concurrently; in a synchronized burst they all ``locate_volumes`` the origin before
anyone finishes, so each pulls from the origin volume -- ``m x`` fabric.

``DedupPolicy`` instead **serializes the burst into a chain/tree**. For each reader,
in order, it:

1. **consults the real directory** (``FakeControllerHandle.locate_volumes`` -> the
   real ``Controller.locate_volumes`` body over the real ``Trie``) to see which
   volumes already hold the key, then
2. **chooses a source** -- the first requester pulls from an *origin* volume (a
   volume that held the key before the burst); every later reader is routed to a
   **peer** that has since become a source, never back to the origin. Selection
   prefers locality (cost tier) and respects a fan-out cap (cap 1 = a chain, cap
   >= 2 = a shallow tree), and
3. after the fetch, does the **real read-through**: the reader ``put``s the key
   into its own co-located volume (a zero-fabric local write) which, through the
   real ``client.put`` path, both stores the payload there and calls the real
   ``notify_put_batch`` -- so the reader is now a real directory source for the
   next reader.

Because exactly one reader ever pulls from an origin volume, the only transfer
whose source is an origin is that first hop: ``fabric_bytes == 1x`` the payload
(the union), regardless of the fan-out cap. Every other reader's source is a peer
on the reader side, which the fabric accounting (source in ``origin_ids``) does not
count. Contrast: :class:`NaivePolicy` leaves ``fabric_bytes == m x``.

Forcing the real client to a chosen source
------------------------------------------
The real ``LocalClient`` picks the volume to read from purely by what
``locate_volumes`` returns (for a whole tensor/object it takes the first entry).
It has no locality/source argument. So the routing decision is expressed by
**scoping each reader's directory view** to the policy-chosen volume: each reader's
client talks to a :class:`_RoutingControllerHandle` that calls the *real* directory
and then narrows the result to the one chosen volume (returning the **real**
``StorageInfo``, unmodified). All other controller endpoints (notably
``notify_put_batch`` for read-through) pass straight through to the real handle, so
the real directory remains the single source of truth.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List

from realsim.coordinator.model import Reader, ReadCoordinator, ReadPolicy
from sim_common.topology import locality


class _RoutedEndpoint:
    """Mimics a Monarch endpoint's ``.call`` / ``.call_one`` awaitable surface.

    Mirrors ``realsim.seams.controller_handle._ControllerEndpoint`` (kept local so
    this module does not depend on that private name); it wraps a plain coroutine
    so the client can invoke it as ``locate_volumes.call_one(...)``.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)

    async def call_one(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)


class _RoutingControllerHandle:
    """A per-reader view of the real controller handle that scopes ``locate_volumes``.

    ``locate_volumes`` is answered by the *real* handle and then narrowed to the
    single volume the :class:`DedupPolicy` chose for this reader (returning the real
    ``StorageInfo`` untouched), so the real client reads from that source. Every
    other endpoint -- including ``notify_put_batch`` used by read-through -- is
    delegated to the real handle unchanged.
    """

    def __init__(self, real_handle: Any, policy: "DedupPolicy", reader_id: str) -> None:
        self._real = real_handle
        self._policy = policy
        self._reader_id = reader_id
        self.locate_volumes = _RoutedEndpoint(self._scoped_locate)

    async def _scoped_locate(self, keys: List[str], **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        full = await self._real.locate_volumes.call_one(keys, **kwargs)
        return self._policy.scope_locate(self._reader_id, full)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything except the scoped locate_volumes (an instance attr,
        # so it is found before __getattr__ ever fires) to the real handle.
        return getattr(self._real, name)


class DedupPolicy(ReadPolicy):
    """1x-fabric dedup routing over the real directory (a real ``ReadPolicy``).

    Args:
        put_value: the key's payload carrier (a ``device="meta"`` tensor or a
            :class:`~realsim.seams.transport.TensorDescriptor`) that a reader
            re-``put``s into its own volume for read-through. Allocation-free.
        fanout_cap: max concurrent peers a single source may feed (1 = chain,
            >= 2 = shallow tree). The fabric stays 1x for any cap.
    """

    name = "dedup"

    def __init__(self, put_value: Any, *, fanout_cap: int = 1) -> None:
        self._put_value = put_value
        self.cap = fanout_cap
        # reader_id -> chosen source volume id, consulted by the routing handle.
        self._route: Dict[str, str] = {}
        # volume id -> peers it has been assigned to serve (tree-shaping tally).
        self._planned: Dict[str, int] = defaultdict(int)
        # volume ids that held the key before the burst (the fabric origins).
        self._origin_vols: set = set()

    # -- routing view consulted by _RoutingControllerHandle ----------------- #
    def scope_locate(
        self, reader_id: str, full: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Narrow a real locate result to this reader's chosen source volume."""
        chosen = self._route.get(reader_id)
        if chosen is None:
            return full
        scoped: Dict[str, Dict[str, Any]] = {}
        for key, volume_map in full.items():
            if chosen in volume_map:
                scoped[key] = {chosen: volume_map[chosen]}
            else:
                scoped[key] = volume_map
        return scoped

    def install_on(self, reader: Reader, real_handle: Any) -> None:
        """Point ``reader``'s real client at a routing handle for this policy."""
        reader.client._controller = _RoutingControllerHandle(
            real_handle, self, reader.id
        )

    # -- routing plan: a 1x read-through chain/tree ------------------------- #
    def _plan_tree(
        self, readers: List[Reader], coordinator: ReadCoordinator
    ) -> List[List[Reader]]:
        """Assign each reader a source and return readers grouped by tree depth.

        The root reader pulls from the closest *origin* volume (the single fabric
        hop). Every other reader is attached, FIFO, to a peer that is still under
        the fan-out cap -- a ``cap``-ary read-through tree (cap 1 = a chain). No
        non-root reader is ever routed to an origin, so fabric stays 1x. Readers at
        the same depth have sources that all finished at the previous depth, so a
        depth level can execute concurrently.
        """
        topo = coordinator.topology
        root = readers[0]
        origins = sorted(self._origin_vols)
        # Root pulls from the closest origin (locality tie-broken by id).
        self._route[root.id] = min(
            origins, key=lambda v: (int(locality(topo[v], topo[root.id])), v)
        )
        depth: Dict[str, int] = {root.id: 0}
        avail: "deque[str]" = deque([root.id])  # peers still under the fan-out cap
        for reader in readers[1:]:
            parent = avail[0]
            self._route[reader.id] = parent
            self._planned[parent] += 1
            depth[reader.id] = depth[parent] + 1
            if self._planned[parent] >= self.cap:
                avail.popleft()
            avail.append(reader.id)

        levels: List[List[Reader]] = [[] for _ in range(max(depth.values()) + 1)]
        for reader in readers:
            levels[depth[reader.id]].append(reader)
        return levels

    # -- the routing seam: stage the burst into a 1x chain/tree ------------- #
    async def run_burst(
        self, coordinator: ReadCoordinator, readers: List[Reader], key: str
    ) -> Dict[str, Any]:
        """Stage the burst so each unique byte crosses the fabric exactly once.

        Consult the real directory for the origins, plan a read-through tree, then
        execute it depth level by depth level -- each level's readers fetch
        concurrently from sources their (already-finished) parents populated. Only
        the root touches an origin, so ``fabric_bytes`` is the 1x union.
        """
        results: Dict[str, Any] = {}
        with coordinator._shared_transport():
            located = await coordinator._locate(key)
            # Volumes that already hold the key are the fabric origins; every
            # read-through peer registered during the burst is not an origin.
            self._origin_vols = set(located.get(key, {}))
            loop = asyncio.get_running_loop()
            coordinator.trace.record(
                loop.time(),
                "coord",
                f"dedup burst: {len(readers)} readers get {key!r} (cap={self.cap})",
            )
            for depth, level in enumerate(self._plan_tree(readers, coordinator)):
                for reader in level:
                    coordinator.trace.record(
                        loop.time(),
                        "coord",
                        f"route reader {reader.id} <- {self._route[reader.id]} "
                        f"(level {depth})",
                    )
                level_results = await asyncio.gather(
                    *(coordinator._fetch_one(r, key) for r in level)
                )
                for reader, res in zip(level, level_results):
                    results[reader.id] = res
            coordinator.metrics.wallclock = loop.time()
        return results

    async def after_fetch(
        self, controller_handle: Any, reader: Reader, key: str
    ) -> None:
        """Real read-through: the reader stores the key into its own volume.

        This drives the real ``client.put`` path -- a zero-fabric local write into
        the reader's co-located volume plus a real ``notify_put_batch`` -- so the
        reader becomes a real directory source the next reader can be routed to.
        """
        await reader.client.put(key, self._put_value)
