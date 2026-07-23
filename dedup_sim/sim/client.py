"""Simulated client -- the only entry point a "user" touches.

``Client.get(reader, key, need)`` mirrors ``ts.get`` exactly: no promise/wait
arguments leak to the caller. The coordinator is invoked *inside* ``get``; the
client then executes the returned plan (start now, or park on a dependency) and
records completion. ``Client.put`` seeds the index (trainer side).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .coordinator import Fetch
from .cost import locality, TIER_LABEL, transfer_time
from sim_common.engine import Sim
from .model import decompose, Region, region_bytes, region_str, Volume
from .trace import Metrics, Trace


class Client:
    """In-process client that plans via the coordinator and executes fetches."""

    def __init__(self, sim: Sim, index, coordinator, topo: Dict[str, Volume],
                 dtype_bytes: int, atomics: List[Region], trace: Trace,
                 metrics: Metrics) -> None:
        self.sim = sim
        self.index = index
        self.coordinator = coordinator
        self.topo = topo
        self.dtype_bytes = dtype_bytes
        self.atomics = atomics
        self.trace = trace
        self.metrics = metrics

    def put(self, key: str, volume_id: str, region: Region) -> None:
        """Seed the storage index (trainer-side put); mirrors ``ts.put``."""
        self.index.notify_put(key, volume_id, region)

    def get(self, reader: Volume, key: str, need: Iterable[Region]) -> None:
        """Fetch ``need`` for ``reader`` -- the user-facing entry point.

        No promise/wait args: the coordinator (and any parking) is entirely
        internal. Records the reader as done when it holds all its atomics.
        """
        atomics = decompose(need, self.atomics)
        plan = self.coordinator.plan_get(reader, key, atomics)
        self.metrics.readers_total += 1

        # Atomics already held by the reader are skipped by the planner; count
        # them as already-assembled so completion accounting is exact.
        planned_regions = {f.region for f in plan}
        for region in atomics:
            if region not in planned_regions:
                self.metrics.assembled[reader.id].add(region)

        for fetch in plan:
            self._trace_plan(reader, key, fetch)

        remaining = {"n": len(plan)}
        if not plan:
            self._reader_done(reader)
            return

        for fetch in plan:
            if fetch.data_dep is None:
                self._acquire(reader, key, fetch, remaining)
            else:
                self.trace.record(
                    self.sim.now, "COORD",
                    f"park {reader.id} {region_str(fetch.region)} "
                    f"-> wait promise from {fetch.src}",
                )
                fetch.data_dep.add_callback(
                    lambda f=fetch: self._acquire(reader, key, f, remaining)
                )

    def _acquire(self, reader: Volume, key: str, fetch: Fetch,
                 remaining: Dict[str, int]) -> None:
        """The source now holds the data; acquire a serving slot then transfer.

        The slot request enforces the fan-out cap at execution time (excess
        consumers queue on the source rather than re-pulling from the trainer).
        """
        self.coordinator.request_slot(
            key, fetch.src,
            lambda: self._start(reader, key, fetch, remaining),
        )

    def _trace_plan(self, reader: Volume, key: str, fetch: Fetch) -> None:
        region = region_str(fetch.region)
        src = self.topo[fetch.src]
        if src.is_trainer:
            note = f"({reader.id} designated source)"
            verb = "PULL"
        elif fetch.data_dep is not None:
            note = "(park: wait promise)"
            verb = "GET "
        else:
            note = "(cache hit)"
            verb = "GET "
        self.trace.record(
            self.sim.now, "COORD",
            f"plan {reader.id} key={key} -> {verb} {region} "
            f"from {fetch.src}  {note}",
        )

    def _start(self, reader: Volume, key: str, fetch: Fetch,
               remaining: Dict[str, int]) -> None:
        src = self.topo[fetch.src]
        dst = self.topo[fetch.dst]
        nbytes = region_bytes(fetch.region, self.dtype_bytes)
        dt = transfer_time(src, dst, nbytes)
        tier_label = TIER_LABEL[locality(src, dst)] if src.id != dst.id else "local"
        self.metrics.edges.append((fetch.src, fetch.dst, fetch.region))
        self.trace.record(
            self.sim.now, "XFER",
            f"{fetch.src} -> {fetch.dst}  {region_str(fetch.region)} {nbytes}B  "
            f"{tier_label:<10} start (eta {self.sim.now + dt:.3f})",
        )

        def done() -> None:
            note = ""
            if fetch.publish_promise is not None:
                note = f"; {fetch.dst} publishes -> resolve, release waiters"
            self.trace.record(
                self.sim.now, "XFER",
                f"{fetch.src} -> {fetch.dst}  {region_str(fetch.region)} done{note}",
            )
            if src.is_trainer:
                self.metrics.fabric_bytes += nbytes
            self.metrics.assembled[fetch.dst].add(fetch.region)
            self.coordinator.on_fetch_complete(key, fetch)
            self.coordinator.release_slot(key, fetch.src)
            remaining["n"] -= 1
            if remaining["n"] == 0:
                self._reader_done(reader)

        self.sim.schedule(dt, done)

    def _reader_done(self, reader: Volume) -> None:
        self.metrics.readers_done += 1
        self.metrics.wallclock = max(self.metrics.wallclock, self.sim.now)
        if self.metrics.readers_done == self.metrics.readers_total:
            self.trace.record(self.sim.now, "DONE", "all readers satisfied")
