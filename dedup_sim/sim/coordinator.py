"""Coordinators under test: ``NaiveCoordinator`` and ``DedupCoordinator``.

Both share an interface so scenarios/tests can swap them:

    plan_get(reader, key, need_atomics) -> list[Fetch]   # runs atomically
    request_slot(key, vol, cb) / release_slot(key, vol)  # execution-time cap
    on_fetch_complete(key, fetch) -> None
    bump_version(key) -> None

``plan_get`` models the coordinator's serialized Monarch mailbox: in the DES it
runs to completion before the next event, so a burst of gets becomes a total
order and in-flight fetches can be treated as *promised* cache sources.

Two counters keep the fan-out cap honest:
  * ``planned[vol]``  -- a plan-time tally used to shape a balanced tree/chain
    (selection prefers a source still under the cap). Persisted per version.
  * ``serving[vol]``  -- actual concurrent in-flight serves, bounded by the cap
    via :meth:`request_slot` / :meth:`release_slot`. Excess consumers queue
    (they never re-pull from the trainer), so fabric stays 1x while no source
    ever exceeds ``FANOUT_CAP`` concurrent serves.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .cost import locality
from .engine import Promise, Sim
from .model import Region, Volume
from .store_index import StoreIndex


@dataclass
class Fetch:
    """One planned transfer of an atomic region to a reader's volume.

    - ``data_dep`` (optional promise): resolves when the *source* holds the
      data; the client parks the reader until then (the park/release mechanic).
      ``None`` when the source already has the data (trainer or a present peer).
    - ``publish_promise``: resolving it on completion turns this reader into a
      cache source (read-through) and releases peers depending on it.
    """

    region: Region
    src: str  # source volume id
    dst: str  # reader volume id
    data_dep: Optional[Promise] = None
    publish_promise: Optional[Promise] = None


def _trainer_holding(index: StoreIndex, topo: Dict[str, Volume], key: str,
                     region: Region) -> str:
    """Return the (deterministic) trainer volume that holds ``region``.

    Source of truth for the first pull: the smallest-id trainer volume whose
    stored region covers ``region``.
    """
    a, b = region
    cands = [
        vol
        for vol, regs in index.locate(key).items()
        if topo[vol].is_trainer and any(s <= a and b <= e for (s, e) in regs)
    ]
    assert cands, f"no trainer volume holds {region} for key {key!r}"
    return min(cands)


class NaiveCoordinator:
    """Baseline: every reader pulls every needed region straight from a trainer.

    No dedup, no cache, no promises -- used to print the ``m x`` fabric baseline.
    Fan-out is unbounded (there is no peer exchange to balance).
    """

    def __init__(self, sim: Sim, index: StoreIndex, topo: Dict[str, Volume]) -> None:
        self.sim = sim
        self.index = index
        self.topo = topo
        self.peak_serving = 0  # not meaningful for naive; kept for interface parity

    def plan_get(self, reader: Volume, key: str,
                 need_atomics: List[Region]) -> List[Fetch]:
        """Plan: pull each atomic region directly from a trainer volume."""
        plan: List[Fetch] = []
        for region in sorted(need_atomics):
            tvol = _trainer_holding(self.index, self.topo, key, region)
            plan.append(Fetch(region=region, src=tvol, dst=reader.id))
        return plan

    def request_slot(self, key: str, vol: str, cb: Callable[[], None]) -> None:
        """No cap in the baseline: start immediately."""
        cb()

    def release_slot(self, key: str, vol: str) -> None:
        """No cap to release."""

    def on_fetch_complete(self, key: str, fetch: Fetch) -> None:
        """No cache to maintain for the naive baseline."""

    def bump_version(self, key: str) -> None:
        """No versioned cache state to invalidate."""


@dataclass
class _KeyState:
    """Per-(key, version) dedup state."""

    sources: Dict[Region, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )  # region -> volumes that HAVE it (present)
    promises: Dict[Region, List[Tuple[str, Promise]]] = field(
        default_factory=lambda: defaultdict(list)
    )  # region -> [(volume, promise)] in-flight; will HAVE it
    planned: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )  # volume -> serves assigned during planning (tree-shaping tally)
    serving: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )  # volume -> actual concurrent in-flight serves
    slot_waiters: Dict[str, List[Callable[[], None]]] = field(
        default_factory=lambda: defaultdict(list)
    )  # volume -> FIFO of callbacks waiting for a free serving slot


class DedupCoordinator:
    """Dynamic dedup coordinator: cache + routing + queuing.

    The first reader to need a region pulls it from a trainer and becomes its
    (promised) cache source; later readers of the same region are routed to a
    present-or-promised peer (never re-pulling from the trainer), building a
    balanced tree/chain incrementally under a fan-out cap. Source choice prefers
    locality (cost tier). State is per ``(key, version)``; ``bump_version``
    drops stale cache.
    """

    def __init__(self, sim: Sim, index: StoreIndex, topo: Dict[str, Volume],
                 fanout_cap: int = 1) -> None:
        self.sim = sim
        self.index = index
        self.topo = topo
        self.cap = fanout_cap
        self._version: Dict[str, int] = {}
        self._state: Dict[Tuple[str, int], _KeyState] = {}
        self.peak_serving = 0

    # -- versioning ------------------------------------------------------- #
    def bump_version(self, key: str) -> None:
        """Bump the commit epoch for ``key``, invalidating cached sources.

        A subsequent burst finds empty ``sources``/``promises`` for the new
        version and re-pulls from the trainer (the only ``is_trainer`` entries
        in the index).
        """
        self._version[key] = self._version.get(key, 0) + 1

    def _state_for(self, key: str) -> _KeyState:
        ver = self._version.setdefault(key, 0)
        skey = (key, ver)
        if skey not in self._state:
            self._state[skey] = _KeyState()
        return self._state[skey]

    # -- planning --------------------------------------------------------- #
    def _pick(self, cands: List[str], reader: Volume, planned: Dict[str, int]) -> str:
        """Choose a source: prefer under-cap, then lowest ``(tier, load, id)``.

        ``cands`` is guaranteed non-empty. Preferring an under-cap source shapes
        the balanced tree; if every candidate is already at the cap we still
        pick one (the least loaded) and the transfer queues at execution time --
        we never fall back to the trainer once a peer holds/promises the region.
        """
        under_cap = [v for v in cands if planned[v] < self.cap]
        if under_cap:
            # Fill a source up to the cap before moving on (shallow tree):
            # lowest cost tier, then lowest id.
            return min(under_cap,
                       key=lambda v: (int(locality(self.topo[v], reader)), v))
        # Every candidate is at the cap: pick the least-loaded so the forced
        # queue spreads evenly. The transfer still queues at execution time --
        # we never re-pull from the trainer once a peer holds/promises it.
        return min(cands, key=lambda v: (int(locality(self.topo[v], reader)),
                                        planned[v], v))

    def plan_get(self, reader: Volume, key: str,
                 need_atomics: List[Region]) -> List[Fetch]:
        """Plan a get atomically (models the serialized actor mailbox)."""
        st = self._state_for(key)
        plan: List[Fetch] = []
        for region in sorted(need_atomics):
            if reader.id in st.sources.get(region, set()):
                continue  # reader already holds this region

            present = [v for v in sorted(st.sources.get(region, set()))
                       if v != reader.id]
            promised = [(v, p) for (v, p) in st.promises.get(region, [])
                        if v != reader.id]
            peer_ids = present + [v for (v, _) in promised]

            # This reader will hold the region after fetching, so it becomes a
            # (promised) source later peers can be routed to -- read-through
            # population (S4) that grows the fan-out tree past the first hop.
            pub = Promise(self.sim, label=f"{key}:{region}@{reader.id}")
            st.promises[region].append((reader.id, pub))

            if peer_ids:
                pick = self._pick(peer_ids, reader, st.planned)
                st.planned[pick] += 1
                data_dep = None if pick in present else next(
                    p for (v, p) in promised if v == pick
                )
                plan.append(Fetch(region=region, src=pick, dst=reader.id,
                                  data_dep=data_dep, publish_promise=pub))
            else:
                tvol = _trainer_holding(self.index, self.topo, key, region)
                st.planned[tvol] += 1
                plan.append(Fetch(region=region, src=tvol, dst=reader.id,
                                  data_dep=None, publish_promise=pub))
        return plan

    # -- execution-time cap (slot queue) ---------------------------------- #
    def request_slot(self, key: str, vol: str, cb: Callable[[], None]) -> None:
        """Acquire a serving slot on ``vol``, or queue until one frees.

        Guarantees actual concurrent serves never exceed ``FANOUT_CAP``.
        """
        st = self._state_for(key)
        if st.serving[vol] < self.cap:
            st.serving[vol] += 1
            self.peak_serving = max(self.peak_serving, st.serving[vol])
            cb()
        else:
            st.slot_waiters[vol].append(cb)

    def release_slot(self, key: str, vol: str) -> None:
        """Free a serving slot on ``vol`` and wake the next queued waiter."""
        st = self._state_for(key)
        if st.serving[vol] > 0:
            st.serving[vol] -= 1
        if st.slot_waiters[vol]:
            nxt = st.slot_waiters[vol].pop(0)
            st.serving[vol] += 1
            self.peak_serving = max(self.peak_serving, st.serving[vol])
            self.sim.schedule(0.0, nxt)

    def on_fetch_complete(self, key: str, fetch: Fetch) -> None:
        """Register the read-through and resolve the reader's publish promise."""
        st = self._state_for(key)
        st.sources[fetch.region].add(fetch.dst)
        self.index.notify_put(key, fetch.dst, fetch.region)  # S4 read-through
        if fetch.publish_promise is not None:
            fetch.publish_promise.resolve()
