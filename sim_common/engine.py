"""Discrete-event simulation engine: ``Sim`` and ``Promise``.

Single-threaded, deterministic. All time flows through :meth:`Sim.schedule`;
there is no wall-clock, no threads, no asyncio and no randomness. Events are
ordered by ``(time, seq)`` where ``seq`` is a monotonic insertion counter that
provides a total order (so the trace is byte-identical across runs).
"""

from __future__ import annotations

import heapq
from typing import Callable, List, Tuple

__all__ = ["Sim", "Promise"]


class Sim:
    """A minimal discrete-event simulator.

    The event queue is a binary heap of ``(time, seq, callback, label)``.
    ``time`` is the simulated firing time and ``seq`` is a monotonically
    increasing integer that breaks ties deterministically (FIFO among events
    scheduled for the same instant). The callback is never compared because
    ``(time, seq)`` is already unique.
    """

    def __init__(self) -> None:
        self.now: float = 0.0
        self._heap: List[Tuple[float, int, Callable[[], None], str]] = []
        self._seq: int = 0

    def schedule(self, delay: float, cb: Callable[[], None], label: str = "") -> int:
        """Schedule ``cb`` to fire ``delay`` units after :attr:`now`.

        Returns the event's sequence number (its deterministic tie-break key).
        """
        assert delay >= 0.0, "cannot schedule into the past"
        seq = self._seq
        self._seq += 1
        heapq.heappush(self._heap, (self.now + delay, seq, cb, label))
        return seq

    def run(self) -> None:
        """Drain the queue in ``(time, seq)`` order.

        Each callback may schedule further events; the loop stops when the
        queue is empty. :attr:`now` is advanced to each event's firing time
        before its callback runs.
        """
        while self._heap:
            t, _seq, cb, _label = heapq.heappop(self._heap)
            self.now = t
            cb()


class Promise:
    """A one-shot future that drives dependencies deterministically.

    Resolving a promise schedules its callbacks at ``delay=0`` on the owning
    :class:`Sim`, so they run *after* the current event completes (no
    reentrancy) and in registration (FIFO) order. This models the
    store-mediated "done" signal: a puller's ``notify_put`` resolves the
    promise and releases the parked readers that depended on it.
    """

    def __init__(self, sim: Sim, label: str = "") -> None:
        self.sim = sim
        self.label = label
        self.resolved: bool = False
        self._cbs: List[Callable[[], None]] = []

    def add_callback(self, cb: Callable[[], None]) -> None:
        """Register ``cb`` to run when the promise resolves.

        If the promise is already resolved, ``cb`` is scheduled immediately at
        ``delay=0`` (still after the current event, preserving ordering).
        """
        if self.resolved:
            self.sim.schedule(0.0, cb)
        else:
            self._cbs.append(cb)

    def resolve(self) -> None:
        """Resolve the promise (idempotent), firing callbacks via the sim."""
        if self.resolved:
            return
        self.resolved = True
        for cb in self._cbs:
            self.sim.schedule(0.0, cb)
        self._cbs = []
