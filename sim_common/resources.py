"""Shared, per-run resource layer: model network/storage contention.

The transport seam (:mod:`realsim.seams.transport`) charges every put/get as a
set of independent virtual-clock sleeps -- one per resource (network, storage,
RAM). With no contention model, two concurrent transfers over the *same* fabric
each assume the full bandwidth: they overlap for free. That is the ``"none"``
default and is exactly the historical behavior.

This module adds an opt-in contention model. A single :class:`ResourceRegistry`
is created once per run and injected into every transport, so all transports in
a run share the same *resources* (a link's egress, a volume's read/write
channel). When several transfers compete for one resource they share its
bandwidth. Three modes are selectable (via the ``contention`` config flag / the
``TOSO_CONTENTION`` env var):

* ``"none"`` -- no contention. The transport never calls into the registry; each
  charge is an independent sleep. Byte-identical to the historical trace.
* ``"serialize"`` -- a resource serves **one** transfer at a time; the rest queue
  (FIFO by a monotonic transfer sequence) and run at full bandwidth afterwards.
  Total time for N contending transfers is ~the sum of their solo costs. Mirrors
  the per-instance compute ``busy_until`` pattern in
  :mod:`kvcache_sim.policy.scheduler`.
* ``"progressive"`` -- max-min fair sharing. Every in-flight transfer on a
  resource gets an equal share of its bandwidth; when a transfer *enters* or
  *leaves* the resource, every other in-flight transfer is re-rated (remaining
  bytes advanced by ``elapsed x old_rate``, an equal share recomputed, and its
  completion timer cancelled + rescheduled). A transfer that starts alone slows
  down when a co-tenant appears and speeds back up when it leaves.

Cost model (how a charge maps onto a resource)
----------------------------------------------
Each transfer carries the analytic decomposition of its cost (see
:func:`sim_common.cost_model.network_rate` / ``storage_rate``): a fixed
``latency`` plus ``nbytes / capacity`` where ``capacity`` is the resource's
bandwidth (bytes/time). ``latency + nbytes/capacity`` is exactly the total the
old single-sleep model charged, so a **lone** transfer under any mode costs the
same analytic ``dt`` as before. Only the *bandwidth* term is shared; the fixed
``latency`` is modeled as a leading, non-contended setup delay (a transfer does
not occupy the resource -- nor is it slowed -- until its latency has elapsed).

Resource identity (keyed by the caller; see the transport for the keys used)
----------------------------------------------------------------------------
* **Network** -- keyed by the *source* endpoint's egress (``("net", src_id)``).
  This captures the motivating hot-source case: ``m`` readers concurrently
  pulling from ONE source share that source's egress bandwidth. Simplifications:
  destination-ingress and switch/core contention are not modeled (future work),
  and a channel's capacity is taken from its first transfer -- i.e. all transfers
  leaving one source are assumed to traverse the same tier (homogeneous egress).
* **Storage** -- keyed per volume + direction (``("store", volume_id, kind)``),
  so concurrent reads of one volume share ``storage_read_bw`` and concurrent
  writes share ``storage_write_bw``.

RAM staging (``mem_copy``) is deliberately left as a plain sleep in the transport
for now; it uses the very same mechanism and can be added later by giving it a
resource key.

Determinism
-----------
Every transfer is stamped with a monotonic ``seq`` at registration. Re-rating
iterates a channel's in-flight transfers in ``seq`` order, and completion timers
are (re)scheduled through the loop's ``call_at`` -- which orders simultaneous
timers by insertion via :class:`sim_common.async_engine._SeqTimerHandle`. So a
re-rate that touches several transfers at the same virtual instant, and several
transfers completing at the same instant, both resolve in a fixed order. Same
input => same trace, under each mode.

Bounce discipline
------------------
A charge suspends exactly once: :meth:`ResourceRegistry.transfer` awaits a single
future that resolves at the transfer's true completion. All the churn of
re-rating happens on loop *timers* (leading-latency handles and completion
handles), never by resuming and re-suspending the charging coroutine. This keeps
the number of coroutine suspends per charge identical to the old one-sleep model.

``asyncio`` note: this module drives futures/timers on the running loop, which is
the deterministic :class:`sim_common.async_engine.AsyncEngine` on the sim path
(virtual clock, single-threaded). It imports only the standard library.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, Optional

from sim_common import config

__all__ = ["CONTENTION_MODES", "ResourceRegistry"]

# The selectable contention models (see the module docstring). ``"none"`` is the
# default and is byte-identical to the historical single-sleep behavior.
CONTENTION_MODES = ("none", "serialize", "progressive")


@dataclass
class _Transfer:
    """One in-flight transfer competing for a resource.

    ``seq`` is the registry-wide monotonic stamp used for deterministic ordering.
    ``remaining`` tracks the bytes not yet moved (only meaningful once the leading
    ``latency`` has elapsed and the transfer is in a channel's byte-phase).
    ``handle`` is the transfer's currently-scheduled loop timer (a leading-latency
    handle while warming up, then a completion handle); it is cancelled and
    replaced on every re-rate.
    """

    seq: int
    remaining: float           # bytes still to move
    latency: float             # fixed, non-contended leading delay
    capacity: float            # this transfer's solo bandwidth (bytes/time)
    future: "asyncio.Future"
    handle: Optional["asyncio.TimerHandle"] = None


class _SerializeChannel:
    """A resource that serves one transfer at a time (others queue FIFO).

    Each transfer runs at full bandwidth for its whole solo cost; contenders wait
    in the arrival (``seq``) order and start only when the current one finishes.
    So N contending transfers cost ~the sum of their solo ``dt``.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: Deque[_Transfer] = deque()
        self._busy = False

    def register(self, t: _Transfer) -> None:
        self._queue.append(t)
        if not self._busy:
            self._start_next()

    def _start_next(self) -> None:
        if not self._queue:
            self._busy = False
            return
        self._busy = True
        t = self._queue.popleft()
        # Solo cost: full bandwidth, so exactly the analytic dt. Grouped as one
        # value so a lone/serial transfer's completion equals latency+nbytes/bw.
        dt = t.latency + t.remaining / t.capacity
        t.handle = self._loop.call_at(self._loop.time() + dt, self._finish, t)

    def _finish(self, t: _Transfer) -> None:
        if not t.future.done():
            t.future.set_result(None)
        self._start_next()


class _ProgressiveChannel:
    """A resource whose bandwidth is shared max-min fairly among its tenants.

    Latency is a leading, non-contended delay: a freshly-registered transfer
    "warms up" for its ``latency`` before it joins the byte-phase (so it neither
    slows others nor is slowed during setup). Once in the byte-phase, all active
    transfers share ``capacity`` equally; entering or leaving the byte-phase
    re-rates every active transfer.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, capacity: float) -> None:
        self._loop = loop
        # Channel capacity is fixed from the first transfer (homogeneous-egress
        # assumption; see the module docstring). Individual transfers may carry a
        # different solo ``capacity``, but the shared resource has one bandwidth.
        self._capacity = capacity
        self._active: Dict[int, _Transfer] = {}   # in byte-phase, keyed by seq
        self._rate = 0.0                           # current per-transfer share
        self._last_update = loop.time()            # when accounting last advanced

    def register(self, t: _Transfer) -> None:
        # Leading latency: schedule the transfer to enter the byte-phase after its
        # fixed setup delay. It does not occupy the resource until then, so no
        # re-rate happens now -- only when it actually starts consuming bandwidth.
        t.handle = self._loop.call_at(
            self._loop.time() + t.latency, self._enter, t
        )

    def _advance(self, now: float) -> None:
        """Debit every active transfer for the bytes moved since ``last_update``."""
        elapsed = now - self._last_update
        if elapsed > 0 and self._active:
            moved = elapsed * self._rate
            for t in self._active.values():
                t.remaining -= moved
                if t.remaining < 0.0:
                    t.remaining = 0.0
        self._last_update = now

    def _recompute(self, now: float) -> None:
        """Recompute the equal share and (re)schedule every completion timer.

        Iterates in ``seq`` order so simultaneous reschedules are inserted into
        the loop's timer heap deterministically.
        """
        k = len(self._active)
        self._rate = self._capacity / k if k else 0.0
        for seq in sorted(self._active):
            t = self._active[seq]
            if t.handle is not None:
                t.handle.cancel()
            dt = t.remaining / self._rate
            t.handle = self._loop.call_at(now + dt, self._complete, seq)

    def _enter(self, t: _Transfer) -> None:
        """Warm-up done: the transfer joins the byte-phase and re-rates the rest."""
        now = self._loop.time()
        self._advance(now)
        self._active[t.seq] = t
        self._recompute(now)

    def _complete(self, seq: int) -> None:
        """A transfer's bytes are exhausted: resolve it and re-rate the rest."""
        now = self._loop.time()
        self._advance(now)
        t = self._active.pop(seq, None)
        if t is None:
            return
        if not t.future.done():
            t.future.set_result(None)
        # The freed capacity is redistributed to whoever is left (they speed up).
        self._recompute(now)


class ResourceRegistry:
    """Per-run shared resource layer (one instance injected into every transport).

    Args:
        mode: one of :data:`CONTENTION_MODES`. ``"none"`` means the transport
            never calls :meth:`transfer` (it sleeps independently), so a registry
            in ``"none"`` mode is inert.
    """

    def __init__(self, mode: str = "none") -> None:
        if mode not in CONTENTION_MODES:
            raise ValueError(
                f"contention mode must be one of {list(CONTENTION_MODES)}, got {mode!r}"
            )
        self.mode = mode
        # Resource key -> its channel. Keys are chosen by the caller (the
        # transport): network egress and per-volume storage read/write channels.
        self._channels: Dict[Hashable, object] = {}
        # Registry-wide monotonic transfer stamp: the deterministic tie-break for
        # re-rating order and simultaneous starts/finishes.
        self._seq = 0

    @classmethod
    def from_config(cls, contention: Optional[str] = None) -> "ResourceRegistry":
        """Build a registry, resolving the mode like the other ambient flags.

        ``contention`` (an explicit override, e.g. from a test) wins; otherwise
        the process-wide :data:`sim_common.config.SimConfig.contention` is used
        (which itself came from ``TOSO_CONTENTION`` / the CLI flag / the default).
        """
        mode = contention if contention is not None else config.current().contention
        return cls(mode)

    async def transfer(
        self,
        key: Hashable,
        *,
        capacity: float,
        latency: float,
        nbytes: int,
    ) -> float:
        """Charge one transfer on resource ``key``; return the loop time at completion.

        Registers the transfer on the resource's channel and awaits a single
        future resolved when the transfer truly finishes (one suspend per charge).
        Only called under ``"serialize"`` / ``"progressive"`` -- ``"none"`` never
        reaches here (the transport sleeps directly).
        """
        loop = asyncio.get_running_loop()
        self._seq += 1
        t = _Transfer(
            seq=self._seq,
            remaining=float(nbytes),
            latency=latency,
            capacity=capacity,
            future=loop.create_future(),
        )
        channel = self._channels.get(key)
        if channel is None:
            if self.mode == "serialize":
                channel = _SerializeChannel(loop)
            else:  # "progressive"
                channel = _ProgressiveChannel(loop, capacity)
            self._channels[key] = channel
        channel.register(t)  # type: ignore[attr-defined]
        await t.future
        return loop.time()
