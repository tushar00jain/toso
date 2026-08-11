"""Deterministic cooperative ``asyncio`` event loop with a virtual clock.

This is the async sibling of :mod:`sim_common.engine`. Where ``engine.Sim`` runs
plain callbacks on a ``(time, seq)`` heap, :class:`AsyncEngine` runs *real*
``async def`` coroutines -- so torchstore's real client/controller code, which is
written against ``asyncio`` (``await``, ``asyncio.sleep``, ``asyncio.gather``),
can execute under simulated time, single-threaded and reproducibly.

Design (chosen for maximum reuse of the real ``asyncio`` machinery)
-------------------------------------------------------------------
We subclass :class:`asyncio.BaseEventLoop` rather than hand-rolling a coroutine
driver. ``BaseEventLoop`` already implements ``call_soon``/``call_at``/
``call_later``/``create_task``/``create_future``/``run_until_complete`` and the
``Task``/``Future`` wakeup plumbing on top of two structures we keep:

* ``self._ready``   -- a ``collections.deque`` of ready callbacks (FIFO).
* ``self._scheduled`` -- a ``heapq`` of ``TimerHandle`` ordered by fire time.

We override only the three things that must change to become a *virtual-time,
I/O-free* loop:

1. :meth:`time` returns ``self._clock`` (simulated seconds) instead of
   ``time.monotonic()``. This is the single source of truth for time; seam and
   adapter code reads it via ``asyncio.get_running_loop().time()`` (no separate
   clock module -- see design doc 3a).
2. :meth:`_run_once` never calls a selector. It drains the ready queue and, when
   the ready queue is empty, performs the classic DES advance: jump the clock to
   the earliest pending timer. So ``await asyncio.sleep(10)`` costs ~0 wall-clock
   -- we move simulated time, we never block.
3. :meth:`_process_events` / :meth:`_write_to_self` are inert (no sockets, no
   self-pipe): all awaited "I/O" in the sim is in-memory futures completed via
   the loop.

Ordering & determinism (mirrors ``engine.py`` conventions)
----------------------------------------------------------
* The ready queue is processed **FIFO by insertion**, exactly like ``engine.py``
  breaks ties by a monotonic ``seq``. All coroutine wakeups (``Task.__step``,
  future done-callbacks) flow through ``call_soon`` -> ``self._ready``, so the
  interleaving of concurrently-runnable coroutines is deterministic.
* Timers live on a ``heapq`` keyed by ``(fire time, seq)`` where ``seq`` is a
  monotonic per-loop insertion counter (see :class:`_SeqTimerHandle`). asyncio's
  own ``TimerHandle`` orders by fire time alone, so timers due at the *same*
  virtual instant would pop in heap-structural order; the ``seq`` tiebreak makes
  them **strictly FIFO** instead -- the same ``(time, seq)`` total order the
  callback engine (:mod:`sim_common.engine`) uses. So an identical sequence of
  scheduling operations replays identically *and* simultaneous timers fire in
  scheduling order (this is what ``realsim_design.md`` §5 already describes).
* Task names are assigned by *this loop* (``task-1``, ``task-2``, ...) instead of
  asyncio's process-global counter, so names -- and therefore the trace -- are
  byte-identical across runs and independent of other loops in the process.
* An optional ``random_seed`` switches the ready queue from FIFO to a
  seeded-random shuffle *per tick*, for class-A interleaving exploration. Same
  seed => same run; default (``None``) => FIFO for reproducible demos.

Tracing
-------
All tracing goes through :class:`sim_common.trace.Trace` (one trace format across
both engines). The loop records scheduling events -- clock advances and task
create/finish -- and exposes :meth:`log` so scenarios can add their own rows at
the current virtual time.
"""

from __future__ import annotations

import asyncio
import heapq
import random
from typing import Any, Coroutine, Optional, Tuple, TypeVar

from sim_common import config
from sim_common.trace import Trace

__all__ = ["AsyncEngine", "run_sim"]

_T = TypeVar("_T")


class _SeqTimerHandle(asyncio.TimerHandle):
    """A :class:`asyncio.TimerHandle` with a FIFO tiebreak among equal fire times.

    asyncio orders timers by ``_when`` alone (its ``TimerHandle.__lt__``), so two
    timers scheduled for the *same* virtual instant pop from the ``heapq`` in
    heap-structural order -- not the order they were scheduled. Carrying a
    monotonic per-loop ``_seq`` and comparing on ``(_when, _seq)`` restores strict
    FIFO among simultaneous timers, matching the ``(time, seq)`` total order the
    callback engine (:mod:`sim_common.engine`) already guarantees. The run is
    deterministic either way; this only removes the counter-intuitive
    "same-time timers reorder" behavior.

    The plain-``TimerHandle`` comparison branches are defensive only: the loop's
    schedule holds nothing but ``_SeqTimerHandle`` instances (every timer is built
    by :meth:`AsyncEngine.call_at`).
    """

    __slots__ = ("_seq",)

    def __init__(self, when, callback, args, loop, context=None, *, seq: int = 0):
        super().__init__(when, callback, args, loop, context)
        self._seq = seq

    def _sort_key(self) -> Tuple[float, int]:
        return (self._when, self._seq)

    def __lt__(self, other: Any) -> Any:
        if isinstance(other, _SeqTimerHandle):
            return self._sort_key() < other._sort_key()
        if isinstance(other, asyncio.TimerHandle):
            return self._when < other._when
        return NotImplemented

    def __le__(self, other: Any) -> Any:
        if isinstance(other, _SeqTimerHandle):
            return self._sort_key() <= other._sort_key()
        if isinstance(other, asyncio.TimerHandle):
            return self._when <= other._when
        return NotImplemented

    def __gt__(self, other: Any) -> Any:
        if isinstance(other, _SeqTimerHandle):
            return self._sort_key() > other._sort_key()
        if isinstance(other, asyncio.TimerHandle):
            return self._when > other._when
        return NotImplemented

    def __ge__(self, other: Any) -> Any:
        if isinstance(other, _SeqTimerHandle):
            return self._sort_key() >= other._sort_key()
        if isinstance(other, asyncio.TimerHandle):
            return self._when >= other._when
        return NotImplemented


class AsyncEngine(asyncio.BaseEventLoop):
    """A single-threaded, deterministic ``asyncio`` loop on a virtual clock.

    Construct one per run (task-name and clock state are per-instance, which is
    what makes two runs byte-identical). Drive it with :meth:`run` /
    :meth:`run_until_complete`, then :meth:`close` it.
    """

    def __init__(
        self,
        *,
        trace: Optional[Trace] = None,
        quiet: Optional[bool] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Virtual clock: simulated seconds. `time()` returns this; it only ever
        # advances in `_run_once`, and only when nothing is ready to run.
        self._clock: float = 0.0
        # Quiet mode ("no tracing at all"): removes every per-event trace side
        # effect -- clock-advance / task create+finish rows and the per-task
        # done-callback -- so a large run pays none of that bookkeeping. It only
        # removes side effects; virtual time, ordering and every metric are
        # byte-identical either way. Ambient default from the config `trace` flag
        # (same pattern as Trace.hash_chain); an explicit `quiet` arg overrides.
        if quiet is None:
            quiet = not config.current().trace
        # Shared trace recorder (reused across both engines for one format). When
        # a trace is supplied its own `enabled` governs; the default one is built
        # disabled in quiet mode so all `record` call sites become no-ops.
        self.trace: Trace = trace if trace is not None else Trace(enabled=not quiet)
        # Deterministic, per-loop task naming (not asyncio's global counter).
        self._task_seq: int = 0
        # Monotonic per-loop timer counter -> the FIFO tiebreak among timers due
        # at the same virtual instant (see `_SeqTimerHandle` / `call_at`).
        self._timer_seq: int = 0
        # Optional seeded-random ready-queue selection for interleaving sweeps.
        self.random_seed: Optional[int] = random_seed
        self._rng: Optional[random.Random] = (
            random.Random(random_seed) if random_seed is not None else None
        )

    # -- virtual clock ----------------------------------------------------

    def time(self) -> float:
        """Return the current *simulated* time (seconds). Never reads a wall clock."""
        return self._clock

    # -- timer scheduling (with a FIFO tiebreak among simultaneous timers) -

    def call_at(  # type: ignore[override]
        self,
        when: float,
        callback: Any,
        *args: Any,
        context: Any = None,
    ) -> "asyncio.TimerHandle":
        """Schedule ``callback`` at virtual time ``when`` with a FIFO tiebreak.

        Mirrors :meth:`asyncio.BaseEventLoop.call_at` but builds a
        :class:`_SeqTimerHandle` carrying a monotonic ``seq``, so timers due at the
        same virtual instant fire in scheduling order rather than heap-structural
        order. ``call_later`` and ``asyncio.sleep`` both route through here, so
        every virtual-clock sleep inherits the tiebreak.
        """
        self._check_closed()
        if self._debug:
            self._check_thread()
            self._check_callback(callback, "call_at")
        timer = _SeqTimerHandle(when, callback, args, self, context, seq=self._timer_seq)
        self._timer_seq += 1
        if timer._source_traceback:
            del timer._source_traceback[-1]
        heapq.heappush(self._scheduled, timer)
        timer._scheduled = True
        return timer

    # -- tracing ----------------------------------------------------------

    def log(self, kind: str, msg: str) -> None:
        """Record a trace row at the current virtual time (for scenario code)."""
        self.trace.record(self._clock, kind, msg)

    # -- task lifecycle (deterministic names + create/finish trace) -------

    def create_task(  # type: ignore[override]
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: Optional[str] = None,
        context: Any = None,
    ) -> "asyncio.Task[_T]":
        """Create a Task with a deterministic, per-loop name and trace it.

        ``asyncio.gather`` / ``ensure_future`` route coroutine arguments through
        here, so gather fan-outs get stable ``task-N`` names and a defined
        completion order under FIFO scheduling.
        """
        if name is None:
            self._task_seq += 1
            name = f"task-{self._task_seq}"
        task = super().create_task(coro, name=name, context=context)
        self.trace.record(self._clock, "task", f"create {name}")
        # Skip the finish-row callback entirely when tracing is off: it exists
        # only to emit the "{status} {name}" row and schedules nothing, so its
        # absence cannot change clock, ordering, naming or any metric.
        if self.trace.enabled:
            task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: "asyncio.Task[Any]") -> None:
        name = task.get_name()
        if task.cancelled():
            status = "cancel"
        elif task.exception() is not None:
            status = "error"
        else:
            status = "finish"
        self.trace.record(self._clock, "task", f"{status} {name}")

    # -- inert I/O (no selectors, no sockets, no self-pipe) ---------------

    def _process_events(self, event_list: Any) -> None:  # pragma: no cover
        """No real I/O: there are never any selector events to process."""

    def _write_to_self(self) -> None:  # pragma: no cover
        """No self-pipe: the loop is single-threaded, nothing to wake."""

    # -- the DES core: drain ready queue, else advance the virtual clock --

    def _run_once(self) -> None:
        """One tick: fire due timers + drain the ready queue; advance if idle.

        Overrides ``BaseEventLoop._run_once`` to remove the selector poll and
        replace the ``select(timeout)`` sleep with a virtual-clock jump.
        """
        # 1. Drop cancelled timers at the head of the schedule.
        while self._scheduled and self._scheduled[0]._cancelled:
            self._timer_cancelled_count -= 1
            handle = heapq.heappop(self._scheduled)
            handle._scheduled = False

        # 2. Virtual-clock advance (classic DES). If nothing is ready to run,
        #    jump simulated time to the earliest pending timer. This is why
        #    `asyncio.sleep(10)` returns immediately in wall-clock: we never
        #    block on a selector, we only move `self._clock` forward.
        if not self._ready and not self._stopping:
            if self._scheduled:
                when = self._scheduled[0]._when
                if when > self._clock:
                    delta = when - self._clock
                    self._clock = when
                    self.trace.record(self._clock, "clock", f"advance +{delta:g}")
            else:
                # Nothing ready and no timers, yet the driving future is not
                # done -> the coroutines are deadlocked on a future that will
                # never be resolved. Fail loudly instead of spinning forever.
                raise RuntimeError(
                    "async_engine deadlock: no ready callbacks and no pending "
                    "timers, but run_until_complete has not finished"
                )

        # 3. Move every timer now due (fire time <= clock) into the ready queue.
        now = self._clock
        while self._scheduled and self._scheduled[0]._when <= now:
            handle = heapq.heappop(self._scheduled)
            handle._scheduled = False
            self._ready.append(handle)

        # 4. Drain exactly the callbacks ready at the start of this tick.
        #    Snapshotting `ntodo` first means callbacks that `call_soon`
        #    themselves run on the *next* tick (matches stdlib semantics and
        #    engine.py's "schedule(0)" -> after the current event).
        ntodo = len(self._ready)
        batch = [self._ready.popleft() for _ in range(ntodo)]
        order = list(range(ntodo))
        if self._rng is not None:
            self._rng.shuffle(order)
        for i in order:
            handle = batch[i]
            if handle._cancelled:
                continue
            handle._run()

    # -- convenience ------------------------------------------------------

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Alias for :meth:`run_until_complete` (drains timers + ready queue)."""
        return self.run_until_complete(coro)


def run_sim(
    coro: Coroutine[Any, Any, _T],
    *,
    random_seed: Optional[int] = None,
    trace: Optional[Trace] = None,
    quiet: Optional[bool] = None,
) -> Tuple[_T, Trace]:
    """Run ``coro`` to completion on a fresh :class:`AsyncEngine`.

    Returns ``(result, trace)`` and closes the loop. Convenience for tests and
    one-shot scenarios; construct :class:`AsyncEngine` directly if you need the
    loop object (e.g. to schedule external timers before running).

    ``quiet`` opts out of all per-event tracing (see :class:`AsyncEngine`); the
    returned trace is then empty. ``None`` (the default) defers to the config's
    ``trace`` flag.
    """
    loop = AsyncEngine(trace=trace, quiet=quiet, random_seed=random_seed)
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.close()
    return result, loop.trace
