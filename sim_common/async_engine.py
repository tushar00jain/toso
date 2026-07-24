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
* Timers live on a ``heapq`` keyed by fire time; ``heapq`` is a pure algorithm,
  so an identical sequence of scheduling operations replays identically. (Among
  timers with the *same* fire time the order is heap-deterministic rather than
  strictly FIFO; if a scenario needs strict FIFO among simultaneous timers,
  stagger them by an epsilon. Flagged for the architect.)
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

from sim_common.trace import Trace

_T = TypeVar("_T")


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
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Virtual clock: simulated seconds. `time()` returns this; it only ever
        # advances in `_run_once`, and only when nothing is ready to run.
        self._clock: float = 0.0
        # Shared trace recorder (reused across both engines for one format).
        self.trace: Trace = trace if trace is not None else Trace()
        # Deterministic, per-loop task naming (not asyncio's global counter).
        self._task_seq: int = 0
        # Optional seeded-random ready-queue selection for interleaving sweeps.
        self.random_seed: Optional[int] = random_seed
        self._rng: Optional[random.Random] = (
            random.Random(random_seed) if random_seed is not None else None
        )

    # -- virtual clock ----------------------------------------------------

    def time(self) -> float:
        """Return the current *simulated* time (seconds). Never reads a wall clock."""
        return self._clock

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
) -> Tuple[_T, Trace]:
    """Run ``coro`` to completion on a fresh :class:`AsyncEngine`.

    Returns ``(result, trace)`` and closes the loop. Convenience for tests and
    one-shot scenarios; construct :class:`AsyncEngine` directly if you need the
    loop object (e.g. to schedule external timers before running).
    """
    loop = AsyncEngine(trace=trace, random_seed=random_seed)
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.close()
    return result, loop.trace
