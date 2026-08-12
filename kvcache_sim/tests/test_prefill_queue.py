"""The prefill queue: what it serialises, and what it contradicts.

Two halves, because the change has two.

The **mechanism** half drives one
:class:`~kvcache_sim.workload._accelerator.SimulatedAccelerator` against a bare
clock: one forward pass at a time, an explicitly sorted service order, and one
occupancy shared with decode. No store, no directory and no topology are needed
for any of it -- a queue is a property of one device.

The **consequence** half runs the scenarios the demo runs. A serving host used to
sleep the wait the control plane predicted (``plan.queue_wait``) and then sleep its
forward pass, which made the prediction unfalsifiable: the measured wait *was* the
predicted wait, always, on every workload. Now the pass is submitted to the host's
accelerator and runs when the device is free, so the two are separately recorded
and can disagree -- and the tests below pin down both that they do and the
structural reason why, which is that control prices a remote prefix pull against
the prefill instance's occupancy and a real device is idle during it.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest kvcache_sim/tests/test_prefill_queue.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from sim_common.async_engine import AsyncEngine

from kvcache_sim.workload._accelerator import (
    BLOCK_TOKENS, SimulatedAccelerator, token_tensor,
)
from kvcache_sim.tests._run import run_shared_prefix


TOKENS = 4 * BLOCK_TOKENS
#: The prompt every submission below hands in. A forward pass takes the tokens it
#: has to compute, not a count of them, so a test that submits one has to have a
#: prompt -- and a ``device="meta"`` one costs nothing to make and says exactly as
#: much as the integer did, which is how long the pass is.
PROMPT = token_tensor(TOKENS)


def _drive(coro):
    """Run one coroutine on a fresh virtual clock; answer with what it answered."""
    loop = AsyncEngine()
    try:
        return loop.run_until_complete(coro())
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# The mechanism: one device, one pass at a time.
# --------------------------------------------------------------------------- #
def test_the_accelerator_runs_one_pass_at_a_time():
    """Three passes submitted at once take three passes' worth of clock.

    The whole claim, in one assertion. Before this, three concurrent prefills on
    one host each slept their own cost and all three finished at ``cost``, with
    the queue they should have formed represented only by a number the scheduler
    had told each of them to sleep beforehand.
    """
    gpu = SimulatedAccelerator()
    cost = gpu.prefill_cost(TOKENS)
    finished = {}

    async def submit(tag: str) -> None:
        await gpu.prefill(PROMPT, tag=tag)
        finished[tag] = asyncio.get_running_loop().time()

    async def drive() -> float:
        await asyncio.gather(*(submit(f"r{i}") for i in range(3)))
        return asyncio.get_running_loop().time()

    total = _drive(drive)
    assert total == pytest.approx(3 * cost)
    assert finished == {
        "r0": pytest.approx(cost),
        "r1": pytest.approx(2 * cost),
        "r2": pytest.approx(3 * cost),
    }


def test_the_wait_is_emergent_and_nobody_was_told_it():
    """A pass submitted behind two others waits for them, and only for them.

    Nothing in this test mentions a queue wait; the second and third submissions
    are handed the same arguments as the first. What each of them experienced is a
    consequence of the device being busy, which is the property the whole change
    exists to buy.
    """
    gpu = SimulatedAccelerator()
    cost = gpu.prefill_cost(TOKENS)
    waited = {}

    async def submit(tag: str) -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await gpu.prefill(PROMPT, tag=tag)
        waited[tag] = loop.time() - start - cost

    async def drive() -> None:
        await asyncio.gather(*(submit(f"r{i}") for i in range(3)))

    _drive(drive)
    assert waited["r0"] == pytest.approx(0.0)
    assert waited["r1"] == pytest.approx(cost)
    assert waited["r2"] == pytest.approx(2 * cost)


def test_the_service_order_is_sorted_not_the_order_the_loop_resumed_them():
    """Same-instant submissions run in ``(submitted, tag)`` order.

    ``r9`` takes the idle device, so it runs first whatever its name. The other
    three are handed in at that same virtual instant in the order ``r3, r1, r2``
    and are served ``r1, r2, r3``: the tie on the submission time is broken by the
    request id, which the workload fixes, rather than by which coroutine the event
    loop happened to resume first, which an unrelated ``await`` upstream could
    reorder without anybody noticing until a run stopped reproducing.
    """
    gpu = SimulatedAccelerator()
    order = []

    async def submit(tag: str) -> None:
        await gpu.prefill(PROMPT, tag=tag)
        order.append(tag)

    async def drive() -> None:
        await asyncio.gather(
            submit("r9"), submit("r3"), submit("r1"), submit("r2")
        )

    _drive(drive)
    assert order == ["r9", "r1", "r2", "r3"]


def test_a_pass_with_nothing_to_compute_does_not_take_a_turn():
    """A fully-cached prompt occupies no device, so it does not queue for one."""
    gpu = SimulatedAccelerator()
    cost = gpu.prefill_cost(TOKENS)
    finished = {}

    async def submit(tag: str, prompt) -> None:
        await gpu.prefill(prompt, tag=tag)
        finished[tag] = asyncio.get_running_loop().time()

    async def drive() -> None:
        # ...and "nothing to compute" is an empty prompt, which is what a prompt
        # entirely covered by cached blocks leaves for the pass.
        await asyncio.gather(submit("r0", PROMPT), submit("r1", token_tensor(0)))

    _drive(drive)
    assert finished["r1"] == pytest.approx(0.0)
    assert finished["r0"] == pytest.approx(cost)


def test_a_prefill_queues_behind_a_decode_step_on_the_same_accelerator():
    """One occupancy, two kinds of work -- which is what coupling now costs.

    The decode engine books its step through ``claim_step`` and the queue books
    its pass on the same field, so a prefill submitted while a step holds the
    device starts when the step ends. Nothing has to be configured, and in
    particular nothing has to be *reserved*: the reservation the serving host used
    to push here on control's behalf existed only because a prefill was invisible
    to this object.
    """
    gpu = SimulatedAccelerator()
    step = gpu.step_cost(4)
    cost = gpu.prefill_cost(TOKENS)

    async def drive() -> float:
        gpu.claim_step(4)
        await gpu.prefill(PROMPT, tag="r0")
        return asyncio.get_running_loop().time()

    assert _drive(drive) == pytest.approx(step + cost)


def test_a_decode_step_queues_behind_a_running_prefill():
    """...and the other direction: the step is booked after the pass ends."""
    gpu = SimulatedAccelerator()
    cost = gpu.prefill_cost(TOKENS)
    step = gpu.step_cost(1)

    async def drive() -> float:
        running = asyncio.get_running_loop().create_task(
            gpu.prefill(PROMPT, tag="r0")
        )
        await asyncio.sleep(0)  # let the pass claim the device
        booked = gpu.claim_step(1)
        await running
        return booked

    assert _drive(drive) == pytest.approx(cost + step)


# --------------------------------------------------------------------------- #
# The consequence: a prediction that can now be wrong, and a run that repeats.
# --------------------------------------------------------------------------- #
def _waits(result):
    """``(predicted, actual)`` prefill queue wait per accepted request."""
    return [
        (r.predicted_queue_wait, r.queue_wait)
        for r in result.ledger.results
        if r.accepted
    ]


def test_the_run_is_deterministic():
    """Same run twice: same trace, same metrics, same measured waits.

    The property the sorted service order buys, and the reason it is sorted. A
    queue served in whatever order the event loop resumed its waiters would pass
    this today and stop passing the day an ``await`` moved upstream of one caller.
    """
    a = run_shared_prefix(seed=1)[0]
    b = run_shared_prefix(seed=1)[0]
    assert a.trace.render() == b.trace.render()
    assert a.ledger.hit_rate == b.ledger.hit_rate
    assert a.ledger.mean_ttft == b.ledger.mean_ttft
    assert a.ledger.mean("queue_wait") == b.ledger.mean("queue_wait")


def test_the_measured_wait_is_no_longer_the_predicted_one():
    """Cache-aware routing mispredicts its own queue, and the run says so.

    The systematic part has one cause: control prices a candidate as
    ``queue -> transfer -> prefill`` and reserves the instance until the end of all
    three, so a remote prefix pull is charged to the *prefill device's* occupancy,
    where a real device is idle while the fabric works. The requests that pull are
    therefore the requests whose predicted wait is too long.
    """
    waits = _waits(run_shared_prefix(seed=1)[0])
    diffs = [actual - predicted for predicted, actual in waits]
    assert any(abs(d) > 1e-3 for d in diffs), "the prediction was never contradicted"
    assert min(diffs) < -1e-3, "no request waited less than it was told it would"


def test_the_baseline_never_pulls_so_its_prediction_stays_exact():
    """The control: divergence tracks the mispricing, not the queue's existence.

    ``LoadBalanceScheduler`` reuses only an instance's own cache, so it never
    prices a transfer into an instance's occupancy, and its model of its queue is
    the queue -- to the last float. If this test ever failed alongside the one
    above, the divergence being measured would be the queue mechanism itself
    rather than what the queue reveals about the forecast.
    """
    for predicted, actual in _waits(run_shared_prefix(seed=1)[1]):
        assert actual == pytest.approx(predicted, abs=1e-9)
