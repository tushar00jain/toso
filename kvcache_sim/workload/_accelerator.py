"""The simulated accelerator: :class:`SimulatedAccelerator`.

The implementation of :class:`kvcache_sim.data._compute.Accelerator` that a *run*
supplies, and the one piece of the compute story that could never lift: it answers
what a forward pass costs from a roofline over the model's flop counts and the
target machine's, and it makes the work take that long by sleeping the virtual
clock. A deployment implements the same protocol by running the model and
measuring.

It lives here for the same reason ``_sim_block_carrier`` does. What a KV block is
stored as differs between a simulated run and a real one, so the run chooses it;
what a forward pass costs and how it is incurred differ the same way, so the run
chooses that too. Neither belongs to the capability, and both used to sit inside it
-- the engines imported the cost model directly and slept.

Occupancy is here too, and it is what makes coupling structural: one of these is
one accelerator, and two engines handed the same object contend on
:attr:`busy_until` while two handed different objects do not.

Deterministic: the only clock is the running loop's, virtual under simulation, and
every duration comes from the run's profiles rather than from anything measured.
"""

from __future__ import annotations

import asyncio

from domain import (
    DEFAULT_MODEL, DEFAULT_PROFILE, decode_step_time, Model, prefill_time,
)

from ..data._compute import Accelerator

__all__ = ["SimulatedAccelerator"]


class SimulatedAccelerator(Accelerator):
    """One host's compute, priced off a roofline and incurred by sleeping.

    Args:
        profile: the target machine's
            :class:`~sim_common.cost_model.MachineProfile` -- what the flops run on.
        model: the served :class:`~domain.llm.Model` -- how many flops the work is.
    """

    def __init__(
        self, *, profile=DEFAULT_PROFILE, model: Model = DEFAULT_MODEL
    ) -> None:
        self.profile = profile
        self.model = model
        #: Sim time this accelerator is occupied until.
        self.busy_until: float = 0.0

    # -- what work costs --------------------------------------------------- #
    def prefill_cost(self, tokens: int) -> float:
        return prefill_time(tokens, self.profile, self.model)

    def step_cost(self, batch_size: int) -> float:
        return decode_step_time(batch_size, self.profile, self.model)

    # -- making it take that long ------------------------------------------ #
    async def prefill(self, tokens: int) -> None:
        """Run the forward pass -- here, wait out what it would have cost.

        Deliberately does *not* queue behind :attr:`busy_until`. The wait a request
        serves before its prefill is the one the control plane predicted and the
        engine slept separately (see
        :meth:`kvcache_sim.data._prefill.PrefillEngine.wait_turn`), so queueing here
        as well would charge it twice. That the two are not the same number is the
        model's weakest joint and is written up there.
        """
        duration = self.prefill_cost(tokens)
        if duration > 0:
            await asyncio.sleep(duration)

    def claim_step(self, batch_size: int) -> float:
        """Book a decode step after whatever already has this accelerator."""
        now = asyncio.get_running_loop().time()
        self.busy_until = max(now, self.busy_until) + self.step_cost(batch_size)
        return self.busy_until

    async def wait_until(self, when: float) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(when - loop.time())

    def reserve(self, until: float) -> None:
        self.busy_until = until
