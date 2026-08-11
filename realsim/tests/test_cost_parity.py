"""The charge and the prediction of a ``get`` must stay one formula.

:func:`sim_common.cost_model._get_time` is the single definition of what serving a
get costs (``storage read + host-RAM staging + fabric``). Two very different
consumers depend on it:

* the transport seam **charges** it, as three virtual-clock sleeps (or one
  combined sleep under ``collapse_charges``); and
* a scheduler **predicts** it, to decide whether pulling a remote copy beats
  recomputing (``kvcache_sim.control.scheduler``).

If those two ever drift apart nothing fails: the sim keeps running and keeps
producing plausible numbers, while every routing decision is made against a cost
the run does not actually pay. This test pins them together by measuring the
*clock advance* of a real get -- exact under the virtual clock -- against
``get_time``. It is the regression guard for the transport's charge composition:
add or drop a component there and this fails.
"""

from __future__ import annotations

import asyncio
import math

import torch

from realsim.simulation import Simulation
from realsim.seams.transport import Endpoint
from sim_common import config
from sim_common.async_engine import run_sim
from sim_common.cost_model import DEFAULT_PROFILE, _get_time, network_time

KEY = "W"
N = 4096  # float32 elements -> a payload big enough that every term is non-zero


def _topology() -> dict[str, Endpoint]:
    """A serving volume and a cross-node client (so the fabric term is non-zero)."""
    return {
        "srv": Endpoint(id="volsrv", host="hostA", node="node0"),
        "cli": Endpoint(id="volcli", host="hostB", node="node1"),
    }


def _measure_get_advance(collapse: bool) -> tuple[float, int]:
    """Return ``(clock advance of one remote get, nbytes served)``."""
    topo = _topology()
    served: list[int] = []

    with config.overrides(collapse_charges=collapse):
        sim = Simulation(topo, profile=DEFAULT_PROFILE)
        mesh = sim.mesh
        mesh.on_transfer = lambda kind, s, d, nbytes, cost: (
            served.append(nbytes) if kind == "get" else None
        )

        async def scenario() -> float:
            loop = asyncio.get_running_loop()
            with mesh.installed():
                # Seed the payload on the serving volume (a local write).
                mesh.bind_source("srv")
                await mesh.client("srv").put(
                    KEY, torch.empty(N, dtype=torch.float32, device="meta")
                )
                # Time exactly the cross-node get.
                mesh.bind_source("cli")
                before = loop.time()
                await mesh.client("cli").get(KEY)
                return loop.time() - before

        try:
            advance = sim.loop.run_until_complete(scenario())
        finally:
            sim.loop.close()

    assert served, "no get transfer was reported"
    return advance, served[0]


def test_transport_get_charge_equals_get_time():
    """A real get advances the clock by exactly ``get_time`` for its bytes."""
    advance, nbytes = _measure_get_advance(collapse=False)
    # Through the stack's own estimator -- the value a scheduler would be handed.
    expected = Simulation(
        _topology(), profile=DEFAULT_PROFILE
    ).transfer_cost.get_time("srv", "cli", nbytes)
    assert nbytes == N * 4  # float32
    # Compared with a tolerance, not bit-exactly: the advance is a difference of
    # two absolute clock readings, so its last bit depends on what the clock had
    # already accumulated, while get_time sums the three terms from zero.
    assert math.isclose(advance, expected, rel_tol=1e-12), (
        f"transport charged {advance!r} for a {nbytes}B get but get_time says "
        f"{expected!r} -- the charge and the prediction have drifted apart"
    )


def test_collapsed_get_charges_the_same_total():
    """``collapse_charges`` merges the sleeps but must not change the total.

    The flag is documented as advancing the clock by the exact same amount, so it
    must agree with ``get_time`` too -- otherwise turning it on would silently
    reprice every fetch.
    """
    per_component, nbytes_a = _measure_get_advance(collapse=False)
    collapsed, nbytes_b = _measure_get_advance(collapse=True)
    assert nbytes_a == nbytes_b
    assert math.isclose(collapsed, per_component, rel_tol=1e-12)


def test_a_colocated_get_is_not_free():
    """A same-volume get still costs a storage read + RAM staging.

    Only the *fabric* term of a get is zero when server and client coincide --
    reading your own pool is not free. This is the case a well-meaning
    "optimization" would short-circuit to ``0.0`` inside ``_get_time``, which would
    silently break its agreement with the transport; the cross-node tests above
    would not catch that, so pin it explicitly.
    """
    topo = {"solo": Endpoint(id="volsolo", host="hostA", node="node0")}
    ep = topo["solo"]
    sim = Simulation(topo, profile=DEFAULT_PROFILE)
    mesh = sim.mesh

    async def scenario() -> tuple[float, int]:
        loop = asyncio.get_running_loop()
        with mesh.installed():
            mesh.bind_source("solo")
            await mesh.client("solo").put(
                KEY, torch.empty(N, dtype=torch.float32, device="meta")
            )
            before = loop.time()
            await mesh.client("solo").get(KEY)
            return loop.time() - before, N * 4

    try:
        advance, nbytes = sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()
    # Priced through the stack's own estimator, not a parallel call to _get_time:
    # the point is that what a scheduler is handed matches what it is charged.
    expected = sim.transfer_cost.get_time("solo", "solo", nbytes)
    assert expected > 0.0, "a co-located get must still cost storage + RAM"
    assert network_time(ep, ep, nbytes, DEFAULT_PROFILE) == 0.0  # only fabric is free
    assert math.isclose(advance, expected, rel_tol=1e-12), (
        f"co-located get advanced the clock {advance!r}, get_time says {expected!r}"
    )
