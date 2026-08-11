"""Storage-volume byte-capacity enforcement (opt-in via the ``MachineProfile``).

The transport seam models transfer *time*; these tests exercise the *capacity*
model that lives in the volume seam (:class:`realsim.seams.volume_service.VolumeService`):
each volume tracks its aggregate resident working set (bytes added on put,
subtracted on delete) around the real ``InMemoryStore`` lifecycle, and rejects a
put that would over-commit a *finite* ``MachineProfile.storage_capacity_bytes``.

Gates:

* default ``math.inf`` capacity -- unbounded, historical behavior; resident /
  peak tracking still works and nothing is raised;
* within a finite capacity -- a burst whose resident set fits succeeds and
  ``peak_resident_bytes`` reflects the real peak;
* over a finite capacity -- the over-committing put raises
  :class:`~realsim.seams.volume_service.StorageCapacityExceeded` carrying the
  volume id / capacity / resident / attempted bytes;
* a real delete frees space -- resident drops and a put that would otherwise
  exceed capacity then fits (peak stays a run-lifetime high-water mark);
* determinism -- the same run reproduces the same resident / peak values;
* asking before refusing -- a full volume asks the directory
  (``proposed.deployment.Controller.evict_for``, answered by the installed
  ``Policy``) which keys to drop, over the same handle every other caller reaches
  it through, and refuses only if nobody answers or too little is freed.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest realsim/tests/test_storage_capacity.py -q
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
import torch

from realsim.adapters.real_client import RealClientAdapter
from realsim.adapters.real_controller import RealControllerAdapter
from putget_sim.workload.put_get import DEFAULT_N, PutGetBurst
from realsim.run import Run
from proposed import Policy, Selection
from realsim.seams.transport import Endpoint
from realsim.seams.volume_handle import LocalVolumeHandle
from realsim.seams.volume_service import StorageCapacityExceeded, VolumeService
from sim_common.cost_model import DEFAULT_PROFILE

# One default-N float32 payload (DEFAULT_N elements x 4 bytes) -- the burst's W.
PAYLOAD_BYTES = DEFAULT_N * 4


def _run_build(profile=None, *, num_readers: int = 3):
    """Run one burst; return ``(results, trace, ctx)``.

    Keeps the fixture and the run result so the caller can inspect the volume
    handles for resident/peak bytes. Any :class:`StorageCapacityExceeded` raised
    during the run propagates out.
    """
    workload = PutGetBurst(num_readers, profile=profile)
    result = Run("unrouted", workload, profile=workload.profile).execute()
    ctx = {
        "volumes": result.sim.mesh.volumes,
        "origin_id": workload.origin_id,
        "expected": workload.expected,
    }
    return result.results, result.trace, ctx


# --------------------------------------------------------------------------
# Default (inf) capacity: unbounded, historical behavior; tracking still works.
# --------------------------------------------------------------------------


def test_default_capacity_is_unbounded_and_tracks_resident():
    # No profile -> DEFAULT_PROFILE, whose storage_capacity_bytes is inf.
    assert DEFAULT_PROFILE.storage_capacity_bytes == float("inf")

    results, _trace, ctx = _run_build()  # must not raise
    volumes = ctx["volumes"]
    origin = ctx["origin_id"]  # endpoint id of the volume holding W ("volp")

    # The origin volume "p" holds exactly W after the seed put; its resident and
    # peak both equal the payload. Every reader volume only served gets (no put),
    # so it holds nothing.
    origin_vol = volumes["p"]
    assert origin_vol.volume_id == origin
    assert origin_vol.service.resident_bytes == PAYLOAD_BYTES
    assert origin_vol.service.peak_resident_bytes == PAYLOAD_BYTES
    assert origin_vol.service.capacity_bytes == float("inf")
    for vid, handle in volumes.items():
        if vid != "p":
            assert handle.service.resident_bytes == 0
            assert handle.service.peak_resident_bytes == 0


# --------------------------------------------------------------------------
# Within a finite capacity: the burst fits and peak reflects the real peak.
# --------------------------------------------------------------------------


def test_within_capacity_succeeds_and_peak_reflects_real_peak():
    # Capacity generously above one payload -> the seed put fits.
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 4)
    _results, _trace, ctx = _run_build(profile=profile)  # must not raise

    origin_vol = ctx["volumes"]["p"]
    assert origin_vol.service.capacity_bytes == PAYLOAD_BYTES * 4
    assert origin_vol.service.resident_bytes == PAYLOAD_BYTES
    assert origin_vol.service.peak_resident_bytes == PAYLOAD_BYTES


# --------------------------------------------------------------------------
# Over a finite capacity: the over-committing put raises with the details.
# --------------------------------------------------------------------------


def test_over_capacity_raises_with_details():
    # Capacity strictly below one payload -> seeding W over-commits the origin.
    cap = PAYLOAD_BYTES // 2
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=cap)

    with pytest.raises(StorageCapacityExceeded) as excinfo:
        _run_build(profile=profile)

    exc = excinfo.value
    assert exc.volume_id == "volp"  # the origin volume's endpoint id
    assert exc.capacity == cap
    assert exc.resident == 0  # nothing resident before the failed seed put
    assert exc.attempted == PAYLOAD_BYTES
    # The message names the volume, its capacity, and the attempted bytes.
    msg = str(exc)
    assert "volp" in msg
    assert str(cap) in msg
    assert str(PAYLOAD_BYTES) in msg


# --------------------------------------------------------------------------
# A real delete frees space so a subsequent over-committing put then fits.
# --------------------------------------------------------------------------


def _meta_payload():
    """A zero-storage meta tensor of PAYLOAD_BYTES modeled bytes."""
    return torch.empty(DEFAULT_N, dtype=torch.float32, device="meta")


async def _put_delete_put() -> VolumeService:
    """Fill a volume to capacity, prove an over-commit fails, delete, then refit.

    Drives real ``LocalClient`` puts through the transport seam and the *real*
    ``InMemoryStore`` delete path (the volume's ``delete`` endpoint).
    """
    controller = RealControllerAdapter()
    # Capacity for exactly two payloads.
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
    topology = {"0": Endpoint(id="vol0", host="hA", node="nA")}
    svc = VolumeService(volume_id="vol0", profile=profile)
    vol = LocalVolumeHandle(svc)
    volumes = {"0": vol}
    producer = RealClientAdapter(
        controller.handle,
        volume_handles=volumes,
        client_volume_id="0",
        topology=topology,
        profile=profile,
    )

    with producer.installed():
        await producer.client.put("A", _meta_payload())
        await producer.client.put("B", _meta_payload())
        # Full to capacity now.
        assert svc.resident_bytes == PAYLOAD_BYTES * 2
        assert svc.peak_resident_bytes == PAYLOAD_BYTES * 2
        # A third payload over-commits and is rejected (data does not land).
        with pytest.raises(StorageCapacityExceeded):
            await producer.client.put("C", _meta_payload())
        assert svc.resident_bytes == PAYLOAD_BYTES * 2  # unchanged by the failure

    # Free space via the REAL store delete path (mirrors StorageVolume.delete).
    await vol.delete.call_one("A")
    assert svc.resident_bytes == PAYLOAD_BYTES  # A's bytes released

    # The same put that just failed now fits.
    with producer.installed():
        await producer.client.put("C", _meta_payload())
    assert svc.resident_bytes == PAYLOAD_BYTES * 2
    # Peak is a run-lifetime high-water mark; it never regresses on delete.
    assert svc.peak_resident_bytes == PAYLOAD_BYTES * 2
    return svc


def test_delete_frees_space_allows_subsequent_put():
    asyncio.run(_put_delete_put())


# --------------------------------------------------------------------------
# Determinism: the same run reproduces the same resident / peak values.
# --------------------------------------------------------------------------


def test_resident_tracking_is_deterministic():
    _r1, t1, c1 = _run_build()
    _r2, t2, c2 = _run_build()
    assert t1.render() == t2.render()
    for vid in c1["volumes"]:
        assert c1["volumes"][vid].service.resident_bytes == c2["volumes"][vid].service.resident_bytes
        assert (
            c1["volumes"][vid].service.peak_resident_bytes
            == c2["volumes"][vid].service.peak_resident_bytes
        )


# --------------------------------------------------------------------------
# Asking control what to drop, instead of only refusing. The volume is the one
# object that knows it is full and the one that cannot know what is worth
# keeping -- so it asks the directory, which is where a Policy is installed,
# over the same handle every other caller reaches it through.
# --------------------------------------------------------------------------


def _evicting(victims, log=None):
    """A Policy whose only opinion is which keys to drop."""

    class _P(Policy):
        name = "evicting"

        async def select(self, view, keys, requester):
            return Selection()

        async def evict(self, view, volume_id, need_bytes):
            if log is not None:
                log.append((volume_id, need_bytes))
            return victims

    return _P()


async def _put_over_capacity(policy, *, wired=True, keys=("A", "B", "C")):
    """Fill a two-payload volume, then put one more with ``policy`` installed.

    Drives the whole declared path -- ``volume -> controller handle -> service ->
    policy`` -- rather than a stand-in callback, because the point of the handle is
    that the ask is a call to another service.
    """
    controller = RealControllerAdapter()
    if policy is not None:
        controller.handle.install_policy(policy, None)
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
    topology = {"0": Endpoint(id="vol0", host="hA", node="nA")}
    svc = VolumeService(
        volume_id="vol0",
        profile=profile,
        controller=controller.handle if wired else None,
    )
    producer = RealClientAdapter(
        controller.handle,
        volume_handles={"0": LocalVolumeHandle(svc)},
        client_volume_id="0",
        topology=topology,
        profile=profile,
    )
    with producer.installed():
        for key in keys:
            await producer.client.put(key, _meta_payload())
    return svc


def test_the_volume_asks_before_refusing_and_the_put_then_fits():
    """A policy that names a victim turns a refusal into an eviction."""
    asked: list[tuple[str, int]] = []
    svc = asyncio.run(_put_over_capacity(_evicting(["A"], asked)))
    # Asked once, for its own id and exactly the overshoot -- not for everything.
    assert asked == [("vol0", PAYLOAD_BYTES)]
    # A left, C landed, and the accounting is exact rather than reset.
    assert svc.resident_bytes == PAYLOAD_BYTES * 2
    assert sorted(svc.store.kv) == ["B", "C"]


def test_an_answer_that_frees_too_little_still_refuses():
    """Eviction is not a licence to over-commit: the put is still rejected."""
    with pytest.raises(StorageCapacityExceeded):
        asyncio.run(_put_over_capacity(_evicting([])))


def test_a_key_this_put_is_writing_is_never_evicted():
    """Freeing the bytes the caller is about to add would drop the new value."""
    svc = asyncio.run(_put_over_capacity(_evicting(["C", "A"])))
    assert sorted(svc.store.kv) == ["B", "C"]  # C survived, A was taken instead


def test_no_policy_installed_is_the_historical_refusal():
    """The directory answers nothing, so the volume refuses as it always did."""
    with pytest.raises(StorageCapacityExceeded):
        asyncio.run(_put_over_capacity(None))


def test_a_volume_with_no_directory_to_ask_also_refuses():
    """And a volume built without a controller has nobody to ask at all."""
    with pytest.raises(StorageCapacityExceeded):
        asyncio.run(_put_over_capacity(_evicting(["A"]), wired=False))


def test_the_mesh_asks_the_installed_policy():
    """The wiring: a full volume reaches the run's own Policy, not a stub.

    Vacuous if the mesh captured ``None`` at construction -- which it would, since
    volumes are built before a policy is installed -- so this pins that the hook
    reads the policy off the service at call time.
    """
    from proposed import Policy, Selection
    from realsim.simulation import Simulation

    asked: list[tuple[str, int]] = []

    class EvictingPolicy(Policy):
        name = "evicting"

        async def select(self, view, keys, requester):
            return Selection()

        async def evict(self, view, volume_id, need_bytes):
            asked.append((volume_id, need_bytes))
            return ["A"]

    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
    topology = {"0": Endpoint(id="vol0", host="hA", node="nA")}
    sim = Simulation(topology, control=EvictingPolicy(), profile=profile)

    async def scenario():
        with sim.mesh.installed():
            # client_for binds the source endpoint the transport prices against.
            for key in ("A", "B", "C"):
                await sim.mesh.client_for("0").put(key, _meta_payload())

    try:
        sim.loop.run_until_complete(scenario())
    finally:
        sim.loop.close()

    assert asked == [("vol0", PAYLOAD_BYTES)], asked
    assert sorted(sim.mesh.volumes["0"].service.store.kv) == ["B", "C"]
