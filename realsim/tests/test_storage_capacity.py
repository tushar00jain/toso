"""Storage-volume byte-capacity enforcement (opt-in via the ``MachineProfile``).

The transport seam models transfer *time*; these tests exercise the *capacity*
model that lives in the volume seam (:class:`realsim.seams.volume_handle.FakeVolumeHandle`):
each volume tracks its aggregate resident working set (bytes added on put,
subtracted on delete) around the real ``InMemoryStore`` lifecycle, and rejects a
put that would over-commit a *finite* ``MachineProfile.storage_capacity_bytes``.

Gates:

* default ``math.inf`` capacity -- unbounded, historical behavior; resident /
  peak tracking still works and nothing is raised;
* within a finite capacity -- a burst whose resident set fits succeeds and
  ``peak_resident_bytes`` reflects the real peak;
* over a finite capacity -- the over-committing put raises
  :class:`~realsim.seams.volume_handle.StorageCapacityExceeded` carrying the
  volume id / capacity / resident / attempted bytes;
* a real delete frees space -- resident drops and a put that would otherwise
  exceed capacity then fits (peak stays a run-lifetime high-water mark);
* determinism -- the same run reproduces the same resident / peak values.

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
from realsim.entrypoint import run_simulation
from putget_sim.workload.put_get import DEFAULT_N, PutGetBurst
from realsim.seams.transport import Endpoint
from realsim.seams.volume_handle import FakeVolumeHandle, StorageCapacityExceeded
from sim_common.async_engine import run_sim
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
    result = run_simulation(workload, profile=workload.profile)
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
    assert origin_vol.resident_bytes == PAYLOAD_BYTES
    assert origin_vol.peak_resident_bytes == PAYLOAD_BYTES
    assert origin_vol.capacity_bytes == float("inf")
    for vid, vol in volumes.items():
        if vid != "p":
            assert vol.resident_bytes == 0
            assert vol.peak_resident_bytes == 0


# --------------------------------------------------------------------------
# Within a finite capacity: the burst fits and peak reflects the real peak.
# --------------------------------------------------------------------------


def test_within_capacity_succeeds_and_peak_reflects_real_peak():
    # Capacity generously above one payload -> the seed put fits.
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 4)
    _results, _trace, ctx = _run_build(profile=profile)  # must not raise

    origin_vol = ctx["volumes"]["p"]
    assert origin_vol.capacity_bytes == PAYLOAD_BYTES * 4
    assert origin_vol.resident_bytes == PAYLOAD_BYTES
    assert origin_vol.peak_resident_bytes == PAYLOAD_BYTES


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


async def _put_delete_put() -> FakeVolumeHandle:
    """Fill a volume to capacity, prove an over-commit fails, delete, then refit.

    Drives real ``LocalClient`` puts through the transport seam and the *real*
    ``InMemoryStore`` delete path (the volume's ``delete`` endpoint).
    """
    controller = RealControllerAdapter()
    # Capacity for exactly two payloads.
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
    topology = {"0": Endpoint(id="vol0", host="hA", node="nA")}
    vol = FakeVolumeHandle(volume_id="vol0", profile=profile)
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
        assert vol.resident_bytes == PAYLOAD_BYTES * 2
        assert vol.peak_resident_bytes == PAYLOAD_BYTES * 2
        # A third payload over-commits and is rejected (data does not land).
        with pytest.raises(StorageCapacityExceeded):
            await producer.client.put("C", _meta_payload())
        assert vol.resident_bytes == PAYLOAD_BYTES * 2  # unchanged by the failure

    # Free space via the REAL store delete path (mirrors StorageVolume.delete).
    await vol.delete.call_one("A")
    assert vol.resident_bytes == PAYLOAD_BYTES  # A's bytes released

    # The same put that just failed now fits.
    with producer.installed():
        await producer.client.put("C", _meta_payload())
    assert vol.resident_bytes == PAYLOAD_BYTES * 2
    # Peak is a run-lifetime high-water mark; it never regresses on delete.
    assert vol.peak_resident_bytes == PAYLOAD_BYTES * 2
    return vol


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
        assert c1["volumes"][vid].resident_bytes == c2["volumes"][vid].resident_bytes
        assert (
            c1["volumes"][vid].peak_resident_bytes
            == c2["volumes"][vid].peak_resident_bytes
        )
