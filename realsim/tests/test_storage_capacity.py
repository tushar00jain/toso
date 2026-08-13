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
* evicting before refusing -- a full volume drops its own least-recently-used
  keys, tells the directory they are gone, and refuses only if that still does not
  make room.

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
from realsim.seams.transport import Endpoint
from realsim.seams.volume_handle import LocalVolumeHandle
from realsim.seams.volume_service import StorageCapacityExceeded, VolumeService
from sim_common.cost_model import DEFAULT_PROFILE
from torchstore.transport.buffers import TransportCache

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
    # The volume's directory identity is the node id, not the endpoint id the
    # transport charges against (``origin`` below) -- it is what the co-located
    # client registers puts under, so it is what a dropped key is reported under.
    assert origin_vol.volume_id == "p"
    assert origin == "volp"
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
    assert exc.volume_id == "p"  # the origin volume's directory id
    assert exc.capacity == cap
    assert exc.resident == 0  # nothing resident before the failed seed put
    assert exc.attempted == PAYLOAD_BYTES
    # The message names the volume, its capacity, and the attempted bytes.
    msg = str(exc)
    assert "'p'" in msg
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
    # ``volume_id`` is the directory identity: the same id the client below
    # registers its puts under (``client_volume_id``), not the endpoint id.
    svc = VolumeService(volume_id="0", profile=profile)
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


async def _reset_then_refill() -> VolumeService:
    """Fill a volume, reset it, fill it again, then push it over capacity."""
    controller = RealControllerAdapter()
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
    topology = {"0": Endpoint(id="vol0", host="hA", node="nA")}
    svc = VolumeService(volume_id="0", profile=profile, controller=controller.handle)
    producer = RealClientAdapter(
        controller.handle,
        volume_handles={"0": LocalVolumeHandle(svc)},
        client_volume_id="0",
        topology=topology,
        profile=profile,
    )
    with producer.installed():
        for key in ("A", "B"):
            await producer.client.put(key, _meta_payload())
    await svc.reset()
    assert svc.resident_bytes == 0
    # A fresh working set under different names, then one payload too many.
    with producer.installed():
        for key in ("X", "Y", "Z"):
            await producer.client.put(key, _meta_payload())
    return svc


def test_a_reset_volume_can_still_make_room():
    """Reset empties the ranking too, or it ranks a working set that is gone.

    The keys a reset dropped are *colder* than anything written afterwards, so a
    ranking that still knows them names them first when the volume next needs
    room. They free nothing -- they are not there -- and the put is refused
    although there was a real victim available all along.
    """
    svc = asyncio.run(_reset_then_refill())
    assert sorted(svc.store.kv) == ["Y", "Z"]  # X was the coldest one that existed


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
# keeping -- so it asks the directory, which is where a KeySelector is installed,
# over the same handle every other caller reaches it through.
# --------------------------------------------------------------------------


async def _put_over_capacity(
    selector=None, *, wired=True, keys=("A", "B", "C"), setup=None
):
    """Fill a two-payload volume, then put one more and let it make room.

    Returns ``(volume service, controller adapter)`` so a caller can check the
    directory as well as the store -- an eviction is not done until the directory
    has stopped saying this volume holds the key. ``setup`` is handed the volume
    before the puts, for per-key state the eviction is supposed to release.
    """
    controller = RealControllerAdapter()
    profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
    topology = {"0": Endpoint(id="vol0", host="hA", node="nA")}
    svc = VolumeService(
        volume_id="0",  # the directory identity, i.e. the client's volume id
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
    if setup is not None:
        setup(svc)
    with producer.installed():
        for key in keys:
            await producer.client.put(key, _meta_payload())
    return svc, controller


def test_a_full_volume_evicts_its_own_coldest():
    """Plain LRU is local knowledge: the volume needs nobody's help to apply it."""
    svc, _controller = asyncio.run(_put_over_capacity())
    assert sorted(svc.store.kv) == ["B", "C"]      # A was the coldest, and went


def test_the_directory_is_told_what_was_evicted():
    """The bytes are gone, so the one service that routes reads has to know.

    The volume reports the drop under its *directory* id; naming itself anything
    else would leave the real ``Controller`` still listing it as a holder (its
    ``_notify_delete`` is ``missing_ok`` and ignores a volume it does not know),
    and the next read would be routed here for a key that is not.
    """
    _svc, controller = asyncio.run(_put_over_capacity())
    directory = controller.controller.keys_to_storage_volumes
    assert "A" not in directory                    # the evicted key, unrouted
    assert list(directory["B"]) == ["0"]           # what is still held, still listed




class _KeyedCache(TransportCache):
    """A per-key transport cache, like the shared-memory and process-group ones.

    Stands in for whatever a real transport registers per key in the volume's
    ``TransportContext``. What matters is only that it is keyed by store key, so
    the entry names a resource that has to be released when the key goes.
    """

    def __init__(self) -> None:
        self.entries: set[str] = set()

    def delete(self, keys: set[str]) -> None:
        self.entries -= keys

    def clear(self) -> None:
        self.entries.clear()


def test_an_evicted_key_releases_what_a_deleted_key_releases():
    """Eviction is a delete, so it goes through the volume's delete.

    Dropping the value and the resident bytes is only two thirds of it: the key's
    transport-cache entry names a live resource (a shared-memory segment, a
    process group), and leaving it behind leaks that resource for a key nobody
    can ask for again. The eviction path used to open-code the other two steps
    and miss this one.
    """

    async def _evict_then_delete():
        svc, _controller = await _put_over_capacity(
            setup=lambda vol: vol.store.transport_context.get(
                _KeyedCache
            ).entries.update({"A", "B"})
        )
        cache = svc.store.transport_context.get(_KeyedCache)
        # "A" was the coldest and was evicted to make room for "C".
        assert sorted(svc.store.kv) == ["B", "C"]
        assert cache.entries == {"B"}, "the evicted key's entry outlived it"
        # ...and the ordinary delete does exactly what that eviction did.
        await svc.delete("B")
        assert cache.entries == set()

    asyncio.run(_evict_then_delete())


def test_a_key_this_put_is_writing_is_never_evicted():
    """Freeing the bytes the caller is about to add would drop the new value."""
    svc, _controller = asyncio.run(_put_over_capacity(keys=("A", "B", "A")))
    assert sorted(svc.store.kv) == ["A", "B"]


def test_a_key_this_put_is_writing_does_not_count_as_room_it_freed():
    """The excluded key is skipped by the ranking, not filtered out of its answer.

    Growing the coldest key asks for one payload's room. The key being written is
    never a victim -- but if that exclusion is applied *after* the scan, the
    ranking still hands it back first, counts its bytes toward the need and stops
    there; the caller then filters it away, frees nothing, and refuses a put that
    "B" had exactly enough room for.
    """

    async def _grow_the_coldest_key():
        controller = RealControllerAdapter()
        profile = replace(DEFAULT_PROFILE, storage_capacity_bytes=PAYLOAD_BYTES * 2)
        svc = VolumeService(
            volume_id="0", profile=profile, controller=controller.handle
        )
        producer = RealClientAdapter(
            controller.handle,
            volume_handles={"0": LocalVolumeHandle(svc)},
            client_volume_id="0",
            topology={"0": Endpoint(id="vol0", host="hA", node="nA")},
            profile=profile,
        )
        with producer.installed():
            await producer.client.put("A", _meta_payload())
            await producer.client.put("B", _meta_payload())
            # "A" is the coldest key, and is the one being overwritten -- by a
            # value one payload bigger than the one it replaces.
            await producer.client.put(
                "A", torch.empty(DEFAULT_N * 2, dtype=torch.float32, device="meta")
            )
        return svc

    svc = asyncio.run(_grow_the_coldest_key())
    # "B" was the only droppable key, and dropping it is what made room.
    assert sorted(svc.store.kv) == ["A"]
    assert svc.resident_bytes == PAYLOAD_BYTES * 2


def test_a_volume_with_no_directory_cannot_report_what_it_dropped():
    """Eviction is local, but telling the directory is not: without a handle the
    volume has nobody to tell, so it refuses rather than drop silently."""
    with pytest.raises(StorageCapacityExceeded):
        asyncio.run(_put_over_capacity(wired=False))


