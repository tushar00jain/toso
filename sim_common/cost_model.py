"""Analytic, target-machine resource cost model (sibling to ``topology.py``).

Every cost here is a *deterministic function of a modeled quantity* (``nbytes``
or ``flops``) and a caller-supplied :class:`MachineProfile` of target-hardware
constants. Nothing is measured on the host running the simulation -- measuring
would couple the sim to the test box and misrepresent the production machine.
This module generalizes :func:`sim_common.topology.transfer_time` (network is
just one resource); it does not fork it -- :func:`network_time` wraps it.

Design shape (mirrors ``transfer_time``):

* constants live in the profile, never baked into the functions;
* each cost fn is ``modeled_quantity x profile -> time`` and returns ``0.0`` for
  a zero quantity (or a same-endpoint network transfer);
* pure arithmetic only -- no clocks, threads, RNG, or measurement, so the whole
  module passes ``realsim/tools/check_contract.py``.

Units are arbitrary but must be *consistent within a profile*: a bandwidth is
``bytes / time`` and a flop rate is ``flops / time``, so ``nbytes / bandwidth``
and ``flops / flop_rate`` both come out in the profile's time unit (the same
unit ``topology.transfer_time`` produces). :data:`DEFAULT_PROFILE` is an
**illustrative** demo profile -- plausible relative magnitudes, *not measured*
from any real device; production callers supply their own from scenario config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Tuple

from sim_common.topology import locality, Tier, transfer_time

__all__ = [
    "MachineProfile",
    "DEFAULT_PROFILE",
    "network_time",
    "network_rate",
    "mem_copy_time",
    "storage_time",
    "storage_rate",
    "compute_time",
    "ProfileTransferCost",
]

# Device families the compute roofline understands. Anything in ``_GPU_DEVICES``
# uses ``gpu_flops`` / ``gpu_mem_bandwidth``; everything else falls back to the
# host (``cpu_flops`` / ``ram_bandwidth``).
_GPU_DEVICES = frozenset({"cuda", "gpu", "hip", "xpu"})
_STORAGE_KINDS = frozenset({"read", "write"})


@dataclass(frozen=True)
class MachineProfile:
    """Target-hardware constants for the analytic cost model (caller-supplied).

    All fields are hardware descriptors kept in scenario/config, never hard-coded
    inside a seam. Bandwidths are ``bytes / time`` and flop rates are
    ``flops / time`` in one consistent (arbitrary) time unit.

    Network:
        tiers: per-:class:`~sim_common.topology.Tier` ``(latency, bandwidth)``
            map, exactly the shape :func:`~sim_common.topology.transfer_time`
            consumes.

    RAM / host memory:
        ram_bandwidth: host memory copy bandwidth (bytes/time).
        ram_latency: fixed per-copy host memory latency (time).

    Persistent storage:
        storage_read_bw / storage_write_bw: read / write bandwidth (bytes/time).
        storage_latency: fixed per-op storage latency (time), applied to both
            read and write.
        storage_capacity_bytes: byte capacity of a single storage volume,
            enforced against the aggregate resident working set on that volume
            (the sum of all bytes currently resident) by the volume seam (see
            :class:`realsim.seams.volume_service.VolumeService`). Like the
            bandwidths this is a target-hardware descriptor, not a debug knob --
            it changes the simulated result (an over-commit raises instead of
            silently fitting), so it is an explicit profile field rather than an
            ambient config flag. Defaults to ``math.inf`` (unbounded), which
            disables the check and keeps the historical behavior byte-identical;
            enforcement is active exactly when this is finite.

    Compute:
        gpu_flops: per-dtype device flop rate (flops/time), keyed by dtype name
            (e.g. ``"float32"``, ``"bfloat16"``). Missing dtypes fall back to
            :attr:`gpu_flops_default` -- peak flops legitimately vary by dtype on
            real accelerators, so this is a mapping rather than a scalar.
        gpu_flops_default: fallback device flop rate for a dtype not in
            :attr:`gpu_flops`.
        gpu_mem_bandwidth: device (HBM) memory bandwidth (bytes/time); the memory
            term of the compute roofline on a GPU device.
        cpu_flops: host flop rate (flops/time); a single scalar (host dtype
            differences are not modeled). The host memory term of the roofline
            uses :attr:`ram_bandwidth`.

    Host<->device transfer:
        h2d_bandwidth / d2h_bandwidth: host-to-device / device-to-host copy
            bandwidth (bytes/time). Optional helpers for callers that model PCIe
            staging; the core cost fns do not require them.
    """

    tiers: Mapping[Tier, Tuple[float, float]]

    ram_bandwidth: float
    ram_latency: float = 0.0

    storage_read_bw: float = 0.0
    storage_write_bw: float = 0.0
    storage_latency: float = 0.0
    storage_capacity_bytes: float = math.inf

    gpu_flops: Mapping[str, float] = field(default_factory=dict)
    gpu_flops_default: float = 0.0
    gpu_mem_bandwidth: float = 0.0

    cpu_flops: float = 0.0

    h2d_bandwidth: float = 0.0
    d2h_bandwidth: float = 0.0


# Illustrative demo profile -- plausible *relative* magnitudes only, NOT measured
# from real hardware. Reuses the per-tier network constants convention from
# ``realsim/seams/transport.py``. Callers building real scenarios should supply
# their own profile instead of leaning on these numbers.
DEFAULT_PROFILE = MachineProfile(
    tiers={
        Tier.SHM: (0.0001, 150000.0),
        Tier.NVLINK: (0.0002, 60000.0),
        Tier.RDMA: (0.0010, 10000.0),
    },
    ram_bandwidth=200000.0,
    ram_latency=0.00005,
    storage_read_bw=20000.0,
    storage_write_bw=10000.0,
    storage_latency=0.001,
    gpu_flops={
        "float32": 1.0e9,
        "float16": 2.0e9,
        "bfloat16": 2.0e9,
    },
    gpu_flops_default=1.0e9,
    gpu_mem_bandwidth=800000.0,
    cpu_flops=5.0e7,
    h2d_bandwidth=25000.0,
    d2h_bandwidth=25000.0,
)


def _is_gpu(device: str) -> bool:
    """True if ``device`` names a compute accelerator (vs. the host CPU)."""
    return device.lower() in _GPU_DEVICES


def _effective_flops(flops_dtype: str, device: str, profile: MachineProfile) -> float:
    """Resolve the peak flop rate for ``(dtype, device)`` from the profile.

    GPU devices use the per-dtype :attr:`MachineProfile.gpu_flops` map (falling
    back to :attr:`gpu_flops_default`); the host uses the scalar
    :attr:`cpu_flops`.
    """
    if _is_gpu(device):
        return profile.gpu_flops.get(flops_dtype, profile.gpu_flops_default)
    return profile.cpu_flops


def _mem_bandwidth(device: str, profile: MachineProfile) -> float:
    """Memory bandwidth backing the roofline's memory term for ``device``."""
    return profile.gpu_mem_bandwidth if _is_gpu(device) else profile.ram_bandwidth


def network_time(src, dst, nbytes: int, profile: MachineProfile) -> float:
    """Time to move ``nbytes`` from ``src`` to ``dst`` over the network.

    Thin wrapper over :func:`sim_common.topology.transfer_time` using the
    profile's per-tier ``(latency, bandwidth)`` map. A same-endpoint or zero-byte
    transfer is free (delegated to ``transfer_time``). ``src``/``dst`` are
    duck-typed on ``.id``/``.host``/``.node`` (see
    :class:`sim_common.topology.Endpoint`).
    """
    return transfer_time(src, dst, nbytes, profile.tiers)


def network_rate(src, dst, profile: MachineProfile) -> Tuple[float, float]:
    """Return the ``(latency, bandwidth)`` of the fabric tier between two endpoints.

    The contention-model decomposition of :func:`network_time`: for a non-trivial
    transfer, ``network_time(...) == latency + nbytes / bandwidth`` with exactly
    this ``(latency, bandwidth)``. The bandwidth is the shared quantity when
    several transfers compete for one link (see :mod:`sim_common.resources`).

    Callers must guard the free cases first: :func:`network_time` returns ``0.0``
    for a same-endpoint or zero-byte transfer *without* consulting a tier, so this
    helper is only meaningful for a real cross-endpoint transfer.
    """
    return profile.tiers[locality(src, dst)]


def mem_copy_time(nbytes: int, profile: MachineProfile) -> float:
    """Time to copy ``nbytes`` through host RAM (latency + bytes/bandwidth).

    Returns ``0.0`` for a zero-byte copy.
    """
    if nbytes <= 0:
        return 0.0
    return profile.ram_latency + nbytes / profile.ram_bandwidth


def storage_time(nbytes: int, kind: str, profile: MachineProfile) -> float:
    """Time to read or write ``nbytes`` to persistent storage.

    ``kind`` is ``"read"`` or ``"write"`` (selecting the matching bandwidth);
    the fixed :attr:`MachineProfile.storage_latency` is added to both. Returns
    ``0.0`` for a zero-byte op. Raises :class:`ValueError` for an unknown kind.
    """
    if kind not in _STORAGE_KINDS:
        raise ValueError(f"storage kind must be one of {sorted(_STORAGE_KINDS)}, got {kind!r}")
    if nbytes <= 0:
        return 0.0
    bw = profile.storage_read_bw if kind == "read" else profile.storage_write_bw
    return profile.storage_latency + nbytes / bw


def _get_time(src, dst, nbytes: int, profile: MachineProfile) -> float:
    """Total time to serve one ``get`` of ``nbytes`` from ``src`` to ``dst``.

    The **canonical composition** of a read: the serving side reads the payload
    back from persistent storage, stages it through host RAM, and ships it over
    the fabric -- ``storage_time(read) + mem_copy_time + network_time``.

    This exists so the component that *charges* a get and any component that
    *predicts* one share a single definition. ``realsim``'s transport seam charges
    exactly these three terms (as three virtual-clock sleeps, or one combined
    sleep under ``collapse_charges``), and a scheduler that predicts a fetch cost
    to route on must agree with them to the last float -- otherwise routing
    decisions are made against a stale model and nothing fails loudly.
    ``realsim/tests/test_cost_parity.py`` pins the two together.

    ``network_time`` is symmetric in its endpoints (it prices the locality
    :class:`~sim_common.topology.Tier` between them), so the argument order here
    is "who serves" / "who receives" and does not affect the result.

    Note this does **not** special-case a same-endpoint get: reading from a
    co-located volume still costs storage + RAM, only the fabric term is zero.
    """
    if nbytes <= 0:
        return 0.0
    return (
        storage_time(nbytes, "read", profile)
        + mem_copy_time(nbytes, profile)
        + network_time(src, dst, nbytes, profile)
    )


def storage_rate(kind: str, profile: MachineProfile) -> Tuple[float, float]:
    """Return the ``(latency, bandwidth)`` of a storage read/write channel.

    The contention-model decomposition of :func:`storage_time`: for a non-empty
    op, ``storage_time(nbytes, kind, profile) == latency + nbytes / bandwidth``
    with exactly this pair. The bandwidth is the shared quantity when several
    ops hit one volume's read (or write) channel (see
    :mod:`sim_common.resources`). Raises :class:`ValueError` for an unknown kind.
    """
    if kind not in _STORAGE_KINDS:
        raise ValueError(f"storage kind must be one of {sorted(_STORAGE_KINDS)}, got {kind!r}")
    bw = profile.storage_read_bw if kind == "read" else profile.storage_write_bw
    return profile.storage_latency, bw


def compute_time(
    flops: float,
    dtype: str,
    device: str,
    profile: MachineProfile,
    nbytes: int = 0,
) -> float:
    """Roofline compute time: ``max(flops / effective_flops, nbytes / mem_bw)``.

    A kernel is bounded by whichever is slower: doing the arithmetic at the
    device's peak flop rate for ``dtype``, or streaming ``nbytes`` through its
    memory. Pass ``nbytes`` (the bytes the kernel touches) to model the
    memory-bound side; omit it (``0``) for a pure-compute estimate.

    ``device`` selects the flop rate and memory bandwidth: a GPU-family device
    (``"cuda"``/``"gpu"``/...) uses :attr:`MachineProfile.gpu_flops` +
    :attr:`gpu_mem_bandwidth`; anything else uses :attr:`cpu_flops` +
    :attr:`ram_bandwidth`. Returns ``0.0`` when both quantities are zero.
    """
    compute_term = 0.0
    if flops > 0:
        eff = _effective_flops(dtype, device, profile)
        compute_term = flops / eff

    mem_term = 0.0
    if nbytes > 0:
        mem_term = nbytes / _mem_bandwidth(device, profile)

    return max(compute_term, mem_term)


class ProfileTransferCost:
    """A :data:`proposed.cost.TransferCost` backed by a profile + topology.

    The simulator's implementation: prices a transfer with :func:`_get_time`,
    the sum of the same three terms the transport seam charges one at a time, so
    a scheduler's prediction and the clock advance it causes cannot drift apart
    (``realsim/tests/test_cost_parity.py`` holds the two to each other).
    """

    def __init__(self, topology, profile: MachineProfile = DEFAULT_PROFILE) -> None:
        self._topology = dict(topology)
        self._profile = profile

    def get_time(self, src_id: str, dst_id: str, nbytes: int) -> float:
        return _get_time(
            self._topology[src_id], self._topology[dst_id], nbytes, self._profile
        )
