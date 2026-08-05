"""A synchronized read-burst scenario over the real TorchStore.

A full-replication read burst that drives the **real** client planning core, the
**real** controller directory, and the **real** ``InMemoryStore`` through the
in-process seams, all on the deterministic virtual-clock engine.

Layout: one *origin* volume on node ``P`` holds ``W``; ``m`` reader volumes on
distinct hosts of node ``R`` each want overlapping data (by default the whole
tensor -- maximal overlap). Under the naive policy every reader pulls from the
origin, so fabric is ``m x`` the payload -- the baseline a dedup policy (see
:class:`~realsim.coordinator.model.ReadPolicy`) would cut toward the 1x union by
registering read-through peers in the real directory.

The whole run (seed put + read burst) executes on one :class:`AsyncEngine`, so
one continuous virtual-time trace is produced and is byte-identical across runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from realsim.coordinator.model import (
    BurstMetrics,
    NaivePolicy,
    Reader,
    ReadCoordinator,
    ReadPolicy,
)
from realsim.mesh import Mesh
from realsim.seams.transport import Endpoint, TensorDescriptor
from sim_common.async_engine import run_sim
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile, compute_time
from sim_common.report import render_tree
from sim_common.trace import Trace

KEY = "W"
DEFAULT_N = 16  # elements in W (float32 -> 4 bytes each)

# TODO(next diff): derive this payload from a ``realsim.model.Model`` instead of a
# bare element count -- see the TODO in ``realsim/model.py`` for the full plan.

# Producer compute model. We assume "generating" W costs a small, constant number
# of flops per element (a stand-in for e.g. a fused multiply-add per element in
# the kernel that materializes the tensor) and touches the whole payload's bytes.
# The roofline in ``cost_model.compute_time`` then picks the binding term
# (compute vs. memory bandwidth) on the producer's device. This is a deliberately
# simple, deterministic model -- realistic *relative* magnitudes, not a claim
# about any specific kernel.
FLOPS_PER_ELEMENT = 2.0

# The producer generates W on an accelerator (the "generator" in a
# training/inference pipeline), so its compute is charged against the profile's
# GPU roofline. This is the modeled *target* device, independent of the meta/
# metadata carrier the data plane uses.
DEFAULT_COMPUTE_DEVICE = "cuda"


def _dtype_name(dtype: torch.dtype) -> str:
    """Cost-model dtype key for a torch dtype (``torch.float32`` -> ``float32``)."""
    return str(dtype).replace("torch.", "")

# The two allocation-free data-plane carriers (see docs/realsim_design.md
# section 7). Both drive the *real* client/controller/InMemoryStore round-trip with
# zero real tensor storage:
#   "meta"     -- the payload is a ``device="meta"`` tensor (a real ``torch.Tensor``
#                 with zero storage but exact shape/dtype/nbytes). Default.
#   "metadata" -- no tensor at all; a ``(shape, dtype)`` :class:`TensorDescriptor`
#                 stands in for the payload (``tensor_val is None``).
MODE_META = "meta"
MODE_METADATA = "metadata"


@dataclass
class BurstResult:
    """Output of one burst run."""

    trace: Trace
    metrics: BurstMetrics
    # reader_id -> fetched payload. In "meta" mode a meta ``torch.Tensor``; in
    # "metadata" mode a :class:`~realsim.seams.transport.TensorDescriptor`.
    results: Dict[str, Any]
    # The ground-truth W carrier (meta tensor or descriptor); both expose
    # ``shape``/``dtype``/``numel()``/``element_size()`` so callers can size it.
    expected: Any
    origin_id: str
    num_readers: int


def _topology(num_readers: int) -> Dict[str, Endpoint]:
    """Origin volume ``p`` on node P; readers ``r0..`` on distinct hosts of node R."""
    topo: Dict[str, Endpoint] = {
        "p": Endpoint(id="volp", host="hP", node="P"),
    }
    for i in range(num_readers):
        topo[f"r{i}"] = Endpoint(id=f"volr{i}", host=f"hR{i}", node="R")
    return topo


def build_burst(
    num_readers: int = 3,
    *,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    device: str = "meta",
    profile: Optional[MachineProfile] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    policy: Optional[ReadPolicy] = None,
    trace: Optional[Trace] = None,
    metrics: Optional[BurstMetrics] = None,
    real_directory: Optional[bool] = None,
):
    """Wire the real objects for a burst; return ``(scenario_coro, ctx)``.

    ``ctx`` carries the pieces a caller/test may want to assert on (the expected
    payload, origin id, coordinator, trace, metrics). ``scenario_coro`` is an
    awaitable that seeds ``W`` on the origin then runs the burst; hand it to
    :func:`sim_common.async_engine.run_sim`.

    ``profile`` is the target-machine
    :class:`~sim_common.cost_model.MachineProfile` (defaults to
    :data:`~sim_common.cost_model.DEFAULT_PROFILE`). It supplies **every** cost
    constant -- it models the *target* hardware being simulated, never the box the
    test runs on. It is threaded to the producer, the readers, and the coordinator
    so network + storage + RAM + compute are all charged from one story.
    ``compute_device`` selects the roofline device for the producer's generate
    step (default ``"cuda"`` -- the accelerator that produces W).

    ``real_directory`` selects the controller directory backing (``None`` -> the
    ambient :data:`sim_common.config.SimConfig.real_directory`, default real
    ``Trie``; ``False`` -> the lightweight dict shim). It changes no metric.

    The network/storage contention model is read ambiently from
    :data:`sim_common.config.SimConfig.contention` (default ``"none"``). The
    :class:`~realsim.mesh.Mesh` builds one shared
    :class:`~sim_common.resources.ResourceRegistry` and injects it into the
    producer, every reader adapter, and its shared transport factory, so all
    transfers in the burst see the same links/stores. Unlike ``real_directory`` a
    non-``"none"`` mode DOES change timing.

    The full resource model exercised by one run:

    * **compute/GPU** -- the producer's generate step (charged here, before the
      put), via :func:`~sim_common.cost_model.compute_time` on ``compute_device``;
    * **network** -- client<->volume fabric transfers (transport seam);
    * **storage** -- write on put, read on serve (transport seam);
    * **RAM** -- host-memory staging copy on serve (transport seam).

    Data plane (allocation-free by default):

    * ``mode="meta"`` (default): ``W`` is a ``torch.empty(n, dtype=dtype,
      device=device)`` tensor, with ``device="meta"`` -- a real tensor of zero
      storage. It passes every ``isinstance``/``shape``/``dtype`` check in the
      real torchstore path and is passed by reference through the fake volume
      handle, so nothing is serialized or allocated.
    * ``mode="metadata"``: ``W`` is a :class:`TensorDescriptor` carrying only
      ``(shape, dtype)`` -- no tensor at all. It is handed to ``client.put`` as an
      arbitrary object so it flows through the real *object* put/get path; the
      transport seam reads ``nbytes`` off the descriptor (``tensor_val is None``).
      See the value-typing note below.
    """
    if mode not in (MODE_META, MODE_METADATA):
        raise ValueError(f"unknown data-plane mode {mode!r}")

    trace = trace if trace is not None else Trace()
    metrics = metrics if metrics is not None else BurstMetrics()
    policy = policy if policy is not None else NaivePolicy()
    profile = profile if profile is not None else DEFAULT_PROFILE

    topology = _topology(num_readers)
    # The mesh builds every real object the burst runs on: the controller adapter
    # (real Trie by default; the opt-in shim swaps only the directory container),
    # a real volume + co-located real LocalClient per node, the shared resource
    # registry, and the shared transport factory.
    mesh = Mesh(
        topology, profile=profile, trace=trace, real_directory=real_directory
    )
    origin_id = topology["p"].id  # the volume that holds W before the burst

    # The producer writes W to its co-located origin volume "p". This is a
    # single-client drive (before the burst), so it uses the adapter's own
    # factory rather than the mesh's shared one.
    producer = mesh.adapter("p")

    # Reader clients are what the coordinator fans out; the coordinator installs
    # the mesh's shared transport factory for the burst, so the readers' own
    # adapter factories are unused during the burst.
    reader_ids = [f"r{i}" for i in range(num_readers)]
    reader_adapters = [mesh.adapter(vid) for vid in reader_ids]
    readers: List[Reader] = [
        Reader(id=vid, client=mesh.client(vid), endpoint=topology[vid])
        for vid in reader_ids
    ]

    coordinator = ReadCoordinator(
        mesh,
        origin_ids={origin_id},
        policy=policy,
        metrics=metrics,
    )

    # W is the value we seed on the origin. Both carriers are allocation-free.
    descriptor = TensorDescriptor(shape=(n,), dtype=dtype)
    if mode == MODE_META:
        # Real tensor, zero storage (device="meta").
        expected: Any = torch.empty(n, dtype=dtype, device=device)
        put_value: Any = expected
    else:  # MODE_METADATA
        # No tensor at all -- the descriptor *is* the payload.
        #
        # put_batch value-typing gotcha (torchstore/client.py): put_batch types
        # the value -- a Tensor/DTensor takes the tensor path, everything else
        # (including ``None``) takes ``Request.from_objects`` and is stored as an
        # OBJECT. Rather than fight that, we lean into it: we hand the descriptor
        # itself (an arbitrary object) to ``client.put``, so it round-trips
        # through the real object put/get path (kv[key] = {"obj": descriptor}) and
        # the reader gets the descriptor back. No volume-handle/RPC-hook bypass is
        # needed; the only seam change is that ``_nbytes`` reads the modeled size
        # off the descriptor when ``tensor_val is None``. The descriptor thus
        # carries (shape, dtype) out-of-band for the cost model.
        expected = descriptor
        put_value = descriptor

    async def scenario_coro() -> Dict[str, Any]:
        # (1) COMPUTE/GPU: the producer "generates" W on its accelerator. Modeled
        # as FLOPS_PER_ELEMENT per element, streaming the whole payload's bytes;
        # the roofline picks the binding term on ``compute_device``. Charged on
        # the virtual clock before the put and recorded in the trace.
        gen_nbytes = descriptor.nbytes
        gen_flops = FLOPS_PER_ELEMENT * descriptor.numel()
        gen_dt = compute_time(
            gen_flops, _dtype_name(dtype), compute_device, profile, gen_nbytes
        )
        await asyncio.sleep(gen_dt)
        trace.record(
            asyncio.get_running_loop().time(),
            "compute",
            f"generate {KEY} flops={gen_flops:g} {gen_nbytes}B "
            f"dev={compute_device} cost={gen_dt:.4f}",
        )
        # (2) Seed W on the origin volume (also populates the real directory). The
        # put path charges network + storage-write in the transport seam.
        with producer.installed():
            await producer.client.put(KEY, put_value)
        # (3) The synchronized read burst. Each get charges storage-read + RAM
        # staging + network in the transport seam.
        return await coordinator.run_burst(readers, KEY)

    ctx = {
        "mesh": mesh,
        "controller": mesh.controller,
        "producer": producer,
        "reader_adapters": reader_adapters,
        "readers": readers,
        "coordinator": coordinator,
        "topology": topology,
        "volumes": mesh.volumes,
        "expected": expected,
        "descriptor": descriptor,
        "mode": mode,
        "origin_id": origin_id,
        "trace": trace,
        "metrics": metrics,
    }
    return scenario_coro, ctx


def run_burst(
    num_readers: int = 3,
    *,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    device: str = "meta",
    profile: Optional[MachineProfile] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    policy: Optional[ReadPolicy] = None,
    random_seed: Optional[int] = None,
    real_directory: Optional[bool] = None,
) -> BurstResult:
    """Run one burst end-to-end on a fresh deterministic engine."""
    trace = Trace()
    metrics = BurstMetrics()
    scenario_coro, ctx = build_burst(
        num_readers,
        n=n,
        dtype=dtype,
        mode=mode,
        device=device,
        profile=profile,
        compute_device=compute_device,
        policy=policy,
        trace=trace,
        metrics=metrics,
        real_directory=real_directory,
    )
    results, trace = run_sim(scenario_coro(), random_seed=random_seed, trace=trace)
    return BurstResult(
        trace=trace,
        metrics=metrics,
        results=results,
        expected=ctx["expected"],
        origin_id=ctx["origin_id"],
        num_readers=num_readers,
    )


def render_burst_summary(res: BurstResult) -> str:
    """Render the fabric/wallclock summary + the source->dest tree."""
    payload = res.expected.numel() * res.expected.element_size()
    union = payload  # the 1x target: W crosses the fabric once
    fabric_x = res.metrics.fabric_bytes / union if union else 0.0
    lines = [
        f"readers: {res.num_readers}   payload(W): {payload}B   "
        f"1x-union target: {union}B",
        f"fabric(origin->readers): {res.metrics.fabric_bytes}B ({fabric_x:.1f}x)   "
        f"total delivered: {res.metrics.total_get_bytes}B",
        f"wallclock: {res.metrics.wallclock:.4f}   "
        f"readers done: {res.metrics.readers_done}/{res.metrics.readers_total}",
        "source->dest (naive: every reader pulls the origin):",
    ]
    for line in render_tree(res.metrics.edges):
        lines.append("    " + line)
    return "\n".join(lines)
