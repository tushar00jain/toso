"""The capability-free fixture: seed one key, then ``m`` clients get it.

The smallest scenario that exercises everything ``realsim`` provides -- the real
``LocalClient`` planning core, the real ``Controller`` directory, the real
``InMemoryStore`` behind the volume seam, the cost model, and the deterministic
virtual-clock engine -- and **nothing** a capability owns. It exists so
``realsim``'s tests have something to run that decides nothing, and so
``dedup_sim`` has a baseline that is the same workload as its routed run.

Layout: one *origin* volume on node ``P`` holds ``W``; ``m`` reader volumes on
distinct hosts of node ``R`` each want it. Every reader is released at the same
instant and simply calls ``client.get(W)``, so with no routing they all locate the
origin before anyone finishes and each pulls from it -- fabric is ``m x`` the
payload.

The scenario is ordinary user code, top to bottom: a ``client.put`` and a gather
of ``client.get``. There is no policy, no coordinator and no execution loop in
it. Handing it a :class:`~proposed.policy.Policy` (and, if the capability needs
one, a :class:`~proposed.plane.DataPlane`) is the *only* change needed to make it
a routed run -- which is exactly how ``dedup_sim`` turns the same ``m x`` burst
into a 1x one.

The whole run (compute + seed put + the gets) executes on one
:class:`~sim_common.async_engine.AsyncEngine`, so it produces one continuous
virtual-time trace that is byte-identical across runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from proposed import DataPlane, Deployment, Policy
from realsim.entrypoint import Result, Workload
from realsim.runner import WorkItem
from realsim.simulation import Simulation
from realsim.seams.transport import Endpoint, TensorDescriptor
from sim_common.cost_model import DEFAULT_PROFILE, MachineProfile, compute_time

KEY = "W"
DEFAULT_N = 16  # elements in W (float32 -> 4 bytes each)

# TODO(next diff): derive this payload from a ``domain.llm.Model`` instead of a
# bare element count -- see the TODO in ``domain/llm.py`` for the full plan.

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

# A factory for the capability's data plane, called with the wired mesh plus the
# key/payload the readers move: ``(mesh, key, value) -> DataPlane``.
MakePlane = Callable[[Deployment, str, Any], DataPlane]


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


def _topology(num_readers: int) -> Dict[str, Endpoint]:
    """Origin volume ``p`` on node P; readers ``r0..`` on distinct hosts of node R."""
    topo: Dict[str, Endpoint] = {
        "p": Endpoint(id="volp", host="hP", node="P"),
    }
    for i in range(num_readers):
        topo[f"r{i}"] = Endpoint(id=f"volr{i}", host=f"hR{i}", node="R")
    return topo


@dataclass
class BurstResult(Result):
    """A :class:`~realsim.entrypoint.Result` plus what the burst itself reports.

    ``results``/``trace``/``ledger``/``sim`` come from the base; the rest are facts
    about the scenario that a summary needs and the ledger does not carry.
    """

    # The ground-truth W carrier (meta tensor or descriptor); both expose
    # ``shape``/``dtype``/``numel()``/``element_size()`` so callers can size it.
    expected: Any
    origin_id: str
    num_readers: int

    @property
    def payload_bytes(self) -> int:
        """Bytes of one W -- the 1x union a routed run drives fabric toward."""
        return self.expected.numel() * self.expected.element_size()


class PutGetBurst(Workload):
    """m readers get one key an origin already holds.

    The capability-free fixture: ordinary user code (a put, then a gather of gets)
    with no routing of its own. Installing a :class:`~proposed.policy.Policy` and a
    :class:`~proposed.plane.DataPlane` turns it into a routed run without touching
    a line of it, which is how ``dedup_sim`` compares the two.

    Args:
        num_readers: how many readers contend for the key.
        n: elements in ``W``; the payload is ``n * itemsize`` bytes.
        dtype: element type of ``W``.
        mode: ``"meta"`` (a zero-storage real tensor, the default) or
            ``"metadata"`` (a ``(shape, dtype)`` descriptor, no tensor at all).
            Both are allocation-free and drive the real store round-trip.
        device: device for the ``"meta"`` carrier.
        profile: target-machine :class:`~sim_common.cost_model.MachineProfile`,
            used here for the producer's generate step; the stack charges every
            other cost from the same one.
        compute_device: roofline device for that generate step.
        make_plane: builds the capability's data plane once the deployment and the
            payload exist. ``None`` -> no plane, the unrouted baseline.
    """

    def __init__(
        self,
        num_readers: int = 3,
        *,
        n: int = DEFAULT_N,
        dtype: torch.dtype = torch.float32,
        mode: str = MODE_META,
        device: str = "meta",
        profile: Optional[MachineProfile] = None,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        make_plane: Optional[MakePlane] = None,
    ) -> None:
        if mode not in (MODE_META, MODE_METADATA):
            raise ValueError(f"unknown data-plane mode {mode!r}")
        self.num_readers = num_readers
        self.dtype = dtype
        self.mode = mode
        self.profile = profile if profile is not None else DEFAULT_PROFILE
        self.compute_device = compute_device
        self.make_plane = make_plane

        self.topology = _topology(num_readers)
        self.origin_id = self.topology["p"].id  # holds W before the burst
        self.reader_ids = [f"r{i}" for i in range(num_readers)]

        # W, as an allocation-free carrier either way.
        self.descriptor = TensorDescriptor(shape=(n,), dtype=dtype)
        if mode == MODE_META:
            # Real tensor, zero storage (device="meta").
            self.expected: Any = torch.empty(n, dtype=dtype, device=device)
        else:
            # No tensor at all -- the descriptor *is* the payload.
            #
            # put_batch value-typing gotcha (torchstore/client.py): put_batch types
            # the value -- a Tensor/DTensor takes the tensor path, everything else
            # (including ``None``) takes ``Request.from_objects`` and is stored as
            # an OBJECT. Rather than fight that, we lean into it and hand the
            # descriptor itself to ``client.put``, so it round-trips through the
            # real object path and the reader gets the descriptor back. The only
            # seam change is that ``_nbytes`` reads the modeled size off the
            # descriptor when ``tensor_val is None``.
            self.expected = self.descriptor
        self.put_value = self.expected

    def build(self, sim: Simulation) -> Tuple[Optional[DataPlane], List[WorkItem]]:
        """One work item per reader, plus the capability's plane if it has one."""
        mesh, trace = sim.mesh, sim.trace
        # Bytes served by the origin are the fabric cost a routing policy exists
        # to cut; everything else is a peer-to-peer hop.
        sim.origins(self.origin_id)

        def _get(reader_id: str) -> Callable[[], Any]:
            """One reader's ordinary user code: bind who I am, then get the key."""

            async def call() -> Any:
                mesh.bind_source(reader_id)
                result = await mesh.client(reader_id).get(KEY)
                trace.record(
                    asyncio.get_running_loop().time(),
                    "burst",
                    f"reader {reader_id} done",
                )
                return result

            return call

        plane = (
            self.make_plane(mesh, KEY, self.put_value)
            if self.make_plane is not None
            else None
        )
        items = [
            WorkItem(id=rid, release_time=0.0, run=_get(rid))
            for rid in self.reader_ids
        ]
        return plane, items

    def result(self, result: Result) -> BurstResult:
        """Add what a burst summary needs beyond the ledger."""
        return BurstResult(
            results=result.results,
            trace=result.trace,
            ledger=result.ledger,
            sim=result.sim,
            expected=self.expected,
            origin_id=self.origin_id,
            num_readers=self.num_readers,
        )

    async def setup(self, sim: Simulation) -> None:
        """Generate W on the producer's accelerator, then seed it on the origin."""
        mesh, trace = sim.mesh, sim.trace
        # The producer writes W to its co-located origin volume. This is a
        # single-client drive (before the burst), so it uses the adapter's own
        # factory rather than the mesh's shared one.
        producer = mesh.adapter("p")
        # (1) COMPUTE/GPU: modeled as FLOPS_PER_ELEMENT per element, streaming the
        # whole payload's bytes; the roofline picks the binding term on
        # ``compute_device``. Charged on the virtual clock before the put.
        gen_nbytes = self.descriptor.nbytes
        gen_flops = FLOPS_PER_ELEMENT * self.descriptor.numel()
        gen_dt = compute_time(
            gen_flops,
            _dtype_name(self.dtype),
            self.compute_device,
            self.profile,
            gen_nbytes,
        )
        await asyncio.sleep(gen_dt)
        trace.record(
            asyncio.get_running_loop().time(),
            "compute",
            f"generate {KEY} flops={gen_flops:g} {gen_nbytes}B "
            f"dev={self.compute_device} cost={gen_dt:.4f}",
        )
        # (2) Seed W on the origin volume (also populates the real directory). The
        # put path charges network + storage-write in the transport seam.
        with producer.installed():
            await producer.client.put(KEY, self.put_value)
        # (3) Every reader then gets W. Each get charges storage-read + RAM
        # staging + network in the transport seam.
        trace.record(
            asyncio.get_running_loop().time(),
            "burst",
            f"{self.num_readers} readers get {KEY!r}",
        )
