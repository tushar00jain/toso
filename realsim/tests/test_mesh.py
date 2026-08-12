"""The shared multi-client wiring primitive: :class:`realsim.mesh.Mesh`.

``Mesh`` owns the pieces every multi-node scenario needs before it can express a
capability -- real volumes, one real ``LocalClient`` per node, one directory, one
resource registry, and the single ``create_transport_buffer`` substitution. These
tests cover the two things a consumer relies on and one hazard the consolidation
closes:

1. a multi-client drive over one mesh charges each operation the *calling*
   client's source locality (the contextvar resolution the process-wide factory
   global forces on us);
2. :attr:`Mesh.on_transfer` is read at call time, so a consumer constructed after
   the mesh (a ledger's transfer accounting) can claim it; and
3. overlapping installs **raise** instead of silently shadowing each other.

(3) is the hazard: ``create_transport_buffer`` is a process-wide module global, so
when two owners were active at once the inner install won, the outer owner's
source binding was never read, and transfers were charged the wrong locality while
the outer owner kept recording metrics as if it were still driving -- a silent
mispricing, not a crash.
"""

from __future__ import annotations

import pytest
import torch

from realsim.mesh import Mesh
from realsim.seams import factory
from realsim.seams.transport import Endpoint
from sim_common.async_engine import run_sim
from sim_common.trace import Trace


def _topology() -> dict[str, Endpoint]:
    """Three nodes: a and b share a node (NVLink), c is remote (cross-node)."""
    return {
        "a": Endpoint(id="vola", host="hostA", node="node0"),
        "b": Endpoint(id="volb", host="hostB", node="node0"),
        "c": Endpoint(id="volc", host="hostC", node="node1"),
    }


def test_mesh_wires_one_client_and_volume_per_node_on_one_directory():
    """Every node gets a real client + volume, all sharing one directory/registry."""
    topo = _topology()
    mesh = Mesh(topo)

    assert mesh.ids == ["a", "b", "c"]
    assert sorted(mesh.volumes) == ["a", "b", "c"]
    assert sorted(mesh.adapters) == ["a", "b", "c"]
    # Distinct clients, one shared controller directory behind them all.
    clients = [mesh.client(v) for v in mesh.ids]
    assert len({id(c) for c in clients}) == 3
    assert all(a.client is not None for a in mesh.adapters.values())
    # The volume handle's id is its DIRECTORY identity -- the node key, which is
    # what the co-located client registers puts under and therefore the only name
    # a key this volume drops can be reported to the directory under. (The
    # endpoint's ``.id``, "vola", is the separate transfer identity.)
    assert mesh.volumes["a"].volume_id == "a"
    assert mesh.topology["a"].id == "vola"
    # One registry shared by every adapter (so concurrent transfers can contend).
    assert mesh.registry is not None


def test_multi_client_drive_charges_the_calling_client_locality():
    """One shared factory, per-operation source: b's get is NVLink, c's is RDMA.

    Both readers pull the same key from the same volume over the *same* installed
    factory. The only thing that distinguishes their cost is the source endpoint
    each bound, which is exactly what the contextvar exists to resolve.
    """
    topo = _topology()
    trace = Trace()
    mesh = Mesh(topo, trace=trace)

    async def scenario():
        with mesh.installed():
            # a writes W to its own volume.
            mesh.bind_source("a")
            await mesh.client("a").put("W", torch.arange(64, dtype=torch.float32))
            # b (same node as a) and c (remote) each read it.
            mesh.bind_source("b")
            await mesh.client("b").get("W")
            mesh.bind_source("c")
            await mesh.client("c").get("W")
        return True

    ok, trace = run_sim(scenario(), trace=trace)
    assert ok

    gets = [e[2] for e in trace.events if e[1] == "xfer" and e[2].startswith("get")]
    # Each reader's transfer is attributed to its own endpoint as the destination,
    # and the serving volume (vola) as the source.
    assert any("vola->volb" in g for g in gets)
    assert any("vola->volc" in g for g in gets)
    # b is on a's node, c is not, so c's read must cost strictly more than b's.
    # "xfer" rows are the fabric rows (storage/RAM staging use "store"/"mem").
    def _cost(marker: str) -> float:
        row = next(g for g in gets if marker in g)
        return float(row.rsplit("cost=", 1)[1].split()[0])

    assert _cost("vola->volc") > _cost("vola->volb") > 0.0


def test_on_transfer_is_read_at_call_time():
    """A hook attached after construction still sees every transfer.

    A consumer needs the mesh to exist before it can be built, so it claims the
    hook post-construction; the factory must therefore not capture it eagerly.
    """
    topo = _topology()
    mesh = Mesh(topo)
    seen: list[tuple[str, str, str, int]] = []

    # Attached *after* Mesh.__init__ built the factory closure.
    mesh.on_transfer = lambda kind, src, dst, nbytes, cost: seen.append(
        (kind, src, dst, nbytes)
    )

    async def scenario():
        with mesh.installed():
            mesh.bind_source("a")
            await mesh.client("a").put("W", torch.arange(16, dtype=torch.float32))
            mesh.bind_source("c")
            await mesh.client("c").get("W")
        return True

    ok, _ = run_sim(scenario())
    assert ok
    assert [s for s in seen if s[0] == "put"], "put was not reported"
    remote_gets = [s for s in seen if s[0] == "get" and s[1] == "vola" and s[2] == "volc"]
    assert remote_gets and all(nbytes > 0 for *_, nbytes in remote_gets)


def test_overlapping_installs_raise_instead_of_shadowing():
    """Two owners cannot hold the process-wide factory patch at once.

    This is the hazard the single substitution point closes: previously the inner
    install silently won and the outer owner's source binding was ignored.
    """
    mesh_one = Mesh(_topology())
    mesh_two = Mesh(_topology())

    with mesh_one.installed():
        with pytest.raises(RuntimeError, match="already substituted"):
            with mesh_two.installed():
                pass  # pragma: no cover - the install must raise
        # The first owner still holds the patch, undisturbed by the refused install.
        assert factory.current_owner() is mesh_one

    # And it is released on exit.
    assert factory.current_owner() is None


def test_install_is_released_after_an_exception():
    """An error inside the block must not leave the global patched."""
    mesh = Mesh(_topology())
    original = factory._CLIENT_MODULE.create_transport_buffer

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with mesh.installed():
            raise _Boom

    assert factory.current_owner() is None
    assert factory._CLIENT_MODULE.create_transport_buffer is original


def test_single_client_adapter_installs_are_sequential_not_nested():
    """The adapter's single-client install obeys the same one-owner rule.

    ``RealClientAdapter.installed()`` pins the source to its own node, so two of
    them overlapping would mean one client's transfers priced as another's.
    """
    mesh = Mesh(_topology())

    # Sequential is fine.
    for vid in ("a", "b"):
        with mesh.adapter(vid).installed():
            assert factory.current_owner() is mesh.adapter(vid)
    assert factory.current_owner() is None

    # Nested is refused.
    with mesh.adapter("a").installed():
        with pytest.raises(RuntimeError, match="already substituted"):
            with mesh.adapter("b").installed():
                pass  # pragma: no cover - the install must raise


def test_current_source_without_a_binding_raises():
    """An unbound source is a caller bug, not something to default away.

    A default would silently misprice every transfer, so the contextvar lookup is
    allowed to fail loudly instead.
    """

    async def scenario():
        with pytest.raises(LookupError):
            factory.current_source()
        return True

    ok, _ = run_sim(scenario())
    assert ok
