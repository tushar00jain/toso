"""Tests for the opt-in dict-shim controller directory (Task B).

The shim swaps only ``Controller.keys_to_storage_volumes`` (a ``Trie``) for a
:class:`~realsim.seams.dict_directory.DictDirectory`, leaving every bit of the
real ``Controller`` decision logic in place. These tests pin three things:

1. **Surface parity** -- the shim handle answers ``locate_volumes`` /
   ``notify_put_batch`` / ``keys`` / ``notify_delete`` / ``notify_delete_batch``
   with the same shapes and semantics (including ``missing_ok`` and
   ``require_fully_committed`` on a partially committed DTensor) as the real
   :class:`~realsim.seams.controller_handle.LocalControllerHandle`.
2. **Divergence gate** -- a small real burst produces a byte-identical trace and
   identical payoff metrics under real vs shim mode.
3. **Selection** -- the ``real_directory`` flag defaults to real, and an explicit
   override (arg or ambient config) selects the shim.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest realsim/tests/test_shim_directory.py -q
"""

from __future__ import annotations

import asyncio

from sim_common import config

from realsim.adapters.real_controller import (
    make_controller_adapter,
    RealControllerAdapter,
)
from realsim.tests._burst import run_burst
from realsim.seams.dict_directory import DictDirectory
from torchstore.transport.types import Request, TensorSlice


# --------------------------------------------------------------------------
# DictDirectory: the Trie surface the Controller relies on.
# --------------------------------------------------------------------------


def test_dict_directory_prefix_matches_trie_semantics():
    d = DictDirectory()
    for key in ("a", "a.b", "a.b.c", "a.bc", "ab", "b.a"):
        d[key] = {}
    kv = d.keys()
    # Token-wise prefix (separator "."), exactly like Trie.filter_by_prefix.
    assert set(kv.filter_by_prefix("a")) == {"a", "a.b", "a.b.c", "a.bc"}
    assert set(kv.filter_by_prefix("a.b")) == {"a.b", "a.b.c"}
    assert set(kv.filter_by_prefix("a.bc")) == {"a.bc"}
    # A prefix no key extends yields [] (Trie swallows pygtrie's KeyError).
    assert kv.filter_by_prefix("zzz") == []
    # Mutable-mapping surface the Controller uses.
    assert "a.b" in d and "nope" not in d
    del d["a.b"]
    assert "a.b" not in d and set(kv.filter_by_prefix("a.b")) == {"a.b.c"}


# --------------------------------------------------------------------------
# Handle surface parity: shim vs the real LocalControllerHandle.
# --------------------------------------------------------------------------


def _object_request(key: str) -> Request:
    """A metadata-only OBJECT put request for ``key``."""
    return Request(key=key, is_object=True)


def _shard_request(key: str, coord: int, mesh: int) -> Request:
    """A TENSOR_SLICE put request: shard ``coord`` of a ``mesh``-way row split."""
    return Request(
        key=key,
        tensor_slice=TensorSlice(
            offsets=(coord, 0),
            coordinates=(coord,),
            global_shape=(mesh, 4),
            local_shape=(1, 4),
            mesh_shape=(mesh,),
        ),
    )


async def _drive_surface(handle):
    """Exercise the full handle surface; return a comparable snapshot dict."""
    # put an object on volume A and one DTensor shard (of 2) on volume A.
    await handle.notify_put_batch.call(
        [_object_request("obj")], "A", pending=False
    )
    await handle.notify_put_batch.call(
        [_shard_request("dt", 0, 2)], "A", pending=False
    )

    snap = {}
    # keys() + prefix filter.
    snap["keys"] = sorted(await handle.keys.call_one())
    # locate an object (fully committed by definition).
    obj_loc = await handle.locate_volumes.call_one(["obj"])
    snap["obj_volumes"] = sorted(obj_loc["obj"])
    # missing_ok: a missing key is omitted, not raised.
    snap["missing_ok"] = await handle.locate_volumes.call_one(["ghost"], missing_ok=True)
    # missing key without missing_ok raises KeyError.
    try:
        await handle.locate_volumes.call_one(["ghost"])
        snap["missing_raises"] = False
    except KeyError:
        snap["missing_raises"] = True
    # require_fully_committed: the DTensor has 1 of 2 shards -> rejected.
    try:
        await handle.locate_volumes.call_one(["dt"])
        snap["partial_raises"] = False
    except KeyError:
        snap["partial_raises"] = True
    # ... but require_fully_committed=False returns the partial entry.
    partial = await handle.locate_volumes.call_one(["dt"], require_fully_committed=False)
    snap["partial_volumes"] = sorted(partial["dt"])
    # complete the DTensor (second shard on volume B) -> now fully committed.
    await handle.notify_put_batch.call(
        [_shard_request("dt", 1, 2)], "B", pending=False
    )
    full = await handle.locate_volumes.call_one(["dt"])
    snap["dt_volumes"] = sorted(full["dt"])
    # delete one volume's shard, then batch-delete the rest.
    await handle.notify_delete.call("dt", "A")
    snap["after_delete"] = sorted((await handle.locate_volumes.call_one(
        ["dt"], require_fully_committed=False))["dt"])
    await handle.notify_delete_batch.call({"B": ["dt"], "A": ["obj"]})
    snap["final_keys"] = sorted(await handle.keys.call_one())
    return snap


def test_shim_handle_surface_matches_real():
    real = asyncio.run(_drive_surface(RealControllerAdapter().handle))
    shim = asyncio.run(_drive_surface(RealControllerAdapter(shim=True).handle))
    assert shim == real
    # sanity on the shared expectations (guards against both being wrong together)
    assert real["missing_raises"] is True
    assert real["partial_raises"] is True
    assert real["obj_volumes"] == ["A"]
    assert real["dt_volumes"] == ["A", "B"]
    assert real["after_delete"] == ["B"]
    assert real["final_keys"] == []


def test_shim_directory_is_a_dict_backing():
    real = RealControllerAdapter()
    shim = RealControllerAdapter(shim=True)
    # Both are the real Controller; only the directory container differs.
    assert type(real.controller).__name__ == "Controller"
    assert type(shim.controller).__name__ == "Controller"
    assert isinstance(shim.controller.keys_to_storage_volumes, DictDirectory)
    assert not isinstance(real.controller.keys_to_storage_volumes, DictDirectory)


# --------------------------------------------------------------------------
# Divergence gate: real vs shim produce byte-identical trace + metrics.
# --------------------------------------------------------------------------


def _burst_snapshot(res):
    """The payoff metrics that must be byte-identical across backings."""
    led = res.ledger
    return (
        res.trace.render(),
        led.origin_bytes,
        led.transfer_bytes,
        led.wallclock,
        led.items_done,
        led.items_total,
        sorted(led.edges),
    )


def test_real_vs_shim_burst_is_byte_identical():
    real = run_burst(num_readers=3, real_directory=True)
    shim = run_burst(num_readers=3, real_directory=False)
    assert _burst_snapshot(shim) == _burst_snapshot(real)


# --------------------------------------------------------------------------
# Selection: default is real; the flag / ambient config selects the shim.
# --------------------------------------------------------------------------


def _reset() -> None:
    import os

    os.environ.pop("TOSO_REAL_DIRECTORY", None)
    config.configure()


def test_flag_defaults_to_real():
    _reset()
    try:
        assert config.current().real_directory is True
        assert make_controller_adapter().shimmed is False
    finally:
        _reset()


def test_explicit_arg_selects_shim():
    _reset()
    try:
        assert make_controller_adapter(False).shimmed is True
        assert make_controller_adapter(True).shimmed is False
    finally:
        _reset()


def test_ambient_config_selects_shim():
    _reset()
    try:
        with config.overrides(real_directory=False):
            assert make_controller_adapter().shimmed is True
        # explicit arg still wins over the ambient flag
        with config.overrides(real_directory=False):
            assert make_controller_adapter(True).shimmed is False
        assert make_controller_adapter().shimmed is False
    finally:
        _reset()
