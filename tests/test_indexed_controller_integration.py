from __future__ import annotations

import argparse
import asyncio

import pytest

from dedup_sim.control._sensor import DedupDirectorySensor
from realsim import demo
from realsim.adapters import real_controller
from realsim.adapters.real_controller import make_controller_adapter
from realsim.seams.dict_directory import DictDirectory
from realsim.tools import benchmark_dedup_control
from sim_common import config
from torchstore.controller import Controller
from torchstore.controllers.indexed import IndexedController
from torchstore.transport import Request, TensorSlice


@pytest.fixture(autouse=True)
def _restore_config():
    with config.overrides():
        yield


def _slice(
    offsets: tuple[int, int],
    local_shape: tuple[int, int],
    coordinate: int,
) -> TensorSlice:
    return TensorSlice(
        offsets=offsets,
        coordinates=(coordinate,),
        global_shape=(8, 8),
        local_shape=local_shape,
        mesh_shape=(2,),
    )


def _requests() -> tuple[list[Request], list[Request]]:
    stored = [
        Request.from_tensor_slice("model.weight", _slice((0, 0), (8, 4), 0)),
        Request.from_tensor_slice("model.weight", _slice((0, 4), (8, 4), 1)),
    ]
    wanted = [Request.from_tensor_slice("model.weight", _slice((0, 0), (4, 8), 0))]
    return stored, wanted


def _adapter(backend: str, *, real_directory: bool = True):
    with config.overrides(
        controller_backend=backend,
        real_directory=real_directory,
    ):
        return make_controller_adapter()


def _populate(adapter):
    stored, wanted = _requests()
    service = adapter.service
    service.notify_put_batch([stored[0]], "trainer-0", pending=False)
    service.notify_put_batch([stored[1]], "trainer-1", pending=False)
    pub = service.notify_put_batch(stored, "generator-pending", pending=True)
    return wanted, pub


def test_config_default_env_override_and_controller_selection(monkeypatch) -> None:
    assert config.SimConfig().controller_backend == "legacy"
    monkeypatch.delenv("TOSO_CONTROLLER_BACKEND", raising=False)
    assert config.configure().controller_backend == "legacy"
    assert isinstance(make_controller_adapter().controller, Controller)

    monkeypatch.setenv("TOSO_CONTROLLER_BACKEND", " InDeXeD ")
    assert config.configure().controller_backend == "indexed"
    assert config.configure(controller_backend="legacy").controller_backend == "legacy"

    legacy = _adapter("legacy", real_directory=False)
    assert isinstance(legacy.controller, Controller)
    assert legacy.shimmed
    assert isinstance(legacy.controller.keys_to_storage_volumes, DictDirectory)

    indexed = _adapter("indexed", real_directory=False)
    assert isinstance(indexed.controller, IndexedController)
    assert not indexed.shimmed
    assert not hasattr(indexed.controller, "keys_to_storage_volumes")


def test_invalid_controller_backend_is_rejected(monkeypatch) -> None:
    with pytest.raises(ValueError, match="controller_backend"):
        config.SimConfig(controller_backend="tree")
    with pytest.raises(ValueError, match="controller_backend"):
        config.configure(controller_backend="tree")
    with pytest.raises(ValueError, match="controller_backend"):
        with config.overrides(controller_backend="tree"):
            pass

    monkeypatch.setenv("TOSO_CONTROLLER_BACKEND", "tree")
    with pytest.raises(ValueError, match="controller_backend"):
        config.configure()


def test_shared_cli_configures_controller_backend(monkeypatch) -> None:
    parser = demo._add_run_flags(argparse.ArgumentParser())
    args = parser.parse_args(["--controller-backend", "indexed"])
    monkeypatch.setattr(demo, "configure_logging", lambda _level: None)

    demo._apply_run_flags(args)

    assert config.current().controller_backend == "indexed"


def test_benchmark_entrypoint_applies_ambient_controller_backend(monkeypatch) -> None:
    selected = []

    def capture(args) -> None:
        selected.append((config.current().controller_backend, args.case))

    monkeypatch.setenv("TOSO_CONTROLLER_BACKEND", "indexed")
    monkeypatch.setattr(benchmark_dedup_control, "_run", capture)

    assert benchmark_dedup_control.main(["--preset", "smoke"]) == 0
    assert selected == [("indexed", "smoke")]


def test_selection_calls_the_public_factory_once(monkeypatch) -> None:
    calls: list[str] = []

    def select(name: str):
        calls.append(name)
        return IndexedController

    monkeypatch.setattr(real_controller, "get_controller_class", select)
    adapter = _adapter("indexed")

    assert isinstance(adapter.controller, IndexedController)
    assert calls == ["indexed"]


def test_legacy_and_indexed_match_through_service_handle_and_sensor() -> None:
    observations = []
    for backend in ("legacy", "indexed"):
        adapter = _adapter(backend)
        wanted, pub = _populate(adapter)
        sensor = DedupDirectorySensor(adapter.service)
        serving = sensor.serving_union(wanted)
        ranked = [
            (pub, "generator-pending"),
            (0, "trainer-1"),
            (0, "trainer-0"),
        ]
        chosen = sensor.greedy_cover(wanted, ranked)
        keys = asyncio.run(adapter.handle.keys.call_one())
        located = asyncio.run(
            adapter.handle.locate_volumes.call_one(
                ["model.weight"], require_fully_committed=True
            )
        )
        observations.append((serving, chosen, keys, tuple(located["model.weight"])))

    assert observations[0] == observations[1]
    assert observations[0][0] == frozenset(
        {
            (0, "trainer-0"),
            (0, "trainer-1"),
            (1, "generator-pending"),
        }
    )
    assert observations[0][1] == [(1, "generator-pending")]
    assert observations[0][2] == ["model.weight"]
    assert set(observations[0][3]) == {"trainer-0", "trainer-1"}


@pytest.mark.parametrize("backend", ["legacy", "indexed"])
def test_live_deletion_and_pending_retirement(backend: str) -> None:
    adapter = _adapter(backend)
    request = Request.from_tensor_slice("model.weight", _slice((0, 0), (8, 4), 0))
    service = adapter.service
    service.notify_put_batch([request], "live", pending=False)
    pub = service.notify_put_batch([request], "pending", pending=True)

    assert service.serving_union([request]) == frozenset(
        {(0, "live"), (pub, "pending")}
    )
    service.notify_delete("model.weight", "live")
    assert service.locate_volumes(["model.weight"], missing_ok=True) == {}
    assert service.serving_union([request]) == frozenset({(pub, "pending")})

    service.notify_delete_batch(pub=pub)
    assert service.serving_union([request]) == frozenset()


def test_indexed_direct_inspection_is_a_snapshot() -> None:
    adapter = _adapter("indexed")
    request = Request.from_tensor_slice("model.weight", _slice((0, 0), (8, 4), 0))
    adapter.service.notify_put_batch([request], "live", pending=False)

    snapshot = adapter.controller.get_keys_to_storage_volumes()
    assert set(snapshot["model.weight"]) == {"live"}
    snapshot.clear()

    fresh = adapter.controller.get_keys_to_storage_volumes()
    assert set(fresh["model.weight"]) == {"live"}
