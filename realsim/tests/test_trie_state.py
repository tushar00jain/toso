"""Live and pending entries sharing the controller trie."""

from __future__ import annotations

from dedup_sim.control._sensor import FanoutSensor, Routed
from dedup_sim.control.routing import _gate_publications
from proposed import Dispatcher
from realsim.adapters.real_controller import RealControllerAdapter
from torchstore import coverage
from torchstore.transport import Request, TensorSlice


def _tensor(key: str) -> Request:
    return Request.from_any(key, None).meta_only()


def _slice(key: str, coordinate: int) -> Request:
    return Request.from_tensor_slice(
        key,
        TensorSlice(
            (coordinate * 4,),
            (coordinate,),
            (12,),
            (4,),
            (3,),
        ),
    ).meta_only()


def _span(key: str, shards: int = 2) -> Request:
    return Request.from_tensor_slice(
        key,
        TensorSlice((0,), (0,), (12,), (shards * 4,), (3,)),
    ).meta_only()


def _service():
    return RealControllerAdapter().service


def _slot(service, key: str = "K", volume: str = "v0"):
    return service.controller.keys_to_storage_volumes[key][volume]


def test_mixed_live_and_pending_share_one_slot_but_locate_projects_live():
    service = _service()
    service.notify_put_batch([_tensor("K")], "v0", pending=False)
    pub = service.notify_put_batch([_tensor("K")], "v0")

    located = service._locate(["K"])

    assert set(located["K"]) == {"v0"}
    assert located["K"]["v0"] is _slot(service)[0]
    assert service.serving_union([_tensor("K")]) == {
        (0, "v0"),
        (pub, "v0"),
    }
    assert set(_slot(service)) == {0, pub}


def test_slice_disjoint_live_and_pending_sources_form_the_greedy_cover():
    service = _service()
    service.notify_put_batch([_slice("K", 0)], "v0", pending=False)
    pub = service.notify_put_batch([_slice("K", 1)], "v0")
    requests = [_span("K")]

    serving = service.serving_union(requests)
    chosen = service.greedy_cover(requests, ((0, "v0"), (pub, "v0")))

    assert serving == {(0, "v0"), (pub, "v0")}
    assert chosen == [(0, "v0"), (pub, "v0")]
    assert _gate_publications(chosen) == {(pub, "v0")}


def test_two_pending_publications_share_one_slot_and_retire_independently():
    service = _service()
    first = service.notify_put_batch([_tensor("K")], "v0")
    second = service.notify_put_batch([_tensor("K")], "v0")
    fanout = FanoutSensor()
    dispatcher = Dispatcher()
    dispatcher.compose(fanout)
    dispatcher.dispatch_sync(
        Routed((first, "v0"), ("origin",), frozenset(), 1.0)
    )
    dispatcher.dispatch_sync(
        Routed((second, "v0"), ("origin",), frozenset(), 2.0)
    )

    assert service.serving_union([_tensor("K")]) == {
        (first, "v0"),
        (second, "v0"),
    }
    assert fanout.arrival((first, "v0")) == 1.0
    assert fanout.arrival((second, "v0")) == 2.0

    service.notify_delete_batch(pub=first)

    assert set(_slot(service)) == {second}
    assert service.serving_union([_tensor("K")]) == {(second, "v0")}


def test_landing_then_retiring_one_pub_keeps_the_other_pending_entry():
    service = _service()
    landed = service.notify_put_batch([_slice("K", 0)], "v0")
    waiting = service.notify_put_batch([_slice("K", 1)], "v0")

    service.notify_put_batch([_slice("K", 0)], "v0", pending=False)
    service.notify_delete_batch(pub=landed)

    assert set(_slot(service)) == {0, waiting}
    assert _slot(service)[0].tensor_slices == {_slice("K", 0).tensor_slice}
    assert _slot(service)[waiting].tensor_slices == {_slice("K", 1).tensor_slice}


def test_retirement_clears_exactly_the_selected_publication():
    service = _service()
    first = service.notify_put_batch([_tensor("K")], "v0")
    second = service.notify_put_batch([_tensor("K")], "v0")

    service.notify_delete_batch(pub=first)

    assert set(_slot(service)) == {second}
    assert first not in service.controller._publications
    assert second in service.controller._publications


def test_deleting_a_live_key_keeps_its_pending_publication():
    service = _service()
    service.notify_put_batch([_tensor("K")], "v0", pending=False)
    pub = service.notify_put_batch([_tensor("K")], "v0")

    service.notify_delete("K", "v0")

    assert set(_slot(service)) == {pub}
    assert service._locate(["K"], missing_ok=True) == {}
    assert service.serving_union([_tensor("K")]) == {(pub, "v0")}


def test_removing_the_last_slot_drops_its_volume_and_trie_key():
    service = _service()
    service.notify_put_batch([_tensor("K")], "live", pending=False)
    pub = service.notify_put_batch([_tensor("K")], "pending")

    service.notify_delete_batch(pub=pub)

    assert set(service.controller.keys_to_storage_volumes["K"]) == {"live"}
    assert set(_slot(service, volume="live")) == {0}

    service.notify_delete("K", "live")

    assert "K" not in service.controller.keys_to_storage_volumes


def test_shape_bucket_tracks_declaration_and_retirement():
    service = _service()
    pub = service.notify_put_batch([_tensor("K")], "v0")
    shape = service.controller._publications[pub].shape

    assert pub in service.controller._shape_pubs[shape]

    service.notify_delete_batch(pub=pub)

    assert shape not in service.controller._shape_pubs


def test_dtensor_serving_union_filters_each_nonmatching_shape():
    service = _service()
    service.notify_put_batch([_slice("K", 0)], "v0", pending=False)
    overlapping = service.notify_put_batch([_slice("K", 1)], "v0")
    disjoint = service.notify_put_batch([_slice("K", 2)], "v0")

    serving = service.serving_union([_span("K")])

    assert serving == {(0, "v0"), (overlapping, "v0")}
    assert (disjoint, "v0") not in serving


def test_matching_shape_bucket_skips_per_publication_overlap_checks(monkeypatch):
    service = _service()
    publications = {
        service.notify_put_batch([_slice("K", 0)], f"v{index}"): f"v{index}"
        for index in range(8)
    }
    calls = 0
    overlaps = coverage._overlaps

    def counted(wanted, offered):
        nonlocal calls
        calls += 1
        return overlaps(wanted, offered)

    monkeypatch.setattr(coverage, "_overlaps", counted)

    serving = service.serving_union([_slice("K", 0)])

    assert serving == set(publications.items())
    assert calls == 0


class _NoScanDirectory(dict):
    def __init__(self, rows):
        super().__init__(rows)
        self.get_calls = 0

    def __iter__(self):
        raise AssertionError("publication retirement scanned the directory")

    def get(self, key, default=None):
        self.get_calls += 1
        return super().get(key, default)


def test_retirement_touches_only_the_publications_own_keys():
    service = _service()
    for index in range(500):
        service.notify_put_batch([_tensor(f"other.{index}")], "live", pending=False)
    directory = _NoScanDirectory(
        service.controller.keys_to_storage_volumes.items()
    )
    service.controller.keys_to_storage_volumes = directory
    requests = [_tensor("small.0"), _tensor("small.1")]
    pub = service.notify_put_batch(requests, "pending")
    directory.get_calls = 0

    service.notify_delete_batch(pub=pub)

    assert directory.get_calls == 2 * len(requests)
    assert all(request.key not in directory for request in requests)
    assert len(directory) == 500
