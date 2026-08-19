"""TorchStore's fetch planner, with a source choice for sliced keys.

``LocalClient._build_volume_requests`` picks one volume per key where the key is
stored whole, and picks none at all where it is stored as slices: it walks every
volume in the located map and fetches every intersecting slice, which fetches a
replicated shard once per replica. Upstream says so itself, in the ``TODO`` inside
``_expand_tensor_slices``.

:class:`GreedyClient` closes that one gap. It is upstream's method with a covered
set threaded through the sliced branch, so map order chooses for slices the way it
already chooses for whole values -- which is what lets a control plane rank sources
and hand the ranking to the store as a located map.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from torchstore.client import LocalClient
from torchstore.controller import ObjectType
from torchstore.transport import Request

__all__ = ["GreedyClient", "PLANNER", "Region", "plan", "region"]

#: A sub-request's extent: its key, and the box it names within that key's value.
#: ``None`` for a whole value, so an object, a tensor and an exactly matching slice
#: all reduce to the region the request itself denotes.
Region = Tuple[str, object]


def region(request: Request) -> Region:
    """The extent ``request`` covers, as the identity of a covered region."""
    tensor_slice = request.tensor_slice
    if tensor_slice is None:
        return request.key, None
    return request.key, (
        tuple(tensor_slice.offsets),
        tuple(tensor_slice.local_shape),
        tuple(tensor_slice.global_shape),
    )


class GreedyClient(LocalClient):
    """A client that takes each requested region from the first volume offering it."""

    def _build_volume_requests(
        self,
        requests: List[Request],
        volume_maps: Mapping[str, Mapping[str, Any]],
        transport_buffer_map: Mapping[str, Any],
    ) -> Tuple[Dict[str, List[Request]], set]:
        """Expand per-key requests into per-volume request lists.

        Copied from ``LocalClient._build_volume_requests`` -- whose body is pinned by
        ``realsim/tests/test_upstream_parity.py`` -- except the lines marked ``OURS``,
        which are the ask: in the sliced branch a volume contributes only regions no
        earlier volume already covers, and a volume that contributes nothing is not
        asked at all.

        Region identity is the sub-request's extent (:func:`region`), so two volumes
        holding the same shard collapse to one fetch and two holding misaligned
        shards both stay. The walk does not stop early when the request is fully
        covered: that needs box subtraction across misaligned grids, and the rest of
        the map costs metadata only.
        """
        volume_requests: dict[str, list[Request]] = defaultdict(list)
        whole_keys: set[str] = set()

        for request in requests:
            volume_map = volume_maps[request.key]

            use_inplace = (
                all(
                    transport_buffer_map[vid].supports_inplace_resharding
                    for vid in volume_map
                )
                and request.tensor_val is not None
                and request.tensor_val.is_contiguous()
            )

            covered: set[Region] = set()  # OURS
            for volume_id, storage_info in volume_map.items():
                if storage_info.object_type == ObjectType.OBJECT:
                    volume_requests[volume_id].append(
                        Request(key=request.key, is_object=True)
                    )
                    whole_keys.add(request.key)
                    break
                elif storage_info.object_type == ObjectType.TENSOR:
                    volume_requests[volume_id].append(request)
                    whole_keys.add(request.key)
                    break
                else:
                    parts = self._expand_tensor_slices(
                        request, storage_info, use_inplace
                    )
                    fresh = [part for part in parts if region(part) not in covered]  # OURS
                    if not fresh:  # OURS
                        continue  # OURS
                    covered.update(region(part) for part in fresh)  # OURS
                    volume_requests[volume_id].extend(fresh)

        return dict(volume_requests), whole_keys


class _NoInplace:
    """A transport stand-in for planning with no destination tensor to write into."""

    supports_inplace_resharding = False


class _NoInplaceMap(dict):
    """Answers for any volume, so no transport map is built per plan.

    ``__missing__`` does not insert, so the one shared instance below is read-only
    however many decisions are in flight.
    """

    _stand_in = _NoInplace()

    def __missing__(self, volume: str) -> _NoInplace:
        return self._stand_in


_NO_INPLACE_MAP = _NoInplaceMap()

#: The planner a dry run drives. Never bound to a controller or a strategy, because
#: ``_build_volume_requests`` reads neither and writes nothing -- it is a pure
#: function of the requests and the located map, so one instance serves every
#: concurrent decision.
PLANNER: GreedyClient = GreedyClient.__new__(GreedyClient)


def plan(
    requests: Sequence[Request],
    volume_maps: Mapping[str, Any],
) -> Dict[str, List[Request]]:
    """``volume -> the sub-requests it is asked for``, planned greedily.

    Requests naming a key ``volume_maps`` does not hold are dropped, so a caller may
    pass a batch wider than the directory answered for.
    """
    served = [request for request in requests if request.key in volume_maps]
    if not served:
        return {}
    planned, _whole = PLANNER._build_volume_requests(
        served, volume_maps, _NO_INPLACE_MAP
    )
    return planned
