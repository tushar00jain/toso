"""Request region identity."""

from __future__ import annotations

from typing import Tuple

from torchstore.transport import Request

__all__ = ["Region", "region"]

Region = Tuple[str, object]


def region(request: Request) -> Region:
    """The extent covered by one request."""
    tensor_slice = request.tensor_slice
    if tensor_slice is None:
        return request.key, None
    return request.key, (
        tensor_slice.offsets,
        tensor_slice.local_shape,
        tensor_slice.global_shape,
    )
