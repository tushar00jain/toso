"""Build the per-rank registrations TorchStore's routing plan is built from."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torchstore.routing.plan import RankRegistration
from torchstore.transport.types import TensorSlice

__all__ = ["registrations"]


def registrations(
    entries: Mapping[str, Mapping[str, Sequence[TensorSlice]]],
    element_sizes: Mapping[str, int],
    key: str = "model",
    mapping: Mapping[str, Sequence[object]] | None = None,
) -> dict[str, RankRegistration]:
    """One registration per rank, in the shape that rank's client would report.

    Volume IDs are rank names, so a simulated rank must be named for the volume
    it stores through.
    """
    return {
        rank: RankRegistration(
            key=key,
            slices={name: tuple(item) for name, item in slices.items()},
            element_sizes={name: element_sizes[name] for name in slices},
            mapping=dict(mapping or {}),
        )
        for rank, slices in entries.items()
    }
