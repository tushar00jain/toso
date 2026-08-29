"""Build the per-rank registrations TorchStore's routing plan is built from."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torchstore.routing.plan import KeyRegistration
from torchstore.transport.types import TensorSlice

__all__ = ["registrations"]


def registrations(
    entries: Mapping[str, Mapping[str, Sequence[TensorSlice]]],
    element_sizes: Mapping[str, int],
    paths: Mapping[str, Sequence[object]] | None = None,
) -> dict[str, dict[str, KeyRegistration]]:
    """Registrations per rank and key, as that rank's client would report them.

    Volume IDs are rank names, so a simulated rank must be named for the volume
    it stores through. Pass ``paths`` only for a publisher that stores a
    mapping object; without it nothing routes ``{namespace}/MAPPING``.
    """
    return {
        rank: {
            name: KeyRegistration(
                tuple(item), element_sizes[name], (paths or {}).get(name)
            )
            for name, item in slices.items()
        }
        for rank, slices in entries.items()
    }
