"""Build the per-rank registrations TorchStore's routing plan is built from."""

from __future__ import annotations

from collections.abc import Mapping

from torchstore.routing.plan import KeyRegistration
from torchstore.transport.types import TensorSlice

__all__ = ["registrations"]


def registrations(
    entries: Mapping[str, Mapping[str, TensorSlice]],
    element_sizes: Mapping[str, int],
) -> dict[str, dict[str, KeyRegistration]]:
    """Registrations per rank and key, as that rank's client would report them.

    Volume IDs are rank names, so a simulated rank must be named for the volume
    it stores through.
    """
    return {
        rank: {
            name: KeyRegistration(item, element_sizes[name])
            for name, item in slices.items()
        }
        for rank, slices in entries.items()
    }
