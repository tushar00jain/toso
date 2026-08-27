"""The production Option B actor behind local endpoint-shaped handles."""

from __future__ import annotations

from typing import Any

from realsim.seams._plane import mount_endpoints

__all__ = ["OptionBService"]


class OptionBService:
    """Expose a production Option B actor through the local simulator."""

    def __init__(self, option_b: Any) -> None:
        self.option_b = option_b
        self.asked = mount_endpoints(self, option_b)
