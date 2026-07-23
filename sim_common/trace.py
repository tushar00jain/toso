"""Generic chronological event recorder for discrete-event simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Trace:
    """Chronological record of simulated events (one line per event).

    Each entry is ``(time, kind, message)``; rendering produces one formatted
    line per event. Because a discrete-event sim is deterministic, two runs
    produce byte-identical trace strings.

    ``time_width`` and ``kind_width`` control the column widths used when
    rendering, so callers can tune the layout without changing behavior.
    """

    events: List[Tuple[float, str, str]] = field(default_factory=list)
    time_width: int = 6
    kind_width: int = 6

    def record(self, now: float, kind: str, msg: str) -> None:
        """Append an event at ``now``."""
        self.events.append((now, kind, msg))

    def render_lines(self) -> List[str]:
        """Render the event trace as a list of formatted lines (one per event)."""
        return [
            f"t={t:{self.time_width}.3f}  {kind:<{self.kind_width}} {msg}"
            for (t, kind, msg) in self.events
        ]

    def render(self) -> str:
        """Render the event trace, one line per event."""
        return "\n".join(self.render_lines())
