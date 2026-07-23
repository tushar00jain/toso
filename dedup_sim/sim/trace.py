"""Trace recorder + event / summary / ASCII-diagram rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .model import Region


@dataclass
class Trace:
    """Chronological record of simulated events.

    Each entry is ``(time, kind, message)``; rendering produces one line per
    event. Because the sim is deterministic, two runs produce byte-identical
    trace strings.
    """

    events: List[Tuple[float, str, str]] = field(default_factory=list)

    def record(self, now: float, kind: str, msg: str) -> None:
        """Append an event at ``now``."""
        self.events.append((now, kind, msg))

    def render_lines(self) -> List[str]:
        """Render the event trace as a list of formatted lines (one per event)."""
        return [
            f"t={t:6.3f}  {kind:<6} {msg}" for (t, kind, msg) in self.events
        ]

    def render(self) -> str:
        """Render the event trace, one line per event."""
        return "\n".join(self.render_lines())


@dataclass
class Metrics:
    """Outcome metrics for one coordinator run."""

    fabric_bytes: int = 0  # trainer -> generator bytes (the fabric cost)
    wallclock: float = 0.0
    readers_total: int = 0
    readers_done: int = 0
    edges: List[Tuple[str, str, Region]] = field(default_factory=list)  # src,dst,region
    assembled: Dict[str, Set[Region]] = field(
        default_factory=lambda: defaultdict(set)
    )
    peak_serving: int = 0


def _edge_graph(edges: List[Tuple[str, str, Region]]):
    """Return ``(children, roots, is_chain)`` derived from fetch edges.

    ``children`` maps a source to its distinct destinations (insertion order);
    ``roots`` are sources that are never a destination; ``is_chain`` is True when
    the who-served-whom graph is a simple path (every node has at most one
    child), which renders inline.
    """
    children: Dict[str, List[str]] = defaultdict(list)
    has_parent: Set[str] = set()
    for src, dst, _region in edges:
        if dst not in children[src]:
            children[src].append(dst)
        has_parent.add(dst)
    roots = sorted({s for (s, _d, _r) in edges if s not in has_parent})
    is_chain = all(len(kids) <= 1 for kids in children.values())
    return children, roots, is_chain


def render_tree(edges: List[Tuple[str, str, Region]]) -> List[str]:
    """Build an ASCII source->dest diagram from the recorded fetch edges.

    Pure chains (each node serves at most one child) render inline
    (``t0 --> g0 --> g1``); branching fan-out renders as an indented tree. When
    the graph is a DAG (a node served by more than one source, common in the
    reshard case), each node's subtree is expanded only on its first appearance;
    later references are shown as ``name (^)`` leaves so the picture stays
    compact.
    """
    children, roots, is_chain = _edge_graph(edges)
    if not roots:
        return ["(no transfers)"]

    lines: List[str] = []
    if is_chain:
        for root in roots:
            parts = [root]
            cur = root
            while children.get(cur):
                cur = children[cur][0]
                parts.append(cur)
            lines.append(" ──▶ ".join(parts))
        return lines

    expanded: Set[str] = set()

    def walk(node: str, prefix: str) -> None:
        kids = children.get(node, [])
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            branch = "└─▶ " if last else "├─▶ "
            if kid in expanded and children.get(kid):
                lines.append(prefix + branch + f"{kid} (^)")
                continue
            expanded.add(kid)
            lines.append(prefix + branch + kid)
            walk(kid, prefix + ("    " if last else "│   "))

    for root in roots:
        lines.append(root)
        expanded.add(root)
        walk(root, "")
    return lines


def render_summary(dedup: Metrics, naive: Metrics, union_bytes: int,
                   fanout_cap: int) -> str:
    """Render the summary block: fabric, wallclock and the ASCII diagram."""
    dedup_x = dedup.fabric_bytes / union_bytes if union_bytes else 0.0
    naive_x = naive.fabric_bytes / union_bytes if union_bytes else 0.0
    saved = naive.fabric_bytes - dedup.fabric_bytes
    _children, _roots, is_chain = _edge_graph(dedup.edges)
    topo = "chain" if is_chain else "tree/DAG"

    lines = [
        f"fabric(trainer->gen): dedup={dedup.fabric_bytes}B ({dedup_x:.1f}x)"
        f"   naive={naive.fabric_bytes}B ({naive_x:.1f}x)   saved {saved}B",
        f"wallclock: dedup={dedup.wallclock:.3f}  naive={naive.wallclock:.3f}"
        f"   (dedup optimizes BYTES; wallclock depends on FANOUT_CAP/topology)",
        f"source->dest (FANOUT_CAP={fanout_cap}, a {topo}):",
    ]
    for line in render_tree(dedup.edges):
        lines.append("    " + line)
    return "\n".join(lines)
