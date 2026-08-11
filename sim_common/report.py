"""Small helpers for rendering a simulation's console report.

Keeps the logging setup and section-header formatting in one place so a demo
entrypoint can route its digest through the ``logging`` module and print
banner-delimited sections consistently.

This module is also the **canonical home** of

* the shared source->dest fetch-graph renderers, :func:`edge_graph` and
  :func:`render_tree`. Both ``dedup_sim`` and ``realsim`` record who-served-whom
  as ``(src, dst, label)`` edges and want the same ASCII picture -- reuse these
  (they are label-agnostic) rather than re-implementing a tree renderer per sim;
* :class:`Ledger`, the measurement half every capability needs: the transfer
  edges + byte counters a run accumulates, one outcome row per work item, and
  the handful of aggregations (sum / mean / percentile / fraction) that every
  report computes over those rows.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Edge",
    "configure_logging",
    "section",
    "edge_graph",
    "render_tree",
    "Outcome",
    "Ledger",
    "percentile",
]

# A fetch edge: (source id, destination id, label). The label is opaque to the
# renderer (a region for dedup_sim, a key/slice for realsim).
Edge = Tuple[str, str, object]


def configure_logging(level: int) -> None:
    """Route log records to stdout as bare messages.

    ``force=True`` resets any root handlers a dependency may have installed at
    import time; otherwise ``basicConfig`` would no-op and the output would be
    silently dropped. The report is the product, so it goes to stdout.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def section(logger: logging.Logger, title: str) -> None:
    """Log a banner-delimited section header at INFO."""
    logger.info("\n%s", "=" * 72)
    logger.info(title)
    logger.info("=" * 72)


# --------------------------------------------------------------------------- #
# Source->dest fetch-graph rendering (shared across sims).
#
# Both dedup_sim and realsim record who-served-whom as a list of
# ``(src, dst, label)`` edges and want the same ASCII picture. The renderer is
# label-agnostic, so it lives here rather than in any one sim.
# --------------------------------------------------------------------------- #

def edge_graph(edges: List[Edge]):
    """Return ``(children, roots, is_chain)`` derived from fetch edges.

    ``children`` maps a source to its distinct destinations (insertion order);
    ``roots`` are sources that are never a destination; ``is_chain`` is True when
    the who-served-whom graph is a simple path (every node has at most one
    child), which renders inline.
    """
    children: Dict[str, List[str]] = defaultdict(list)
    has_parent: Set[str] = set()
    for src, dst, _label in edges:
        if dst not in children[src]:
            children[src].append(dst)
        has_parent.add(dst)
    roots = sorted({s for (s, _d, _l) in edges if s not in has_parent})
    is_chain = all(len(kids) <= 1 for kids in children.values())
    return children, roots, is_chain


def render_tree(edges: List[Edge]) -> List[str]:
    """Build an ASCII source->dest diagram from the recorded fetch edges.

    Pure chains (each node serves at most one child) render inline
    (``t0 ──▶ g0 ──▶ g1``); branching fan-out renders as an indented tree. When
    the graph is a DAG (a node served by more than one source), each node's
    subtree is expanded only on its first appearance; later references are shown
    as ``name (^)`` leaves so the picture stays compact.
    """
    children, roots, is_chain = edge_graph(edges)
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


# --------------------------------------------------------------------------- #
# Ledger: the measurement half of a run.
#
# Every capability accumulates the same three things -- who served whom, how
# many bytes crossed which link, and one outcome row per work item -- and then
# reduces the rows with the same four aggregations. That was written twice (a
# burst's fabric accounting and the KV-cache request table); it lives here once.
# --------------------------------------------------------------------------- #


@dataclass
class Outcome:
    """The default outcome row: one work item, released and finished.

    A capability with a richer per-item outcome (``kvcache_sim``'s
    ``RequestResult``) puts its own dataclass in :attr:`Ledger.rows` instead --
    the aggregation helpers read attributes by name, so they work on any row
    type.
    """

    id: str
    released: float = 0.0
    done: float = 0.0


@dataclass
class Ledger:
    """Transfer edges + byte counters + outcome rows for one run.

    Byte accounting mirrors what a fabric-reduction capability is measured on:

    * :attr:`transfer_bytes` -- every byte delivered by a ``get``;
    * :attr:`origin_bytes` -- the subset served by a volume in :attr:`origins`,
      i.e. the bytes that had to cross from a pre-existing source. With no
      routing the two are equal (``m x`` for ``m`` readers of one key); a
      dedup/cache-aware policy drives ``origin_bytes`` toward the 1x union while
      ``transfer_bytes`` stays ``m x``. (The name is deliberately literal: a
      capability whose "fabric bytes" mean something else -- ``kvcache_sim``
      sums a per-request planned transfer -- names its own reading of the rows.)

    :attr:`edges` are ``(src, dst, label)`` triples for :func:`render_tree`.
    """

    rows: List[Any] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    transfer_bytes: int = 0
    origin_bytes: int = 0
    wallclock: float = 0.0
    items_total: int = 0
    # Volume/endpoint ids that held the data before the run: a transfer sourced
    # at one of these is a fabric byte.
    origins: Set[str] = field(default_factory=set)

    # -- recording ---------------------------------------------------------- #
    def add(self, row: Any) -> None:
        """Append one outcome row."""
        self.rows.append(row)

    def record_transfer(
        self, kind: str, src_id: str, dst_id: str, nbytes: int, cost: float
    ) -> None:
        """Account one transport transfer (the ``Mesh.on_transfer`` signature).

        Only ``get`` transfers are counted: a ``put`` is the writer storing its
        own data, not a byte someone had to be served.
        """
        if kind != "get":
            return
        self.transfer_bytes += nbytes
        if src_id in self.origins:
            self.origin_bytes += nbytes
        if nbytes > 0:
            self.edges.append((src_id, dst_id, dst_id))

    # -- aggregation over rows ---------------------------------------------- #
    @property
    def items_done(self) -> int:
        """Work items that produced an outcome row."""
        return len(self.rows)

    def select(self, where: Optional[Callable[[Any], bool]] = None) -> List[Any]:
        """Rows matching ``where`` (all rows when omitted)."""
        return self.rows if where is None else [r for r in self.rows if where(r)]

    def total(
        self, attr: str, where: Optional[Callable[[Any], bool]] = None
    ) -> Any:
        """Sum of ``attr`` over the selected rows (``0`` when there are none)."""
        return sum(getattr(r, attr) for r in self.select(where))

    def mean(
        self, attr: str, where: Optional[Callable[[Any], bool]] = None
    ) -> float:
        """Mean of ``attr`` over the selected rows (``0.0`` when there are none)."""
        rows = self.select(where)
        return sum(getattr(r, attr) for r in rows) / len(rows) if rows else 0.0

    def percentile(
        self, attr: str, pct: float, where: Optional[Callable[[Any], bool]] = None
    ) -> float:
        """The ``pct`` percentile of ``attr`` over the selected rows.

        Nearest-rank on the sorted values (index ``pct/100 * n``, clamped to the
        last element), which is what the KV-cache report has always used; ``0.0``
        when there are no rows.
        """
        return percentile([getattr(r, attr) for r in self.select(where)], pct)

    def count(self, where: Callable[[Any], bool]) -> int:
        """How many rows match ``where``."""
        return sum(1 for r in self.rows if where(r))

    def fraction(
        self,
        where: Callable[[Any], bool],
        over: Optional[Callable[[Any], bool]] = None,
        *,
        empty: float = 1.0,
    ) -> float:
        """Fraction of the ``over`` rows that match ``where`` (``empty`` if none)."""
        rows = self.select(over)
        if not rows:
            return empty
        return sum(1 for r in rows if where(r)) / len(rows)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank ``pct`` percentile of ``values`` (``0.0`` when empty)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, int(pct / 100.0 * len(ordered)))
    return ordered[idx]
