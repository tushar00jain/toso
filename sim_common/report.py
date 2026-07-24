"""Small helpers for rendering a simulation's console report.

Keeps the logging setup and section-header formatting in one place so a demo
entrypoint can route its digest through the ``logging`` module and print
banner-delimited sections consistently.

This module is also the **canonical home** of the shared source->dest fetch-graph
renderers, :func:`edge_graph` and :func:`render_tree`. Both ``dedup_sim`` and
``realsim`` record who-served-whom as ``(src, dst, label)`` edges and want the
same ASCII picture -- reuse these (they are label-agnostic) rather than
re-implementing a tree renderer per sim.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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
