"""Prose budget report: how much of a module is comments and docstrings.

The budget CLAUDE.md sets is per *unit kind*, not per file: declarations
(module docstring, abstract/Protocol member, dataclass field) may carry prose,
concrete method bodies may not. So the report is four numbers per file:

* **prose share** -- comment and docstring lines over total lines;
* **the split** -- docstring lines vs comment lines. A file whose prose is
  almost all docstring has its prose at the top of each unit rather than beside
  the line that needs it;
* **over-body** -- concrete docstrings longer than the code they head. A
  member whose body is ``...``/``pass``, or which is marked abstract, is a
  declaration and is not counted;
* **over-15** -- docstrings past the length that CLAUDE.md says needs a reason,
  concrete and declaration alike.

It reports; it does not fail. What the right share is depends on what the file
is, so the number is for a reader to judge and to diff across two runs.

Run it directly::

    python -m realsim.tools.prose_budget [path ...]

with no path for every ``control/`` and ``data/`` directory in the checkout.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path
from typing import Iterable, List, NamedTuple, Set

from realsim.tools.check_contract import REPO_ROOT

__all__ = ["main"]

#: A docstring past this many lines "needs a reason" (CLAUDE.md).
_LONG_DOCSTRING = 15

#: Directories scanned when the CLI is given no path.
_DEFAULT_DIRS = ("control", "data")


class _Doc(NamedTuple):
    """One docstring: how long it is, and how long the body under it is.

    ``body`` is ``None`` for a module or class docstring and for a declaration
    (an abstract member, or one whose body is only ``...``/``pass``) -- those are
    exempt from the over-body rule, so there is nothing to compare against.
    """

    lineno: int
    lines: int
    body: int | None


class _Report(NamedTuple):
    """One file's counts. Summing two of these gives the aggregate."""

    total: int
    docstring: int
    comment: int
    over_body: int
    over_long: int

    def __add__(self, other: "_Report") -> "_Report":
        return _Report(*(a + b for a, b in zip(self, other)))

    @property
    def prose(self) -> int:
        return self.docstring + self.comment


def _docstring_of(node) -> ast.Expr | None:
    """The node's docstring statement, or ``None``."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return first
    return None


def _is_declaration(node) -> bool:
    """True for a member that declares rather than implements.

    An ``@abstractmethod``/``@abstractproperty``, or a body that is only
    ``...``/``pass`` after the docstring -- which is what a Protocol member and
    an overload look like.
    """
    decorators = {
        getattr(d, "attr", None) or getattr(d, "id", None)
        for d in getattr(node, "decorator_list", [])
    }
    if decorators & {"abstractmethod", "abstractproperty", "overload"}:
        return True
    rest = node.body[1:] if _docstring_of(node) else node.body
    for stmt in rest:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is Ellipsis:
                continue
        return False
    return True


def _body_lines(node, code_lines: Set[int]) -> int:
    """Lines of actual code in the body, docstring and blank lines excluded."""
    rest = node.body[1:] if _docstring_of(node) else node.body
    if not rest:
        return 0
    span = range(rest[0].lineno, (rest[-1].end_lineno or rest[-1].lineno) + 1)
    return sum(1 for n in span if n in code_lines)


def _docstrings(tree: ast.Module, code_lines: Set[int]) -> List[_Doc]:
    """Every docstring in the module, in line order."""
    out: List[_Doc] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        doc = _docstring_of(node)
        if doc is None:
            continue
        lines = (doc.end_lineno or doc.lineno) - doc.lineno + 1
        concrete = isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not _is_declaration(node)
        out.append(_Doc(
            doc.lineno, lines, _body_lines(node, code_lines) if concrete else None
        ))
    return sorted(out)


def _line_kinds(text: str) -> tuple:
    """``(docstring lines, comment lines, code lines)`` as disjoint line sets.

    A line carrying both code and a trailing comment counts as a comment line,
    so the three sets partition every non-blank line and the shares sum to the
    prose share.
    """
    tree = ast.parse(text)
    doc_lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                doc_lines |= set(
                    range(node.lineno, (node.end_lineno or node.lineno) + 1)
                )
    comment_lines: Set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
    comment_lines -= doc_lines
    code_lines = {
        i for i, line in enumerate(text.splitlines(), 1)
        if line.strip() and i not in doc_lines and i not in comment_lines
    }
    return doc_lines, comment_lines, code_lines


def _measure(path: Path) -> tuple:
    """``(report, docstrings)`` for one source file."""
    text = path.read_text()
    doc_lines, comment_lines, code_lines = _line_kinds(text)
    docs = _docstrings(ast.parse(text), code_lines)
    return _Report(
        total=len(text.splitlines()),
        docstring=len(doc_lines),
        comment=len(comment_lines),
        over_body=sum(1 for d in docs if d.body is not None and d.lines > d.body),
        over_long=sum(1 for d in docs if d.lines > _LONG_DOCSTRING),
    ), docs


def _sources(paths: Iterable[Path]) -> List[Path]:
    """Every ``.py`` under the given paths, sorted, ``__pycache__`` skipped."""
    out: Set[Path] = set()
    for p in paths:
        found = [p] if p.is_file() else p.rglob("*.py")
        out |= {f for f in found if "__pycache__" not in f.parts}
    return sorted(out)


def _default_paths(root: Path) -> List[Path]:
    """Every plane directory in the checkout: ``*/control/``, ``*/data/``."""
    return sorted(
        d for pkg in sorted(root.iterdir()) if pkg.is_dir()
        for d in (pkg / name for name in _DEFAULT_DIRS)
        if (d / "__init__.py").is_file()
    )


_HEADER = f"{'file':<40} {'lines':>6} {'prose':>6} {'%':>5} " \
          f"{'doc':>6} {'cmt':>6} {'>body':>6} {'>15':>4}"


def _row(name: str, r: _Report) -> str:
    pct = 100.0 * r.prose / r.total if r.total else 0.0
    return (
        f"{name:<40} {r.total:>6} {r.prose:>6} {pct:>4.0f}% "
        f"{r.docstring:>6} {r.comment:>6} {r.over_body:>6} {r.over_long:>4}"
    )


def main(argv: List[str] | None = None) -> int:
    """CLI: print the per-file and aggregate prose budget.

    ``--detail`` also names each docstring the last two columns counted, by line.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    detail = "--detail" in args
    args = [a for a in args if a != "--detail"]
    paths = [Path(a).resolve() for a in args] or _default_paths(REPO_ROOT)
    total = _Report(0, 0, 0, 0, 0)
    print(_HEADER)
    for src in _sources(paths):
        report, docs = _measure(src)
        total = total + report
        try:
            name = str(src.relative_to(REPO_ROOT))
        except ValueError:
            name = src.name
        print(_row(name, report))
        if detail:
            for d in docs:
                over = d.body is not None and d.lines > d.body
                if over or d.lines > _LONG_DOCSTRING:
                    body = "declaration" if d.body is None else f"body {d.body}"
                    print(f"    line {d.lineno:>4}: {d.lines} lines, {body}")
    print("-" * len(_HEADER))
    print(_row("TOTAL", total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
