"""Concurrency-contract lint for the deterministic simulation code paths.

The ``realsim`` fidelity story rests on a single guarantee: every simulated code
path is single-threaded and drives time through the loop's virtual clock, so the
same inputs always produce a byte-identical trace. This module is a small,
dependency-free AST checker that *enforces* that guarantee -- it fails if the
simulated code paths reach for a primitive that would break determinism:

* ``threading`` / ``_thread`` / ``multiprocessing`` imports or use,
* ``os.fork`` / ``os.forkpty`` (process forking),
* ``time.sleep`` (a blocking wall-clock sleep),
* wall-clock *reads* used in control flow -- ``time.time`` / ``perf_counter`` /
  ``monotonic`` (and their ``*_ns`` forms), and
* unseeded randomness -- any module-global ``random.<fn>()`` call, an unseeded
  ``random.Random()``, or ``random.SystemRandom``.

**What is deliberately allowed:**

* ``asyncio.sleep`` -- the sanctioned way to advance time. It schedules against
  the running loop's clock, which is *virtual* (and free) under
  ``sim_common.async_engine.AsyncEngine`` and only a tiny real nap under a plain
  asyncio loop. It never reads the wall clock directly.
* ``random.Random(seed)`` with an explicit seed argument -- deterministic.
* Wall-clock *reads* inside test modules (``tests/`` or ``test_*.py``): those
  measure elapsed wall time only to *assert* that virtual time advanced for free
  (e.g. that ``await asyncio.sleep(10)`` cost ~0s). They are assertions about the
  engine, never control flow in a simulated path, so they are permitted in tests
  but still banned in library code.

Scope is intentionally narrow: ``realsim/`` and ``sim_common/`` only. The sibling
``../torchstore`` is out of scope and is *not* scanned -- it owns one benign
wall-clock read (``torchstore/logging.py::LatencyTracker`` uses
``perf_counter()`` for DEBUG-only elapsed display; it never affects control flow
or the ``Trace``).

Run it directly::

    PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.check_contract

or via the test that wraps it (``realsim/tests/test_contract.py``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple

# --------------------------------------------------------------------------- #
# What we scan.
# --------------------------------------------------------------------------- #

# Repo root is two levels up from this file (realsim/tools/check_contract.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("realsim", "sim_common")

# --------------------------------------------------------------------------- #
# Banned canonical references.
# --------------------------------------------------------------------------- #

# Importing any of these modules in a simulated path is a violation on its own.
BANNED_IMPORT_MODULES = frozenset({"threading", "_thread", "multiprocessing"})

# Fully-qualified callables banned everywhere in scope (library and tests).
BANNED_ALWAYS: Dict[str, str] = {
    "os.fork": "os.fork spawns a second process (nondeterministic, multi-process)",
    "os.forkpty": "os.forkpty forks a process (nondeterministic, multi-process)",
    "time.sleep": "time.sleep blocks on the wall clock; use asyncio.sleep on the "
    "loop's virtual clock instead",
}

# Wall-clock *reads*: banned in library code (control-flow / timing on wall time),
# allowed in test modules (which measure elapsed time only to assert the virtual
# clock is free).
WALLCLOCK_READS: Dict[str, str] = {
    "time.time": "reads the wall clock",
    "time.time_ns": "reads the wall clock",
    "time.perf_counter": "reads the wall clock",
    "time.perf_counter_ns": "reads the wall clock",
    "time.monotonic": "reads the wall clock",
    "time.monotonic_ns": "reads the wall clock",
    "time.process_time": "reads the wall clock",
}


class Violation(NamedTuple):
    """One contract breach: where it is, a short code, and why it matters."""

    path: str          # repo-relative path
    lineno: int
    code: str          # short machine-ish tag, e.g. "threading-import"
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.path}:{self.lineno}: [{self.code}] {self.message}"


def is_test_file(path: Path) -> bool:
    """True for test modules (under a ``tests`` dir or named ``test_*.py``)."""
    return "tests" in path.parts or path.name.startswith("test_")


class _ContractVisitor(ast.NodeVisitor):
    """Resolve import aliases, then flag banned references node-by-node."""

    def __init__(self, rel_path: str, allow_wallclock_reads: bool) -> None:
        self.rel_path = rel_path
        self.allow_wallclock_reads = allow_wallclock_reads
        self.violations: List[Violation] = []
        # name-in-this-module -> canonical module ("t" -> "time")
        self._module_alias: Dict[str, str] = {}
        # name-in-this-module -> canonical callable ("fork" -> "os.fork")
        self._callable_alias: Dict[str, str] = {}

    # -- import bookkeeping -------------------------------------------------- #

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            bound = alias.asname or top
            if top in BANNED_IMPORT_MODULES:
                self._add(node.lineno, f"{top}-import",
                          f"imports {alias.name!r} (threads/processes are banned "
                          f"on the deterministic sim path)")
            self._module_alias[bound] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top = module.split(".")[0]
        if top in BANNED_IMPORT_MODULES:
            self._add(node.lineno, f"{top}-import",
                      f"imports from {module!r} (threads/processes are banned on "
                      f"the deterministic sim path)")
        for alias in node.names:
            bound = alias.asname or alias.name
            self._callable_alias[bound] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # -- usage checks ------------------------------------------------------- #

    def _canonical_attr(self, node: ast.Attribute) -> str | None:
        """Resolve ``base.attr`` to ``realmodule.attr`` via the alias map."""
        if isinstance(node.value, ast.Name):
            base = self._module_alias.get(node.value.id, node.value.id)
            return f"{base}.{node.attr}"
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        canonical = self._canonical_attr(node)
        if canonical is not None:
            self._check_canonical(canonical, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # random.* needs call-site context (seeded vs not), so handle it here.
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base = self._module_alias.get(func.value.id, func.value.id)
            if base == "random":
                self._check_random_call(func.attr, node, func.lineno)
        # Direct-imported banned callables: ``from os import fork; fork()``.
        if isinstance(func, ast.Name):
            canonical = self._callable_alias.get(func.id)
            if canonical is not None:
                self._check_canonical(canonical, node.lineno)
        self.generic_visit(node)

    def _check_random_call(self, attr: str, call: ast.Call, lineno: int) -> None:
        if attr == "Random":
            if not call.args and not call.keywords:
                self._add(lineno, "unseeded-random",
                          "random.Random() with no seed is nondeterministic; "
                          "pass an explicit seed")
            return  # random.Random(seed) is fine
        if attr == "SystemRandom":
            self._add(lineno, "unseeded-random",
                      "random.SystemRandom draws from OS entropy (unseedable)")
            return
        # Any other random.<fn>() call uses the process-global RNG state.
        self._add(lineno, "unseeded-random",
                  f"random.{attr}() uses the unseeded module-global RNG; construct "
                  f"a seeded random.Random(seed) instead")

    def _check_canonical(self, canonical: str, lineno: int) -> None:
        if canonical in BANNED_ALWAYS:
            self._add(lineno, "wallclock-sleep" if canonical == "time.sleep"
                      else "fork", BANNED_ALWAYS[canonical])
        elif canonical in WALLCLOCK_READS and not self.allow_wallclock_reads:
            self._add(lineno, "wallclock-read",
                      f"{canonical} {WALLCLOCK_READS[canonical]}; time must flow "
                      f"through the loop's virtual clock")

    def _add(self, lineno: int, code: str, message: str) -> None:
        self.violations.append(Violation(self.rel_path, lineno, code, message))


def scan_source(source: str, rel_path: str, *, is_test: bool) -> List[Violation]:
    """Scan a single source string; return violations (sorted by line)."""
    tree = ast.parse(source, filename=rel_path)
    visitor = _ContractVisitor(rel_path, allow_wallclock_reads=is_test)
    visitor.visit(tree)
    # Dedupe (a Name can be visited twice) and sort deterministically.
    seen = set()
    out: List[Violation] = []
    for v in visitor.violations:
        key = (v.path, v.lineno, v.code)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return sorted(out)


def scan_paths(paths: Iterable[Path], *, root: Path = REPO_ROOT) -> List[Violation]:
    """Scan the given files/dirs (recursively) for contract violations."""
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.suffix == ".py":
            files.append(p)
    violations: List[Violation] = []
    for f in files:
        try:
            rel = str(f.resolve().relative_to(root))
        except ValueError:
            rel = str(f)
        violations.extend(
            scan_source(f.read_text(), rel, is_test=is_test_file(f))
        )
    return sorted(violations)


def scan_default() -> List[Violation]:
    """Scan the in-scope package dirs (``realsim/`` + ``sim_common/``)."""
    return scan_paths([REPO_ROOT / d for d in SCAN_DIRS])


def format_violations(violations: Iterable[Violation]) -> str:
    """Render violations one per line for a test failure message or the CLI."""
    return "\n".join(str(v) for v in violations)


def main(argv: List[str] | None = None) -> int:
    """CLI: print any violations and exit non-zero if the contract is broken."""
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(a) for a in argv] if argv else [REPO_ROOT / d for d in SCAN_DIRS]
    violations = scan_paths(paths)
    scanned = ", ".join(str(p) for p in paths)
    if violations:
        print(f"Concurrency-contract check FAILED ({len(violations)} violation(s)):")
        print(format_violations(violations))
        return 1
    print(f"Concurrency-contract check passed (scanned: {scanned}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
