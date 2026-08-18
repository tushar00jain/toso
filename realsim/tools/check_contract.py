"""Contract lint for the deterministic simulation code paths.

Two contracts, one AST walk.

1. Concurrency
--------------
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

2. Plane separation
-------------------
A capability folder splits into ``control/`` (decides) and ``data/`` (executes).
Renaming folders does not hold on its own -- the previous split leaked precisely
because a scheduler held the objects it was not supposed to drive, so it simply
drove them. The rule that makes the leak impossible is an import-direction one:

    ``*/control/`` may not import ``*/data/``, the mesh, or a store client.

Control receives an environment and sensors and returns a decision; anything
that moves bytes reaches it as an *observation*, never as a handle. That is
mechanically checkable, so it is checked here rather than left to review.
Importing in the other direction is fine and expected: the data plane is handed
the decisions.

There is a second direction, for the same reason one level out:

    ``*/control/`` and ``*/data/`` may not import ``*/workload/``.

``control/`` and ``data/`` are the halves that would ship -- ordinary application
code against a ``Deployment``. ``workload/`` is the run's scaffolding: what to
simulate, which comparisons to draw, how to narrate them. It has no counterpart
in production, so anything the shipping halves need from it is misfiled. This
caught ``Request`` living in ``kvcache_sim/workload/`` while the scheduler, the
serving plane and the decode engine all imported it; it belongs in ``control/``
with the ``Plan`` and ``Completion`` it is reasoned about alongside.

Scope: the simulation packages -- ``realsim/``, ``sim_common/``, ``domain/``, and
the capability packages ``putget_sim/`` / ``dedup_sim/`` / ``kvcache_sim/`` (whose
``control/`` and ``data/`` folders the plane rules apply to; ``putget_sim`` has
neither, because it decides nothing and executes nothing). The sibling ``../torchstore`` is out of scope
and is *not* scanned -- it owns one benign wall-clock read
(``torchstore/logging.py::LatencyTracker`` uses ``perf_counter()`` for DEBUG-only
elapsed display; it never affects control flow or the ``Trace``).

Run it directly::

    PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.check_contract

or via the test that wraps it (``realsim/tests/test_contract.py``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple

__all__ = [
    "REPO_ROOT",
    "SCAN_DIRS",
    "BANNED_IMPORT_MODULES",
    "BANNED_ALWAYS",
    "WALLCLOCK_READS",
    "CONTROL_FORBIDDEN",
    "DATA_FORBIDDEN",
    "CONTROL_FORBIDDEN_NAMES",
    "DATA_SEGMENT",
    "CONTROL_SEGMENT",
    "WORKLOAD_SEGMENT",
    "PROPOSED_PKG",
    "PROPOSED_FORBIDDEN",
    "Violation",
    "is_test_file",
    "is_control_module",
    "is_capability_data_module",
    "is_proposed_module",
    "resolve_module",
    "scan_source",
    "scan_paths",
    "scan_default",
    "format_violations",
    "main",
]

# --------------------------------------------------------------------------- #
# What we scan.
# --------------------------------------------------------------------------- #

# Repo root is two levels up from this file (realsim/tools/check_contract.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = (
    "dedup_sim", "domain", "kvcache_sim", "proposed", "putget_sim", "realsim",
    "sim_common",
)

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


# --------------------------------------------------------------------------- #
# Plane separation: what a ``control/`` module may not import.
# --------------------------------------------------------------------------- #

# Matched against the *resolved* dotted module path (relative imports included).
# A module is banned if it equals one of these or starts with it plus a dot.
CONTROL_FORBIDDEN: Dict[str, str] = {
    "torchstore.client": "a real store client (control decides; it never calls the store)",
    "torchstore.api": "the store API (control returns decisions; data executes them)",
    "torchstore.storage_volume": "a storage volume (control never moves stored data)",
    "realsim.mesh": "the mesh (control gets an environment and sensors)",
    "realsim.adapters": "a real client/controller adapter",
    "realsim.seams": "the store seams (transport, volumes, controller handle)",
    "proposed.plane": "the DataPlane interface (that is the executing half)",
    "realsim.runner": "the Runner (releasing work is execution, not decision)",
    "sim_common": "simulation internals -- take machine facts from domain",
}

# What a capability's ``data/`` may not import. It is application code: it calls
# ordinary torchstore APIs against a proposed.deployment.Deployment, and the
# harness that *is* that deployment under simulation is wired up in workload/.
DATA_FORBIDDEN: Dict[str, str] = {
    "realsim": "the simulator (data/ is real code -- take a Deployment instead)",
    "sim_common": "simulation internals (machine facts come from domain)",
}

# Re-exports mean a module-path ban is not enough: ``proposed`` deliberately
# surfaces its whole contract at package level, so ``from proposed import X`` has
# to be judged on X. Names control may not pull out of it, whatever the path.
CONTROL_FORBIDDEN_NAMES: Dict[str, str] = {
    "DataPlane": "the DataPlane interface (that is the executing half)",
    "Deployment": "a Deployment (control never reaches the store)",
}

# Any resolved module with this path segment is a capability's data plane.
DATA_SEGMENT = "data"
# ...and this one marks the importing module as control.
CONTROL_SEGMENT = "control"
# The run's scaffolding: real code -- control/ and data/ -- may not import it,
# because production has no workload package to import.
WORKLOAD_SEGMENT = "workload"

# The ``proposed`` package is the surface argued for upstream, so it may use
# torchstore but must remain independent of simulator and capability packages.
PROPOSED_PKG = "proposed"
PROPOSED_FORBIDDEN: Dict[str, str] = {
    "realsim": "simulator scaffolding (the proposal has to stand without it)",
    "sim_common": "simulation primitives (the proposal depends on nothing here)",
    "dedup_sim": "a capability (the proposal must not know its consumers)",
    "kvcache_sim": "a capability (the proposal must not know its consumers)",
    "putget_sim": "a capability (the proposal must not know its consumers)",
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


def is_control_module(rel_path: str) -> bool:
    """True for a capability's control-plane module (``<pkg>/control/...``)."""
    parts = Path(rel_path).parts
    return CONTROL_SEGMENT in parts[:-1]


def is_capability_data_module(rel_path: str) -> bool:
    """True for a capability's data-plane module (``<pkg>/data/...``)."""
    parts = Path(rel_path).parts
    return len(parts) > 1 and parts[0].endswith("_sim") and DATA_SEGMENT in parts[:-1]


def is_proposed_module(rel_path: str) -> bool:
    """True for a module in the upstream-proposal package (``proposed/...``)."""
    parts = Path(rel_path).parts
    return bool(parts) and parts[0] == PROPOSED_PKG


def resolve_module(rel_path: str, level: int, module: str) -> str:
    """Resolve an import to an absolute dotted path.

    ``level`` is ``ast.ImportFrom.level`` (0 = absolute, 1 = ``.``, 2 = ``..``).
    A relative import is resolved against the importing file's own package, taken
    from its repo-relative path, so ``kvcache_sim/control/scheduler.py`` doing
    ``from ..data.store import KVStore`` resolves to ``kvcache_sim.data.store``.
    """
    if level == 0:
        return module
    package = list(Path(rel_path).parts[:-1])   # drop the file name
    if level > 1:
        package = package[: len(package) - (level - 1)]
    return ".".join([*package, module] if module else package)


class _ContractVisitor(ast.NodeVisitor):
    """Resolve import aliases, then flag banned references node-by-node."""

    def __init__(
        self,
        rel_path: str,
        allow_wallclock_reads: bool,
        is_control: bool = False,
        is_proposed: bool = False,
        is_capability_data: bool = False,
    ) -> None:
        self.rel_path = rel_path
        self.allow_wallclock_reads = allow_wallclock_reads
        # Control-plane modules additionally may not import the executing half.
        self.is_control = is_control
        self.is_proposed = is_proposed
        self.is_capability_data = is_capability_data
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
            self._check_plane_import(alias.name, node.lineno)
            self._module_alias[bound] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top = module.split(".")[0]
        if top in BANNED_IMPORT_MODULES:
            self._add(node.lineno, f"{top}-import",
                      f"imports from {module!r} (threads/processes are banned on "
                      f"the deterministic sim path)")
        resolved = resolve_module(self.rel_path, node.level, module)
        self._check_plane_import(resolved, node.lineno)
        if self.is_control:
            for alias in node.names:
                why = CONTROL_FORBIDDEN_NAMES.get(alias.name)
                if why is not None:
                    self._add(node.lineno, "control-imports-execution",
                              f"control imports {alias.name!r} from {resolved!r}: "
                              f"that is {why}")
        for alias in node.names:
            bound = alias.asname or alias.name
            self._callable_alias[bound] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # -- plane separation ---------------------------------------------------- #

    def _check_plane_import(self, module: str, lineno: int) -> None:
        """Flag a ``control/`` module importing the executing half.

        Also flags a ``proposed/`` module reaching into simulator-only packages.
        """
        if not module:
            return
        if self.is_proposed:
            for banned, why in PROPOSED_FORBIDDEN.items():
                if module == banned or module.startswith(banned + "."):
                    self._add(lineno, "proposed-imports-simulator",
                              f"proposed imports {module!r}: that is {why}")
                    return
            return
        if self.is_capability_data:
            for banned, why in DATA_FORBIDDEN.items():
                if module == banned or module.startswith(banned + "."):
                    self._add(lineno, "data-imports-simulator",
                              f"data imports {module!r}: that is {why}")
                    return
            if WORKLOAD_SEGMENT in module.split("."):
                self._add(lineno, "plane-imports-workload",
                          f"data imports {module!r}: workload/ is the run's "
                          f"scaffolding and has no counterpart in production, so "
                          f"a shipping plane cannot need anything from it. Move "
                          f"the shared type down to control/")
            return
        if not self.is_control:
            return
        parts = module.split(".")
        if WORKLOAD_SEGMENT in parts:
            self._add(lineno, "plane-imports-workload",
                      f"control imports {module!r}: workload/ is the run's "
                      f"scaffolding and has no counterpart in production, so a "
                      f"shipping plane cannot need anything from it. Define the "
                      f"shared type in control/ instead")
            return
        if DATA_SEGMENT in parts:
            self._add(lineno, "control-imports-data",
                      f"control imports {module!r}: a control-plane module may "
                      f"not reach into a data plane. Take the decision out to the "
                      f"caller, or accept the result back as an observation")
            return
        for banned, why in CONTROL_FORBIDDEN.items():
            if module == banned or module.startswith(banned + "."):
                self._add(lineno, "control-imports-execution",
                          f"control imports {module!r}: that is {why}")
                return

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
    visitor = _ContractVisitor(
        rel_path,
        allow_wallclock_reads=is_test,
        is_control=is_control_module(rel_path),
        is_proposed=is_proposed_module(rel_path),
        is_capability_data=is_capability_data_module(rel_path),
    )
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
    """Scan every in-scope package dir (see :data:`SCAN_DIRS`)."""
    return scan_paths([REPO_ROOT / d for d in SCAN_DIRS])


def format_violations(violations: Iterable[Violation]) -> str:
    """Render violations one per line for a test failure message or the CLI."""
    return "\n".join(str(v) for v in violations)


def main(argv: List[str] | None = None) -> int:
    """CLI: print any violations and exit non-zero if a contract is broken."""
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(a) for a in argv] if argv else [REPO_ROOT / d for d in SCAN_DIRS]
    violations = scan_paths(paths)
    scanned = ", ".join(str(p) for p in paths)
    if violations:
        print(f"Contract check FAILED ({len(violations)} violation(s)):")
        print(format_violations(violations))
        return 1
    print(f"Contract check passed (scanned: {scanned}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
