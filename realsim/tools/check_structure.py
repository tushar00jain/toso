"""Structure lint for the sim packages: the shape, mechanically.

:mod:`realsim.tools.check_contract` enforces what a module may *import*. This
enforces what a package *is* -- the part that had been maintained by hand across
six files per package and had already drifted (two of three READMEs stopped
naming a structural file after it was added).

Three rules, none of which a type system can express:

1. **A sim package has the same parts.** Every ``*_sim/`` carries ``__init__.py``,
   ``__main__.py``, ``README.md``, ``workload/`` and ``report/``. ``control/`` and
   ``data/`` are optional -- ``putget_sim`` deliberately has neither, which is
   what makes it the baseline -- but a package with one must have both, because
   the plane split is meaningless with only half of it.
2. **A folder-private module is underscored.** A module imported only from inside
   its own directory is private, and its name should say so. Without this the
   rule decays the moment nobody re-runs the audit by hand. :data:`PUBLIC_ANYWAY`
   is the explicit, reviewable list of exceptions.
3. **A README's layout block matches the tree.** It may not name a ``.py`` file
   that does not exist, and it may not omit one of the package's structural
   files. Prose drifts silently; this makes it fail.
4. **Every module declares its surface.** A module with public top-level
   definitions declares ``__all__``, it names only things that exist, and it
   names all of them. Before this, roughly a third of the modules had one and
   the rest did not, four of the lists were incomplete (``sim_common.cost_model``
   omitted ``ProfileTransferCost``, which ``realsim.simulation`` imports), and
   nothing said which was intended. ``__init__.py`` is exempt -- a package's
   ``__all__`` is a curated re-export list, not a mirror of its own contents --
   and so is ``__main__.py``.

Rule 4 checks that ``__all__`` is *complete*, not that each name *deserves* to be
public. That sounds like rule 2 one level down, and it was tried: flagging every
exported name with no consumer outside its own module produces ~70 hits, and most
are correct as they stand -- type aliases that appear only in this module's own
annotations (``MakePlane``, ``DecodeLoad``), rule tables that exist to be read
(``BANNED_ALWAYS``), types a caller receives without importing
(``KVView.pin`` -> ``PinnedKVView``), and exceptions a caller catches
(``StorageCapacityExceeded``). A rule whose exception list is longer than its
findings is not enforcing anything, so name-level privacy stays a review
question. Renaming to ``_thing`` is the answer when review says so.

The ``Demo`` contract is *not* checked here. ``realsim.demo.Demo`` is an ABC with
an abstract ``scenarios()`` and a required ``name``/``description``, so a demo
that does not declare its parts cannot be constructed -- and
``realsim/tests/test_demos.py`` constructs all three. A lint would only restate
what the type already refuses.

Run it directly::

    PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.check_structure

or via the test that wraps it (``realsim/tests/test_contract.py``).
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set

from realsim.tools.check_contract import (
    format_violations,
    resolve_module,
    REPO_ROOT,
    Violation,
)

__all__ = [
    "SIM_SUFFIX",
    "REQUIRED_FILES",
    "REQUIRED_DIRS",
    "PLANE_DIRS",
    "GRAPH_PKGS",
    "PUBLIC_ANYWAY",
    "sim_packages",
    "check_package_parts",
    "check_private_naming",
    "check_readme_layout",
    "public_defs",
    "declared_all",
    "check_module_exports",
    "check_all",
    "main",
]

# Packages whose shape is checked. A sim is a ``*_sim``; ``realsim`` is the
# foundation and has a different (documented) shape.
SIM_SUFFIX = "_sim"

#: What every sim package carries.
REQUIRED_FILES = ("__init__.py", "__main__.py", "README.md")
REQUIRED_DIRS = ("workload", "report")
#: All-or-neither: a capability that decides must also execute.
PLANE_DIRS = ("control", "data")

#: Packages whose modules take part in the privacy rule.
GRAPH_PKGS = (
    "dedup_sim", "domain", "kvcache_sim", "proposed", "putget_sim", "realsim",
    "sim_common",
)

#: Modules that no neighbour imports but that are public on purpose.
PUBLIC_ANYWAY: Dict[str, str] = {
    "proposed": "the whole package is the contract, surfaced via its __init__",
    "realsim/mesh.py": "named in check_contract's CONTROL_FORBIDDEN and built on "
                       "by capabilities through sim.mesh",
    "realsim/simulation.py": "the stack every capability assembles",
    "realsim/demo.py": "the Demo/Scenario contract a __main__ declares, and the "
                       "run flags every one of them shares",
    "realsim/run.py": "Workload/Run/Result/Report/execute -- the run lifecycle",
    "realsim/tools/check_contract.py": "a CLI (python -m ...)",
    "realsim/tools/check_structure.py": "a CLI (python -m ...)",
    "sim_common/engine.py": "the ancestor callback DES, kept as reference",
    "sim_common/diverge.py": "a divergence-bisection tool for debugging a run",
    "sim_common/config.py": "the ambient run config every leaf reads",
}


def sim_packages(root: Path = REPO_ROOT) -> List[Path]:
    """Every ``*_sim`` package directory, sorted."""
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.endswith(SIM_SUFFIX) and (p / "__init__.py").exists()
    )


def check_package_parts(root: Path = REPO_ROOT) -> List[Violation]:
    """Rule 1: every sim package has the same parts."""
    out: List[Violation] = []
    for pkg in sim_packages(root):
        for name in REQUIRED_FILES:
            if not (pkg / name).is_file():
                out.append(Violation(
                    f"{pkg.name}/", 0, "missing-part",
                    f"a sim package must carry {name!r}",
                ))
        for name in REQUIRED_DIRS:
            if not (pkg / name / "__init__.py").is_file():
                out.append(Violation(
                    f"{pkg.name}/", 0, "missing-part",
                    f"a sim package must carry a {name}/ package "
                    f"({name}/__init__.py not found)",
                ))
        present = [d for d in PLANE_DIRS if (pkg / d / "__init__.py").is_file()]
        if len(present) == 1:
            missing = next(d for d in PLANE_DIRS if d not in present)
            out.append(Violation(
                f"{pkg.name}/", 0, "half-a-plane-split",
                f"has {present[0]}/ but no {missing}/: the plane split means "
                f"nothing with only half of it. A capability that decides must "
                f"also execute; one that does neither has neither",
            ))
    return sorted(out)


def _import_graph(root: Path = REPO_ROOT, pkgs: Sequence[str] = GRAPH_PKGS):
    """``module path -> set of importing module paths``, re-exports resolved.

    ``pkgs`` is a parameter so a test can point the rule at a synthetic tree and
    prove it still fires -- see ``realsim/tests/test_contract.py``.
    """
    mods: Dict[str, Path] = {}
    for pkg in pkgs:
        for f in sorted((root / pkg).rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(root)
            dotted = ".".join(rel.with_suffix("").parts)
            if dotted.endswith(".__init__"):
                dotted = dotted[: -len(".__init__")]
            mods[dotted] = rel

    # A package __init__ that re-exports a name makes the defining module public.
    reexport: Dict[tuple, str] = {}
    for dotted, rel in mods.items():
        if rel.name != "__init__.py":
            continue
        for node in ast.walk(ast.parse((root / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                src = resolve_module(str(rel), node.level, node.module or "")
                for alias in node.names:
                    reexport[(dotted, alias.asname or alias.name)] = src

    importers: Dict[str, Set[str]] = defaultdict(set)
    for dotted, rel in mods.items():
        for node in ast.walk(ast.parse((root / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                base = resolve_module(str(rel), node.level, node.module or "")
                for alias in node.names:
                    target = f"{base}.{alias.name}"
                    if target in mods:                       # from pkg import mod
                        importers[target].add(dotted)
                    elif (base, alias.name) in reexport:     # re-export
                        importers[reexport[(base, alias.name)]].add(dotted)
                    elif base in mods:
                        importers[base].add(dotted)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in mods:
                        importers[alias.name].add(dotted)
    return mods, importers


def _is_test(dotted: str) -> bool:
    return "tests" in dotted.split(".") or dotted.split(".")[-1].startswith("test_")


def check_private_naming(
    root: Path = REPO_ROOT, pkgs: Sequence[str] = GRAPH_PKGS
) -> List[Violation]:
    """Rule 2: a module only its own folder imports is named ``_thing.py``."""
    mods, importers = _import_graph(root, pkgs)
    out: List[Violation] = []
    for dotted, rel in sorted(mods.items()):
        if rel.name == "__init__.py" or rel.name == "__main__.py":
            continue
        if _is_test(dotted) or rel.name.startswith("_"):
            continue
        if str(rel) in PUBLIC_ANYWAY or rel.parts[0] in PUBLIC_ANYWAY:
            continue
        imps = importers.get(dotted, set())
        if not imps:
            continue  # nothing imports it at all -- not a naming question
        outside = {
            s for s in imps
            if not _is_test(s) and Path(mods[s]).parent != rel.parent
        }
        if outside:
            continue
        out.append(Violation(
            str(rel), 0, "public-name-private-module",
            f"only {rel.parent}/ imports it, so the name should say so: "
            f"rename to _{rel.name} (or add it to PUBLIC_ANYWAY with a reason)",
        ))
    return sorted(out)


def _layout_block(text: str) -> str | None:
    """The fenced block under a ``## Layout`` / ``## Module layout`` heading."""
    m = re.search(
        r"^##+ (?:Module layout|Layout)\s*\n(.*?)```\n(.*?)```", text, re.S | re.M
    )
    return m.group(2) if m else None


def check_readme_layout(root: Path = REPO_ROOT) -> List[Violation]:
    """Rule 3: a README's layout block names real files, and all the key ones."""
    out: List[Violation] = []
    every_py = {
        f.name for f in root.rglob("*.py")
        if "__pycache__" not in f.parts and ".venv" not in f.parts
    }
    for pkg in sim_packages(root):
        readme = pkg / "README.md"
        if not readme.is_file():
            continue  # already reported by check_package_parts
        block = _layout_block(readme.read_text())
        rel_readme = f"{pkg.name}/README.md"
        if block is None:
            out.append(Violation(
                rel_readme, 0, "missing-layout",
                "no '## Layout' section with a fenced tree: the package's shape "
                "has to be written down where a reader looks for it",
            ))
            continue
        named = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*\.py)", block))
        for ghost in sorted(named - every_py):
            out.append(Violation(
                rel_readme, 0, "layout-names-missing-file",
                f"the layout block names {ghost!r}, which does not exist",
            ))
        # Structural files must appear; leaf modules are the author's call.
        structural = {"__main__.py"} | {
            f.name for d in (*REQUIRED_DIRS, *PLANE_DIRS)
            for f in (pkg / d).glob("*.py")
            if (pkg / d).is_dir() and f.name != "__init__.py"
        }
        for missing in sorted(structural - named):
            out.append(Violation(
                rel_readme, 0, "layout-omits-file",
                f"{missing!r} is part of the package but the layout block does "
                f"not name it",
            ))
    return sorted(out)


def public_defs(tree: ast.Module) -> List[str]:
    """Public top-level definitions, in definition order."""
    out: List[str] = []
    for n in tree.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not n.name.startswith("_"):
                out.append(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_") \
                        and t.id != "__all__":
                    out.append(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            if not n.target.id.startswith("_"):
                out.append(n.target.id)
    seen: Set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def declared_all(tree: ast.Module) -> List[str] | None:
    """The module's ``__all__``, or ``None`` if it declares none."""
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets
        ):
            return [e.value for e in n.value.elts]
    return None


def check_module_exports(root: Path = REPO_ROOT) -> List[Violation]:
    """Rule 4: a module's ``__all__`` exists and matches its public surface."""
    out: List[Violation] = []
    for pkg in GRAPH_PKGS:
        for f in sorted((root / pkg).rglob("*.py")):
            if "__pycache__" in f.parts or "tests" in f.parts:
                continue
            if f.name in ("__init__.py", "__main__.py"):
                continue
            rel = str(f.relative_to(root))
            tree = ast.parse(f.read_text())
            defined = public_defs(tree)
            if not defined:
                continue
            declared = declared_all(tree)
            if declared is None:
                out.append(Violation(
                    rel, 0, "missing-all",
                    f"defines {', '.join(defined)} but declares no __all__: a "
                    f"module's surface should be stated, not inferred",
                ))
                continue
            names = set(defined) | {
                a.asname or a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names
            } | {
                a.asname or a.name
                for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names
            }
            for ghost in [x for x in declared if x not in names]:
                out.append(Violation(
                    rel, 0, "all-names-nothing",
                    f"__all__ names {ghost!r}, which this module neither defines "
                    f"nor imports",
                ))
            for missing in [x for x in defined if x not in declared]:
                out.append(Violation(
                    rel, 0, "all-omits-public",
                    f"{missing!r} is public but missing from __all__ (add it, or "
                    f"rename it _{missing} if it is not part of the surface)",
                ))
    return sorted(out)


def check_all(root: Path = REPO_ROOT) -> List[Violation]:
    """Every structure rule, in one list."""
    return sorted(
        check_package_parts(root)
        + check_private_naming(root)
        + check_readme_layout(root)
        + check_module_exports(root)
    )


def main(argv: List[str] | None = None) -> int:
    """CLI: print any violations and exit non-zero if the structure has drifted."""
    violations = check_all()
    if violations:
        print(f"Structure check FAILED ({len(violations)} violation(s)):")
        print(format_violations(violations))
        return 1
    print(f"Structure check passed ({len(sim_packages())} sim packages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
