"""Structure lint for the sim packages: the shape, mechanically.

:mod:`realsim.tools.check_contract` enforces what a module may *import*. This
enforces what a package *is* -- the part that had been maintained by hand across
six files per package and had already drifted (two of three READMEs stopped
naming a structural file after it was added).

Six rules, none of which a type system can express:

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
5. **A public function has a consumer.** A public top-level function that no
   *other* module uses is rule 2 one level down, with the same remedy: name it
   ``_thing``. :data:`PUBLIC_NAMES` is the explicit, reviewable list of
   exceptions.
6. **A ``data/`` module may call a control port, never read it.** The planes run
   in different services, so what passes between them has to be something a wire
   could carry. Calling a port the module imports from its sibling ``control/`` is
   a request -- as is reading a member and invoking a send mode on it
   (``port.member.call_one(...)``), which is the shape a Monarch handle has.
   Reading a field for its value, handing a bound method out as a callback, or
   ``getattr``-ing it are not. A port is any imported class that is
   not a dataclass -- the values that legitimately cross are dataclasses, and
   reading *their* fields is the point of sending them.

Rule 4 checks that ``__all__`` is *complete*, not that each name *deserves* to be
public -- it reads "public" off the leading underscore and nothing else. So a
function could be dead, exported, and green: ``longest_prefix_run`` sat in
``kvcache_sim/control/request.py`` with no caller but a test while
``KVView.prefix_lengths`` carried its own copy of the same walk, and every rule
above passed. Rule 5 is the answer, narrowed until it was worth enforcing:

* over *all* public names it is unenforceable -- 78 hits, mostly correct as they
  stand: type aliases used only in this module's annotations (``MakePlane``,
  ``Edge``), rule tables that exist to be read (``BANNED_ALWAYS``), types a
  caller receives without importing (``KVView.pin`` -> ``PinnedKVView``),
  exceptions a caller catches (``StorageCapacityExceeded``). A rule whose
  exception list is longer than its findings enforces nothing;
* over public **functions** it is 12, because every category above is a class or
  a value, not a function. Those 12 were resolved (10 renamed, 2 in
  :data:`PUBLIC_NAMES` with reasons), so the rule now runs at zero.

Two judgements are wired into it. A **test is not a consumer**: a test importing
a name is what kept ``longest_prefix_run`` alive, so counting tests would leave
the hole open -- which means a helper written *for* tests must say so in
:data:`PUBLIC_NAMES`. And the line the exceptions draw is helper vs entry point:
a helper only its own module calls is private, while an entry point that exists
to be called from outside the repo's own graph (``run_sim``) has no in-repo
consumer by construction and is not thereby dead. Classes and module-level values
stay a review question, for the reasons listed above.

Two limits, stated rather than hidden. The rule skips the modules
:data:`PUBLIC_ANYWAY` already declares public on purpose -- these lint CLIs, the
ambient config, the run lifecycle -- where "nothing in the graph imports this" is
the documented condition, not a finding; enforcing it there costs 17 exemptions
to buy nothing. And "uses it" is resolved statically: an import of the name, an
attribute through an imported module, or a package re-export. A name reached only
by ``getattr`` would read as unused.

Rule 6 is the mechanical half of "no state object is shared across the planes",
which had been a written rule with nothing behind it: ``data/`` -> ``control/`` is
a legal import, so the contract lint saw nothing while four things crossed that a
wire could not carry -- a live ``DecodeEngine`` handed to the scheduler, a bound
``observe_compute_busy`` fired per decode step, a subscript into control's
``busy_until``, and ``getattr(scheduler, "tbt_enabled")``. The rule is
call-vs-read rather than a member whitelist, because that distinction alone
catches all four and needs no list to maintain: what the port *offers* is a type
checker's question, what the host is *allowed to touch* is this one. Values that
cross (a ``Plan``, a ``Completion``) are dataclasses, not Protocols, so reading
their fields stays legal -- which is the point of sending them.

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
    CONTROL_SEGMENT,
    DATA_SEGMENT,
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
    "PUBLIC_NAMES",
    "SEND_MODES",
    "sim_packages",
    "check_package_parts",
    "check_private_naming",
    "check_name_privacy",
    "check_plane_ports",
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

#: How a caller may reach a member of a service it holds a reference to: Monarch's
#: modes (``monarch._src.actor.endpoint.Endpoint``), mirrored by
#: :class:`realsim.seams.link.LocalEndpoint`. Rule 6 accepts a member read followed
#: by one of these, because that pair *is* the call.
SEND_MODES = frozenset({"call", "call_one", "broadcast", "choose", "stream"})

#: Public *names* nothing outside their module uses, public on purpose (rule 5).
#: Keyed ``<repo-relative module>:<name>``. The distinction being drawn is
#: helper vs entry point: a helper only its own module calls is private, but an
#: entry point whose whole purpose is out-of-band use has no in-repo caller by
#: construction and is not thereby dead.
PUBLIC_NAMES: Dict[str, str] = {
    "sim_common/async_engine.py:run_sim": "the engine-only run entry point -- a "
                                          "scenario needing no mesh goes through "
                                          "it instead of Simulation",
    "realsim/seams/factory.py:current_owner": "introspection on the process-wide "
                                              "patch: what asserts that install "
                                              "discipline held",
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


def _module_map(root: Path, pkgs: Sequence[str]) -> Dict[str, Path]:
    """``dotted module path -> repo-relative path`` for every module in ``pkgs``."""
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
    return mods


def _import_graph(root: Path = REPO_ROOT, pkgs: Sequence[str] = GRAPH_PKGS):
    """``module path -> set of importing module paths``, re-exports resolved.

    ``pkgs`` is a parameter so a test can point the rule at a synthetic tree and
    prove it still fires -- see ``realsim/tests/test_contract.py``.
    """
    mods = _module_map(root, pkgs)

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


def _name_consumers(root: Path, pkgs: Sequence[str]):
    """``(defining module, name) -> set of modules that use it``.

    Rule 2's graph is module-granular; this is the same idea one level down. A
    use is either an import of the name (``from m import go``) or an attribute
    reference through the module (``import m`` ... ``m.go()``), and a package
    ``__init__`` that re-exports a name passes its own consumers back to the
    module that defined it -- so a name published through a package counts as
    used by whoever imports it from there.
    """
    mods = _module_map(root, pkgs)
    trees = {d: ast.parse((root / r).read_text()) for d, r in mods.items()}
    consumers: Dict[tuple, Set[str]] = defaultdict(set)
    for dotted, rel in mods.items():
        modname: Dict[str, str] = {}    # local binding -> the module it names
        for node in ast.walk(trees[dotted]):
            if isinstance(node, ast.ImportFrom):
                base = resolve_module(str(rel), node.level, node.module or "")
                for alias in node.names:
                    if f"{base}.{alias.name}" in mods:      # from pkg import mod
                        modname[alias.asname or alias.name] = f"{base}.{alias.name}"
                    elif base in mods:                      # from mod import name
                        consumers[(base, alias.name)].add(dotted)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    # Only a plain (or aliased) module binding; ``import a.b``
                    # binds ``a``, whose attributes are packages, not names.
                    if alias.name in mods and (alias.asname or "." not in alias.name):
                        modname[alias.asname or alias.name] = alias.name
        for node in ast.walk(trees[dotted]):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                src = modname.get(node.value.id)
                if src is not None:
                    consumers[(src, node.attr)].add(dotted)
    for dotted, rel in mods.items():
        if rel.name != "__init__.py":
            continue
        for node in ast.walk(trees[dotted]):
            if isinstance(node, ast.ImportFrom):
                src = resolve_module(str(rel), node.level, node.module or "")
                for alias in node.names:
                    name = alias.asname or alias.name
                    consumers[(src, name)] |= consumers.get((dotted, name), set())
    return mods, trees, consumers


def _entry_points(tree: ast.Module) -> Set[str]:
    """Names referenced under ``if __name__ == "__main__":`` (a CLI's own hook)."""
    out: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            out |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return out


def check_name_privacy(
    root: Path = REPO_ROOT, pkgs: Sequence[str] = GRAPH_PKGS
) -> List[Violation]:
    """Rule 5: a public function no other module uses is named ``_thing``."""
    mods, trees, consumers = _name_consumers(root, pkgs)
    out: List[Violation] = []
    for dotted, rel in sorted(mods.items()):
        if rel.name in ("__init__.py", "__main__.py") or _is_test(dotted):
            continue
        if str(rel) in PUBLIC_ANYWAY or rel.parts[0] in PUBLIC_ANYWAY:
            continue
        entry = _entry_points(trees[dotted])
        for node in trees[dotted].body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if name.startswith("_") or name in entry:
                continue
            if f"{rel}:{name}" in PUBLIC_NAMES:
                continue
            outside = {
                c for c in consumers.get((dotted, name), set())
                if c != dotted and not _is_test(c)
            }
            if outside:
                continue
            out.append(Violation(
                str(rel), node.lineno, "public-name-no-consumer",
                f"{name!r} is public but no other module uses it: rename it "
                f"_{name} (or add {rel}:{name} to PUBLIC_NAMES with a reason). "
                f"A test importing it is not a consumer -- that is how a helper "
                f"with no callers stays alive",
            ))
    return sorted(out)


def _is_dataclass(cls: ast.ClassDef) -> bool:
    """True for ``@dataclass`` / ``@dataclass(...)`` on a class definition."""
    for dec in cls.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) else getattr(
            target, "id", None
        )
        if name == "dataclass":
            return True
    return False


def _control_ports(rel: Path, tree: ast.Module, mods: Dict[str, Path],
                   trees: Dict[str, ast.Module]) -> Set[str]:
    """Names this module imports from a sibling ``control/`` that are *ports*.

    A ``Plan``, a ``Completion`` or a ``Request`` crossing the plane boundary is a
    *value* and its fields are meant to be read; those are dataclasses. A port is
    an object living in the other plane, and in this codebase that is a plain
    class -- ``Policy`` and ``Coordinator`` both, following the same convention
    torchstore uses for ``TorchStoreStrategy``. So the discriminator is the
    dataclass decorator, not a base: it keeps holding when a port stops being a
    ``Protocol`` and becomes an ordinary base class.
    """
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        src = resolve_module(str(rel), node.level, node.module or "")
        if CONTROL_SEGMENT not in src.split(".") or src not in trees:
            continue
        defined = {
            n.name: n for n in trees[src].body if isinstance(n, ast.ClassDef)
        }
        for alias in node.names:
            cls = defined.get(alias.name)
            if cls is not None and not _is_dataclass(cls):
                out.add(alias.asname or alias.name)
    return out


def _port_names(tree: ast.Module, ports: Set[str]) -> tuple:
    """``(local names, self attributes)`` in this module bound to a port type."""
    local: Set[str] = set()
    attrs: Set[str] = set()

    def named(ann) -> bool:
        if isinstance(ann, ast.Name):
            return ann.id in ports
        return isinstance(ann, ast.Constant) and ann.value in ports

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
                if named(a.annotation):
                    local.add(a.arg)
        elif isinstance(node, ast.AnnAssign) and named(node.annotation):
            t = node.target
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                    and t.value.id == "self":
                attrs.add(t.attr)
            elif isinstance(t, ast.Name):
                local.add(t.id)
    # ``self.x = <a port parameter>`` binds the attribute too.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) \
                and node.value.id in local:
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                        and t.value.id == "self":
                    attrs.add(t.attr)
    return local, attrs


def check_plane_ports(
    root: Path = REPO_ROOT, pkgs: Sequence[str] = GRAPH_PKGS
) -> List[Violation]:
    """Rule 6: a ``data/`` module may **call** a control port, never read it."""
    mods = _module_map(root, pkgs)
    trees = {d: ast.parse((root / r).read_text()) for d, r in mods.items()}
    out: List[Violation] = []
    for dotted, rel in sorted(mods.items()):
        parts = rel.parts
        if not (parts and parts[0].endswith(SIM_SUFFIX)) or _is_test(dotted):
            continue
        if DATA_SEGMENT not in parts[:-1]:
            continue
        tree = trees[dotted]
        ports = _control_ports(rel, tree, mods, trees)
        if not ports:
            continue
        local, attrs = _port_names(tree, ports)
        if not (local or attrs):
            continue

        def is_port(node) -> bool:
            if isinstance(node, ast.Name):
                return node.id in local
            return (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in attrs
            )

        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        # ``port.member.call_one(...)`` is a call, not a field read: a handle to a
        # service offers an endpoint per member and the caller picks how to send,
        # which is Monarch's shape (see realsim.seams.link.LocalEndpoint). The
        # member read is sanctioned exactly when a send mode is invoked on it.
        endpoint_reads = {
            id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and n.attr in SEND_MODES
            and id(n) in called
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "getattr" and node.args \
                    and is_port(node.args[0]):
                out.append(Violation(
                    str(rel), node.lineno, "data-reads-control-port",
                    "getattr on a control port: the other plane's fields are not "
                    "this host's to inspect, and a default hides that it moved",
                ))
            elif isinstance(node, ast.Attribute) and is_port(node.value) \
                    and id(node) not in called \
                    and id(node) not in endpoint_reads:
                out.append(Violation(
                    str(rel), node.lineno, "data-reads-control-port",
                    f"reads {node.attr!r} off a control port instead of calling "
                    f"it or sending to it ({', '.join(sorted(SEND_MODES))}). "
                    f"Control is a different service: a field read (or a bound "
                    f"method handed out as a callback) is not something a wire can "
                    f"carry -- add it to the port as a member, or have the run "
                    f"wire the value into this plane directly",
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
        + check_name_privacy(root)
        + check_plane_ports(root)
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
