"""The contract lint, wired into the test suite.

``test_sim_paths_obey_the_concurrency_contract`` fails the build if any simulated
code path in the scanned packages reaches for a determinism-breaking primitive
(threads, forks, wall-clock sleeps/reads, unseeded randomness), **or** if a
capability's ``control/`` module imports the executing half (a ``data/`` package,
the mesh, or a store client). The companion tests prove the checker actually
detects each banned pattern and does not flag the sanctioned ones -- so a green
run means the contract holds, not that the lint is asleep.

``test_sim_packages_keep_their_shape`` does the same for the *structure* lint:
the parts every ``*_sim`` carries, the underscore on a folder-private module and
on a public function nothing outside its module uses, a README layout block that
matches the tree, and an ``__all__`` on every module that matches its public
surface.

See ``realsim/tools/check_contract.py`` and ``check_structure.py`` for the full
contracts and their rationale.
"""

from __future__ import annotations

from realsim.tools.check_contract import (
    format_violations,
    is_control_module,
    resolve_module,
    scan_default,
    scan_source,
)
from realsim.tools import check_structure


def _codes(source: str, *, is_test: bool = False, path: str = "snippet.py"):
    return {v.code for v in scan_source(source, path, is_test=is_test)}


def _parse(source: str):
    import ast

    return ast.parse(source)


CONTROL = "kvcache_sim/control/scheduler.py"
DATA = "kvcache_sim/data/serving.py"
PROPOSED = "proposed/policy.py"


def test_sim_paths_obey_the_concurrency_contract():
    """The real tree must be clean (this is the enforcing check)."""
    violations = scan_default()
    assert not violations, (
        "concurrency-contract violations on the sim path:\n"
        + format_violations(violations)
    )


# --------------------------------------------------------------------------
# Structure: the shape of a sim package.
# --------------------------------------------------------------------------


def test_sim_packages_keep_their_shape():
    """The real tree must be clean (this is the enforcing check)."""
    violations = check_structure.check_all()
    assert not violations, (
        "structure violations in the sim packages:\n"
        + format_violations(violations)
    )


def test_structure_lint_finds_all_three_sims():
    """A rule that silently scanned nothing would pass forever."""
    names = [p.name for p in check_structure.sim_packages()]
    assert names == ["dedup_sim", "kvcache_sim", "putget_sim"]


def test_structure_lint_flags_a_missing_part(tmp_path):
    """A sim package without the required parts fails."""
    (tmp_path / "toy_sim").mkdir()
    (tmp_path / "toy_sim" / "__init__.py").write_text("")
    codes = {v.code for v in check_structure.check_package_parts(tmp_path)}
    assert "missing-part" in codes


def test_structure_lint_flags_half_a_plane_split(tmp_path):
    """control/ without data/ is a split that means nothing."""
    pkg = tmp_path / "toy_sim"
    for part in ("", "workload", "report", "control"):
        (pkg / part).mkdir(parents=True, exist_ok=True)
        (pkg / part / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text("")
    (pkg / "README.md").write_text("")
    codes = {v.code for v in check_structure.check_package_parts(tmp_path)}
    assert "half-a-plane-split" in codes


def test_structure_lint_flags_a_module_with_no_all(tmp_path):
    """A module that defines something public must say what its surface is."""
    src = "import os\n\n\ndef helper():\n    pass\n"
    assert check_structure.public_defs(_parse(src)) == ["helper"]
    assert check_structure.declared_all(_parse(src)) is None


def test_structure_lint_reads_a_declared_all():
    src = '__all__ = ["a", "b"]\n'
    assert check_structure.declared_all(_parse(src)) == ["a", "b"]


def test_public_defs_sees_classes_functions_and_constants_but_not_privates():
    src = (
        "CAP = 3\n"
        "_HIDDEN = 4\n"
        "class Thing:\n    pass\n"
        "class _Secret:\n    pass\n"
        "def go():\n    pass\n"
        "async def go_async():\n    pass\n"
        "def _helper():\n    pass\n"
    )
    assert check_structure.public_defs(_parse(src)) == [
        "CAP", "Thing", "go", "go_async"
    ]


def test_every_module_in_scope_actually_declares_its_surface():
    """The rule would be vacuous if it matched nothing; it covers ~50 modules."""
    import ast as _ast
    from pathlib import Path as _Path

    checked = 0
    for pkg in check_structure.GRAPH_PKGS:
        for f in (check_structure.REPO_ROOT / pkg).rglob("*.py"):
            if "__pycache__" in f.parts or "tests" in f.parts:
                continue
            if f.name in ("__init__.py", "__main__.py"):
                continue
            tree = _ast.parse(f.read_text())
            if check_structure.public_defs(tree):
                assert check_structure.declared_all(tree) is not None, f
                checked += 1
    assert checked > 40, f"only {checked} modules in scope -- rule 4 is too narrow"


def _toy_pkg(root, files):
    """Write ``{relative path: source}`` under ``root/toy_sim`` and return it."""
    for rel, src in files.items():
        path = root / "toy_sim" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)
    return ["toy_sim"]


def test_structure_lint_flags_a_folder_private_module(tmp_path):
    """Rule 2 fires: only its own folder imports it, so the name must say so."""
    pkgs = _toy_pkg(tmp_path, {
        "__init__.py": "",
        "workload/__init__.py": "",
        "workload/helper.py": "__all__ = ['go']\n\n\ndef go():\n    pass\n",
        "workload/scenarios.py": "from .helper import go\n",
    })
    violations = check_structure.check_private_naming(tmp_path, pkgs)
    assert [v.code for v in violations] == ["public-name-private-module"]
    assert violations[0].path.endswith("helper.py")


def test_structure_lint_accepts_an_underscored_private_module(tmp_path):
    """The same tree, renamed: no finding."""
    pkgs = _toy_pkg(tmp_path, {
        "__init__.py": "",
        "workload/__init__.py": "",
        "workload/_helper.py": "__all__ = ['go']\n\n\ndef go():\n    pass\n",
        "workload/scenarios.py": "from ._helper import go\n",
    })
    assert check_structure.check_private_naming(tmp_path, pkgs) == []


def test_structure_lint_accepts_a_module_imported_across_folders(tmp_path):
    """What ``kvcache_sim/control/request.py`` is: three planes pass its type."""
    pkgs = _toy_pkg(tmp_path, {
        "__init__.py": "",
        "workload/__init__.py": "",
        "control/request.py": "__all__ = ['Request']\n\n\nclass Request:\n    pass\n",
        "data/__init__.py": "",
        "data/serving.py": "from ..control.request import Request\n",
    })
    assert check_structure.check_private_naming(tmp_path, pkgs) == []


def test_structure_lint_ignores_a_module_nothing_imports(tmp_path):
    """Unimported is a deadness question, not a naming one -- rule 2 stays quiet."""
    pkgs = _toy_pkg(tmp_path, {
        "__init__.py": "",
        "workload/__init__.py": "",
        "workload/orphan.py": "__all__ = ['go']\n\n\ndef go():\n    pass\n",
    })
    assert check_structure.check_private_naming(tmp_path, pkgs) == []


def test_private_naming_rule_actually_examines_the_tree():
    """It would be vacuous if the import graph came back empty."""
    mods, importers = check_structure._import_graph()
    assert len(mods) > 50, f"only {len(mods)} modules in the graph"
    # The real tree's known-public and known-private neighbours, both resolved.
    assert "kvcache_sim.control.request" in mods
    assert {i.split(".")[0] for i in importers["kvcache_sim.control.request"]} >= {
        "kvcache_sim"
    }
    assert "kvcache_sim.workload._generator" in mods


def test_structure_lint_flags_a_public_name_only_its_own_module_uses(tmp_path):
    """Rule 5 fires: what ``longest_prefix_run`` was -- a helper with no caller.

    The defining module uses it, and a test imports it. Neither makes it surface.
    """
    pkgs = _toy_pkg(tmp_path, {
        "__init__.py": "",
        "workload/__init__.py": "",
        "control/view.py": (
            "__all__ = ['View']\n\n\n"
            "def walk(keys):\n    return len(keys)\n\n\n"
            "class View:\n    def lengths(self, keys):\n        return walk(keys)\n"
        ),
        "tests/__init__.py": "",
        "tests/test_view.py": "from ..control.view import walk\n",
    })
    violations = check_structure.check_name_privacy(tmp_path, pkgs)
    assert [v.code for v in violations] == ["public-name-no-consumer"]
    assert violations[0].path.endswith("view.py")
    assert "walk" in violations[0].message


def test_structure_lint_accepts_a_name_another_module_uses(tmp_path):
    """One real importer -- module or package re-export -- and the name is surface."""
    common = {
        "__init__.py": "",
        "workload/__init__.py": "",
        "control/view.py": "__all__ = ['walk']\n\n\ndef walk(keys):\n    return keys\n",
    }
    direct = _toy_pkg(tmp_path / "a", {
        **common, "workload/scenarios.py": "from ..control.view import walk\n",
    })
    assert check_structure.check_name_privacy(tmp_path / "a", direct) == []

    # Published through the package, imported from there: still a consumer.
    reexport = _toy_pkg(tmp_path / "b", {
        **common,
        "control/__init__.py": "from .view import walk\n",
        "workload/scenarios.py": "from ..control import walk\n",
    })
    assert check_structure.check_name_privacy(tmp_path / "b", reexport) == []


def test_structure_lint_accepts_an_attribute_call_and_a_cli_entry_point(tmp_path):
    """``import m`` ... ``m.go()`` is use, and a ``__main__`` hook is a caller."""
    pkgs = _toy_pkg(tmp_path / "c", {
        "__init__.py": "",
        "workload/__init__.py": "",
        "workload/helper.py": "__all__ = ['go']\n\n\ndef go():\n    pass\n",
        "workload/scenarios.py": (
            "from . import helper\n\n\ndef _run():\n    return helper.go()\n"
        ),
        "tool.py": (
            "__all__ = ['main']\n\n\ndef main():\n    return 0\n\n\n"
            "if __name__ == '__main__':\n    raise SystemExit(main())\n"
        ),
    })
    assert check_structure.check_name_privacy(tmp_path / "c", pkgs) == []


def test_name_privacy_rule_actually_examines_the_tree():
    """Vacuous if no name resolved: pin one known consumer edge in the real tree."""
    mods, trees, consumers = check_structure._name_consumers(
        check_structure.REPO_ROOT, check_structure.GRAPH_PKGS
    )
    assert len(mods) > 50, f"only {len(mods)} modules in the graph"
    users = consumers[("sim_common.async_engine", "AsyncEngine")]
    assert "realsim.simulation" in users, sorted(users)


def test_structure_lint_reads_a_layout_block():
    """The README parser must actually find a block, or rule 3 is vacuous."""
    block = check_structure._layout_block(
        "## Layout\n\nsome prose\n\n```\npkg/\n  thing.py\n```\n"
    )
    assert block is not None and "thing.py" in block


def test_lint_flags_threading():
    assert "threading-import" in _codes("import threading\nthreading.Lock()\n")
    assert "threading-import" in _codes("from threading import Thread\n")
    assert "multiprocessing-import" in _codes("import multiprocessing as mp\n")


def test_lint_flags_fork_and_wallclock_sleep():
    assert "fork" in _codes("import os\nos.fork()\n")
    assert "fork" in _codes("from os import fork\nfork()\n")
    assert "wallclock-sleep" in _codes("import time\ntime.sleep(1)\n")


def test_lint_flags_wallclock_reads_in_library_but_not_tests():
    src = "import time as wc\nx = wc.perf_counter()\n"
    assert "wallclock-read" in _codes(src, is_test=False)
    # In a test module, wall-clock reads (assertion measurement) are allowed.
    assert "wallclock-read" not in _codes(src, is_test=True)


def test_lint_flags_unseeded_random_but_allows_seeded():
    assert "unseeded-random" in _codes("import random\nrandom.random()\n")
    assert "unseeded-random" in _codes("import random\nrandom.Random()\n")
    assert "unseeded-random" in _codes("import random\nrandom.SystemRandom()\n")
    # A seeded Random and asyncio.sleep are the sanctioned primitives.
    assert _codes("import random\nr = random.Random(7)\n") == set()
    assert _codes("import asyncio\nasync def f():\n    await asyncio.sleep(5)\n") == set()


# --------------------------------------------------------------------------
# Plane separation: control/ may not import data/, the mesh, or a client.
# --------------------------------------------------------------------------


def test_control_module_is_recognised_by_path():
    assert is_control_module(CONTROL)
    assert is_control_module("dedup_sim/control/routing.py")
    assert not is_control_module(DATA)
    # A file *named* control.py is not a control package.
    assert not is_control_module("kvcache_sim/control.py")


def test_relative_imports_resolve_against_the_importing_package():
    assert resolve_module(CONTROL, 0, "realsim.mesh") == "realsim.mesh"
    assert resolve_module(CONTROL, 1, "_cache") == "kvcache_sim.control._cache"
    assert resolve_module(CONTROL, 2, "data.store") == "kvcache_sim.data.store"
    assert resolve_module(CONTROL, 1, "request") == "kvcache_sim.control.request"


def test_lint_flags_control_importing_data():
    # Both spellings of the same mistake.
    assert "control-imports-data" in _codes(
        "from ..data.store import KVStore\n", path=CONTROL
    )
    assert "control-imports-data" in _codes(
        "import kvcache_sim.data._decode\n", path=CONTROL
    )


def test_lint_flags_control_importing_the_mesh_or_a_client():
    assert "control-imports-execution" in _codes(
        "from realsim.mesh import Mesh\n", path=CONTROL
    )
    assert "control-imports-execution" in _codes(
        "from torchstore.client import LocalClient\n", path=CONTROL
    )
    assert "control-imports-execution" in _codes(
        "from realsim.seams.transport import InMemoryTransport\n", path=CONTROL
    )
    assert "control-imports-execution" in _codes(
        "from realsim.runner import Runner\n", path=CONTROL
    )


def test_lint_allows_what_control_is_supposed_to_use():
    allowed = (
        "from proposed import Policy, Selection\n"
        "from proposed import View\n"
        "from proposed import TransferCost\n"
        "from domain import prefill_time, MachineProfile\n"
        "from ._cache import LRUCache\n"
        "from .request import Request\n"
    )
    assert _codes(allowed, path=CONTROL) == set()


def test_lint_flags_a_plane_importing_the_run_scaffolding():
    """control/ and data/ ship; workload/ does not exist in production."""
    for path in (CONTROL, DATA):
        for line in (
            "from ..workload.request import Request\n",
            "from ..workload.scenarios import make_topology\n",
            "import kvcache_sim.workload.scenarios\n",
        ):
            assert "plane-imports-workload" in _codes(line, path=path), (path, line)


def test_workload_may_import_the_planes_it_wires():
    """The direction that is fine: the run's scaffolding reaches for real code."""
    src = (
        "from ..control.scheduler import CacheAwareScheduler\n"
        "from ..control.request import Request\n"
        "from ..data.serving import ServingPlane\n"
        "from ..report.summary import HotspotReport\n"
    )
    assert _codes(src, path="kvcache_sim/workload/scenarios.py") == set()


def test_lint_keeps_control_out_of_the_simulator():
    """Estimates come through a protocol; machine facts come from domain."""
    assert "control-imports-execution" in _codes(
        "from sim_common.cost_model import _get_time\n", path=CONTROL
    )
    # Both the module path and the package re-export, since proposed surfaces
    # its whole contract at package level.
    for line in (
        "from proposed.plane import DataPlane\n",
        "from proposed import DataPlane\n",
        "from proposed import Deployment\n",
    ):
        assert "control-imports-execution" in _codes(line, path=CONTROL), line


def test_data_may_hold_decisions_but_not_the_simulator():
    """data/ executes decisions, but it is application code: no harness in it."""
    allowed = (
        "from proposed import Deployment\n"
        "from proposed import DataPlane\n"
        "from domain import Model\n"
        "from ..control.scheduler import Plan\n"
        "from .store import KVStore\n"
    )
    assert _codes(allowed, path=DATA) == set()
    for line in ("from realsim.mesh import Mesh\n", "from sim_common.trace import Trace\n"):
        assert "data-imports-simulator" in _codes(line, path=DATA), line


def test_the_simulator_rules_do_not_apply_to_workload():
    """workload/ is where the harness is wired up, so it may name it."""
    src = "from realsim.mesh import Mesh\nfrom sim_common.trace import Trace\n"
    assert _codes(src, path="kvcache_sim/workload/_serving.py") == set()


def test_lint_flags_the_proposal_leaning_on_the_simulator():
    """``proposed/`` has to be implementable inside torchstore with nothing under it."""
    for line in (
        "from realsim.mesh import Mesh\n",
        "from realsim.seams.controller_handle import FakeControllerHandle\n",
        "import torchstore\n",
        "from kvcache_sim.control._source import LongestPrefixPolicy\n",
    ):
        assert "proposed-imports-simulator" in _codes(line, path=PROPOSED), line


def test_the_proposal_stands_on_its_own():
    """It may use itself and the stdlib -- nothing else, not even sim_common."""
    assert _codes(
        "from proposed import Endpoint, locality, Tier\n"
        "from .view import View\n"
        "import asyncio\n",
        path=PROPOSED,
    ) == set()
    assert "proposed-imports-simulator" in _codes(
        "from sim_common.topology import Endpoint\n", path=PROPOSED
    )
