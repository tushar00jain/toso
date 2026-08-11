"""The contract lint, wired into the test suite.

``test_sim_paths_obey_the_concurrency_contract`` fails the build if any simulated
code path in the scanned packages reaches for a determinism-breaking primitive
(threads, forks, wall-clock sleeps/reads, unseeded randomness), **or** if a
capability's ``control/`` module imports the executing half (a ``data/`` package,
the mesh, or a store client). The companion tests prove the checker actually
detects each banned pattern and does not flag the sanctioned ones -- so a green
run means the contract holds, not that the lint is asleep.

``test_sim_packages_keep_their_shape`` does the same for the *structure* lint:
the parts every ``*_sim`` carries, the underscore on a folder-private module, and
a README layout block that matches the tree.

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
    assert resolve_module(CONTROL, 2, "workload.request") == "kvcache_sim.workload.request"


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
        "from ..workload.request import Request\n"
    )
    assert _codes(allowed, path=CONTROL) == set()


def test_lint_keeps_control_out_of_the_simulator():
    """Estimates come through a protocol; machine facts come from domain."""
    assert "control-imports-execution" in _codes(
        "from sim_common.cost_model import get_time\n", path=CONTROL
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
