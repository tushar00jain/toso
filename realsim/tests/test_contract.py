"""The contract lint, wired into the test suite.

``test_sim_paths_obey_the_concurrency_contract`` fails the build if any simulated
code path in the scanned packages reaches for a determinism-breaking primitive
(threads, forks, wall-clock sleeps/reads, unseeded randomness), **or** if a
capability's ``control/`` module imports the executing half (a ``data/`` package,
the mesh, or a store client). The companion tests prove the checker actually
detects each banned pattern and does not flag the sanctioned ones -- so a green
run means the contract holds, not that the lint is asleep.

See ``realsim/tools/check_contract.py`` for the full contract and its rationale.
"""

from __future__ import annotations

from realsim.tools.check_contract import (
    format_violations,
    is_control_module,
    resolve_module,
    scan_default,
    scan_source,
)


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
    assert resolve_module(CONTROL, 1, "cache") == "kvcache_sim.control.cache"
    assert resolve_module(CONTROL, 2, "data.store") == "kvcache_sim.data.store"
    assert resolve_module(CONTROL, 2, "workload.request") == "kvcache_sim.workload.request"


def test_lint_flags_control_importing_data():
    # Both spellings of the same mistake.
    assert "control-imports-data" in _codes(
        "from ..data.store import KVStore\n", path=CONTROL
    )
    assert "control-imports-data" in _codes(
        "import kvcache_sim.data.decode\n", path=CONTROL
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
        "from proposed.policy import Policy, Selection\n"
        "from proposed.view import View\n"
        "from domain.llm import prefill_time\n"
        "from sim_common.cost_model import get_time\n"
        "from .cache import LRUCache\n"
        "from ..workload.request import Request\n"
    )
    assert _codes(allowed, path=CONTROL) == set()


def test_the_rule_only_applies_to_control():
    """The data plane is *supposed* to hold the mesh and the decisions."""
    src = (
        "from realsim.mesh import Mesh\n"
        "from ..control.scheduler import Plan\n"
        "from .store import KVStore\n"
    )
    assert _codes(src, path=DATA) == set()


def test_lint_flags_the_proposal_leaning_on_the_simulator():
    """``proposed/`` has to be implementable inside torchstore with nothing under it."""
    for line in (
        "from realsim.mesh import Mesh\n",
        "from realsim.seams.controller_handle import FakeControllerHandle\n",
        "import torchstore\n",
        "from kvcache_sim.control.source import LongestPrefixPolicy\n",
    ):
        assert "proposed-imports-simulator" in _codes(line, path=PROPOSED), line


def test_lint_allows_what_the_proposal_may_use():
    """Locality types are part of the ask, so the proposal may name them."""
    allowed = (
        "from sim_common.topology import Endpoint, locality, Tier\n"
        "from .view import View\n"
    )
    assert _codes(allowed, path=PROPOSED) == set()
