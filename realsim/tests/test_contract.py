"""The concurrency-contract lint, wired into the test suite.

``test_sim_paths_obey_the_concurrency_contract`` fails the build if any simulated
code path under ``realsim/`` or ``sim_common/`` reaches for a determinism-breaking
primitive (threads, forks, wall-clock sleeps/reads, unseeded randomness). The
companion tests prove the checker actually detects each banned pattern and does
not flag the sanctioned ones -- so a green run means the contract holds, not that
the lint is asleep.

See ``realsim/tools/check_contract.py`` for the full contract and its rationale.
"""

from __future__ import annotations

from realsim.tools.check_contract import (
    format_violations,
    scan_default,
    scan_source,
)


def _codes(source: str, *, is_test: bool = False):
    return {v.code for v in scan_source(source, "snippet.py", is_test=is_test)}


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
