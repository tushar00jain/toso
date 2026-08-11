"""Perf guard for the realsim scenario (see ``docs/realsim_design.md`` s12).

Two invariants are enforced here:

1. **No real allocation on the sim path.** The whole reason realsim uses meta
   tensors / descriptors is that the simulation must carry *zero* real tensor
   bytes no matter how large the modeled payload is. ``test_meta_*`` /
   ``test_metadata_*`` run a burst whose modeled payload is a quarter-gigabyte and
   assert (a) every carrier has a null data pointer / is a descriptor and (b) peak
   process RSS does **not** grow anywhere near the materialized size. An accidental
   real ``torch.empty(n)`` regression would blow past the RSS headroom, so this
   catches the exact failure the design warns about.

2. **The dedup capability runs at parity with the base burst.** ``test_realsim_*``
   runs the base realsim (naive) burst and the ``dedup_sim`` dedup scenario each in
   a fresh subprocess (so the comparison is the real end-to-end, import-dominated
   cost the product owner cares about) and asserts the base burst's wall + peak RSS
   stay within a tolerant multiple of the dedup scenario's. Both drive the same real
   torchstore code on the same import baseline, so parity is expected; the bound is
   loose enough not to flake but tight enough to catch a gross regression (heavy new
   imports, real work).

Wall-clock reads here are assertion measurement, not sim-path control flow, so
they are allowed under the concurrency contract (see
``realsim/tools/check_contract.py``).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from realsim.tests._burst import run_burst
from putget_sim.workload.put_get import MODE_META, MODE_METADATA, TensorDescriptor

REPO_ROOT = Path(__file__).resolve().parents[2]

# A modeled payload this large (67M float32 elements = 256 MiB) would be
# impossible to miss in RSS if it were ever materialized; a meta tensor /
# descriptor is zero storage, so peak RSS must stay essentially flat.
BIG_N = 1 << 26
BIG_PAYLOAD_BYTES = BIG_N * 4
# Generous headroom vs. the 256 MiB a real allocation would cost: interpreter
# churn is well under this, a real-tensor regression is well over it.
RSS_HEADROOM_KB = 64 * 1024  # 64 MiB


def _maxrss_kb() -> int:
    """Peak resident set size of this process so far, in KiB (Linux ru_maxrss)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_meta_sim_path_allocates_no_real_data_even_at_scale():
    # Warm up first so imports/one-time caches are already counted in ``before``;
    # ru_maxrss is a high-water mark, so a big real allocation during the measured
    # run would raise it even if it were freed afterwards.
    run_burst(num_readers=2, n=16, mode=MODE_META)
    before = _maxrss_kb()
    res = run_burst(num_readers=3, n=BIG_N, mode=MODE_META)
    after = _maxrss_kb()

    # Carrier invariant: real tensors, but on the meta device with a null data
    # pointer -- i.e. zero real allocation.
    assert isinstance(res.workload.expected, torch.Tensor)
    assert res.workload.expected.device.type == "meta"
    assert res.workload.expected.data_ptr() == 0
    for payload in res.results.values():
        assert isinstance(payload, torch.Tensor)
        assert payload.data_ptr() == 0
    # The modeled size is exact even though nothing was allocated.
    assert res.workload.expected.numel() * res.workload.expected.element_size() == BIG_PAYLOAD_BYTES

    # The whole point: a 256 MiB modeled payload must not move peak RSS.
    assert after - before < RSS_HEADROOM_KB, (
        f"peak RSS grew {after - before} KiB for a {BIG_PAYLOAD_BYTES} B modeled "
        "payload -- the sim path allocated real tensor data (regression)"
    )


def test_metadata_sim_path_carries_only_descriptors_at_scale():
    run_burst(num_readers=2, n=16, mode=MODE_METADATA)
    before = _maxrss_kb()
    res = run_burst(num_readers=3, n=BIG_N, mode=MODE_METADATA)
    after = _maxrss_kb()

    assert isinstance(res.workload.expected, TensorDescriptor)
    for payload in res.results.values():
        assert isinstance(payload, TensorDescriptor)
    assert res.workload.expected.nbytes == BIG_PAYLOAD_BYTES
    assert after - before < RSS_HEADROOM_KB


# --------------------------------------------------------------------------- #
# Parity vs. a capability sim (subprocess, import-dominated end-to-end cost).
# --------------------------------------------------------------------------- #

# Each snippet imports the sim, runs a small scenario, and prints its own peak
# RSS (KiB) on the last stdout line. Kept small so the measured cost is dominated
# by the shared torch/monarch import baseline, which is the point of the parity.
_REALSIM_SNIPPET = (
    "import resource;"
    "from putget_sim.workload.scenarios import burst;"
    "burst(3, n=1024)[0].execute();"
    "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
)
_DEDUP_SNIPPET = (
    "import resource;"
    "from dedup_sim.workload.scenarios import dedup_vs_baseline;"
    "from putget_sim.workload.put_get import PutGetBurst;"
    "[r.execute() for r in dedup_vs_baseline(burst=PutGetBurst(3, n=1024), caps=(1,))];"
    "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
)


def _measure(snippet: str) -> tuple[float, int]:
    """Run ``snippet`` in a fresh interpreter; return (wall_seconds, peak_rss_kb)."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    wall = time.perf_counter() - start
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    rss_kb = int(proc.stdout.strip().splitlines()[-1])
    return wall, rss_kb


def test_realsim_run_is_not_materially_more_expensive_than_model_sim():
    dwall, drss = _measure(_DEDUP_SNIPPET)
    rwall, rrss = _measure(_REALSIM_SNIPPET)

    # The dedup scenario and the base burst both drive the real torchstore code and
    # are dominated by the shared torch/monarch import, so the base burst must not be
    # *materially* more expensive than the dedup run. Bounds are loose (the baseline
    # is import-dominated and noisy) but a gross regression (e.g. a heavy new import
    # or real per-element work) trips them.
    assert rwall <= dwall * 2.0 + 1.0, (
        f"realsim burst wall {rwall:.2f}s vs dedup scenario {dwall:.2f}s -- the base "
        "burst became materially more expensive to run"
    )
    assert rrss <= drss * 1.5 + 200 * 1024, (
        f"realsim burst peak RSS {rrss} KiB vs dedup scenario {drss} KiB -- the base "
        "burst uses materially more memory"
    )
