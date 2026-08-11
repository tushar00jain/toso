"""One unrouted burst, for the tests that drive the engine through it.

``realsim``'s own tests exercise the stack through ``putget_sim``'s
capability-free fixture (see ``realsim/README.md``). They want one line per run,
so this pairs :func:`putget_sim.workload.scenarios.burst` with
:func:`realsim.run.execute` -- the same two calls a demo makes, with the run
knobs the tests vary exposed as arguments.

Test scaffolding, not API: a capability declares :class:`~realsim.run.Run` values
and lets ``execute`` drive them.
"""

from __future__ import annotations

from typing import Optional

import torch

from putget_sim.workload.put_get import DEFAULT_N, MODE_META, PutGetBurst
from realsim.run import execute, Result, Run
from sim_common.cost_model import MachineProfile

__all__ = ["run_burst"]


def run_burst(
    num_readers: int = 3,
    *,
    n: int = DEFAULT_N,
    dtype: torch.dtype = torch.float32,
    mode: str = MODE_META,
    profile: Optional[MachineProfile] = None,
    random_seed: Optional[int] = None,
    real_directory: Optional[bool] = None,
) -> Result:
    """Run one unrouted burst end-to-end on a fresh deterministic engine."""
    workload = PutGetBurst(num_readers, n=n, dtype=dtype, mode=mode, profile=profile)
    return execute(
        Run("unrouted", workload, profile=workload.profile),
        random_seed=random_seed,
        real_directory=real_directory,
    )
