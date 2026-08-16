"""Every run of every scenario, over a knob matrix, as one diffable line each.

The third of this package's CLIs, and the one that *detects* a change:
:mod:`realsim.tools.check_contract` and :mod:`realsim.tools.check_structure` police the
shape, :mod:`sim_common.diverge` localizes a divergence once one is known, and this
reports the numbers a change must not move -- each run's fingerprint and the headline
metrics its ledger keeps.

Comparing two checkouts is a diff of two runs of this::

    PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.parity

which is why nothing here knows about checkouts, worktrees or where it lives: a report
line carries no path, no timestamp and no wallclock, so two runs of the same tree are
byte-identical. A dependency's import banner may precede the report; it says the same
thing in either tree.

How to read a line that moved
-----------------------------
A line carries a fingerprint and then metrics, and the two answer different questions.

* a **metric** that moved is a behaviour change: reject it, or say what the change was
  meant to move and why the new number is the right one;
* a **fingerprint** that moved with every metric identical is events *reordered*, not
  events changed. The digest chains the whole ordered event sequence
  (:meth:`sim_common.trace.Trace.fingerprint`), so it pins the relative order of two
  events at one simulated instant -- including where nothing gives them one. Where a
  tie-break is stated (an id, a sequence number) that order is a guarantee and moving it
  is a bug; where nothing states one, two concurrent events may land either way and both
  are the run. So such a line is neither automatically a bug nor automatically fine: name
  the events that swapped, the instant they swapped at
  (:func:`sim_common.diverge.first_divergence`) and what makes them concurrent, then
  rebaseline.

The first line is this file's own hash, which is what makes that diff mean anything: a
comparison between two *versions of this tool* is not a comparison of two trees, and
without the line it looks exactly like one. If it differs, copy this file into the other
checkout and sweep again -- with a non-prompting command (``command cp -f``,
``install -m 644``), since an interactive ``cp`` waits for a confirmation an unattended
run never sends, which is a deadlock rather than a delay.

The matrix is the default knobs plus the boundary latencies, because those are the
knobs a change to how a request is routed moves first (:mod:`sim_common.config`). Cells
run in the order given and scenarios in the order a demo declares them, so a sweep is
reproducible in its own right -- which matters more than it looks: a run's numbers can
depend on what ran before it in the same process, so a fixed order is what makes two
sweeps comparable at all.

Not a test. What must *always* hold is asserted in ``*/tests/``; this answers "did
anything move, and where", which no assertion can phrase in advance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import io
import logging
import sys
from contextlib import redirect_stdout
from importlib import import_module
from typing import Any, Dict, List, Sequence, Tuple

from sim_common import config

from realsim.demo import Demo, Scenario
from realsim.run import Result, Run

__all__ = [
    "DEFAULT_CELLS",
    "METRICS",
    "SIM_SUFFIX",
    "parse_cell",
    "demos",
    "measure",
    "sweep",
    "main",
]

#: What a change to routing, placement or a seam moves first: nothing, then each
#: boundary latency on its own. One-knob cells, so a line that moves names its cause.
DEFAULT_CELLS: Tuple[str, ...] = (
    "default",
    "client_rtt=0.5",
    "client_rtt=2.0",
    "control_rtt=0.5",
    "control_rtt=2.0",
)

#: Ledger readings printed for a run that has them, in this order. A capability's own
#: ledger keeps what its scenarios are judged on and nothing else, so each line carries
#: the subset that exists -- absent is absent, not zero.
METRICS: Tuple[str, ...] = (
    "hit_rate",
    "mean_ttft",
    "mean_latency",
    "fabric_bytes",
    "handoff_bytes",
    "transfer_bytes",
    "origin_bytes",
    "wallclock",
    "rejections",
    "items_done",
)

SIM_SUFFIX = "_sim"


def parse_cell(spec: str) -> Dict[str, Any]:
    """One matrix cell, as the config overrides it names.

    ``default`` is no overrides; anything else is ``knob=value`` pairs separated by
    commas. Values are read as int, then float, then bool, then string -- the order
    :class:`~sim_common.config.SimConfig`'s fields come in.
    """
    if spec == "default":
        return {}
    knobs: Dict[str, Any] = {}
    for pair in spec.split(","):
        name, _, raw = pair.partition("=")
        knobs[name.strip()] = _value(raw.strip())
    return knobs


def _value(raw: str) -> Any:
    for read in (int, float):
        try:
            return read(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def demos(only: Sequence[str] = ()) -> List[Tuple[str, Demo]]:
    """``(sim package, its demo)`` for every ``*_sim``, by package name.

    Found rather than listed, off the same mark ``check_structure`` uses -- a package
    whose ``__main__`` declares a :class:`~realsim.demo.Demo` is a sim with scenarios to
    sweep -- so a capability written next is swept without touching this.
    """
    out: List[Tuple[str, Demo]] = []
    for name in sorted(_packages()):
        if only and name not in only:
            continue
        module = import_module(f"{name}.__main__")
        for _attr, value in sorted(vars(module).items()):
            if inspect.isclass(value) and issubclass(value, Demo) and value is not Demo:
                out.append((name, value()))
                break
    return out


def _declared(demo: Demo) -> argparse.Namespace:
    """The demo's own flags at their defaults, which a scenario may read.

    A scenario is declared against the command line its demo offers (how many readers,
    which carrier), so a sweep of what that demo runs by default is a sweep with those
    defaults -- taken from the demo rather than restated here.
    """
    parser = argparse.ArgumentParser(add_help=False)
    demo.flags(parser)
    return parser.parse_args([])


def _packages() -> List[str]:
    """Every importable ``*_sim`` package under the repo root."""
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    return [
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.endswith(SIM_SUFFIX) and (p / "__init__.py").exists()
    ]


def measure(result: Result) -> str:
    """One run's fingerprint and readings, as the tail of its line."""
    ledger = result.ledger
    read = []
    for name in METRICS:
        if not hasattr(ledger, name):
            continue
        value = getattr(ledger, name)
        read.append(f"{name}={value:.6f}" if isinstance(value, float)
                    else f"{name}={value}")
    return " ".join([result.trace.fingerprint(), *read])


def sweep(
    cells: Sequence[str] = DEFAULT_CELLS,
    sims: Sequence[str] = (),
    scenarios: Sequence[str] = (),
) -> List[str]:
    """One line per ``(cell, sim, scenario, run)``, in that order.

    Fingerprints are on for the whole sweep
    (:attr:`sim_common.config.SimConfig.fingerprint`), which costs the hash chain and
    changes nothing measured.

    A run's own output is not the report, so what it prints while it runs is swallowed
    and the engine's own logger is detached: a discarded engine leaves its decode tasks
    to the collector, and what that logs names the file they were suspended in -- one
    path in a line makes two checkouts differ for a reason that is not a behaviour
    change. Each finished run is collected here rather than at interpreter shutdown, so
    that happens while there is still something swallowing it.
    """
    logging.getLogger("asyncio").propagate = False
    found = demos(sims)
    lines: List[str] = []
    with redirect_stdout(io.StringIO()):
        for spec in cells:
            knobs = parse_cell(spec)
            for package, demo in found:
                for scenario in demo.scenarios():
                    if scenarios and scenario.name not in scenarios:
                        continue
                    with config.overrides(fingerprint=True, **knobs):
                        for run in scenario.runs(_declared(demo)):
                            lines.append(_line(spec, package, scenario, run))
                            gc.collect()
    return lines


def _line(spec: str, package: str, scenario: Scenario, run: Run) -> str:
    """Execute one run and render it: knobs, where it came from, what it produced."""
    label = run.label or "-"
    return (
        f"{spec:<18} {package:<12} {scenario.name:<16} {label:<14} "
        f"{measure(run.execute())}"
    )


def _version() -> str:
    """This file's own content hash: what two checkouts have to agree on first."""
    with open(__file__, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()[:12]


def main(argv: List[str] | None = None) -> int:
    """CLI: print the sweep. Exit status says nothing -- the diff does."""
    parser = argparse.ArgumentParser(
        prog="python -m realsim.tools.parity",
        description="One line per run: fingerprint and headline metrics, over a knob "
                    "matrix. Diff two checkouts' output to see what a change moved.",
    )
    parser.add_argument(
        "--cell", action="append", metavar="KNOB=VALUE[,...]",
        help=f"a matrix cell ('default' for none). Repeatable; defaults to "
             f"{', '.join(DEFAULT_CELLS)}",
    )
    parser.add_argument(
        "--sim", action="append", metavar="PACKAGE",
        help="sweep only this sim package. Repeatable; defaults to all of them",
    )
    parser.add_argument(
        "--scenario", action="append", metavar="NAME",
        help="sweep only this scenario. Repeatable; defaults to all of them",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    print(f"tool {_version()}")
    for line in sweep(
        tuple(args.cell) if args.cell else DEFAULT_CELLS,
        tuple(args.sim or ()),
        tuple(args.scenario or ()),
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
