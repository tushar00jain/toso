"""A sim's command line, declared rather than hand-rolled: :class:`Demo`.

Everything a demo needs is here -- the shared run flags, how they become process
config, how a trace is dumped, and the parse/dispatch loop. They used to be split
across ``cli.py`` and this module even though nothing but this module read them,
which just meant one more file to open. They are underscored for the same reason:
a demo declares itself and :meth:`Demo.main` applies the flags, so no caller ever
names them.

Three ``__main__.py`` files had grown the same skeleton -- a parser, the five
shared run flags, an apply-the-flags call, a ``_section`` helper (byte-identical
in two of them), a trace-dump helper, a ``SCENARIOS`` dict, a run-all-or-run-one
dispatch, and a ``if __name__`` tail. Nothing held those seven things in shape
except that whoever wrote the third demo copied the second.

Here they are once. A capability subclasses :class:`Demo`, declares its scenarios
and is done; the abstract :meth:`Demo.scenarios` is what makes "declared its
scenarios" a requirement rather than a convention:

    class DedupDemo(Demo):
        name = "dedup_sim"
        description = "Dedup read-routing on the real TorchStore directory."

        def scenarios(self):
            return [Scenario("dedup", _dedup)]

    if __name__ == "__main__":
        DedupDemo().main()

A scenario is a :class:`Scenario` subclass with two methods, not a callback:
:meth:`Scenario.runs` declares the configurations and :meth:`Scenario.show`
narrates the results. :meth:`Demo.main` is what executes, between them -- so
"which simulations does this scenario run" is answerable by reading a method that
runs nothing, and every sim executes the same way. Narration stays code because a
scenario's prose is genuinely bespoke; what it is *handed* is a :class:`Console`,
so section headers, trace dumps and the summary line are one implementation
rather than three. The scenarios live in the capability's
``workload/scenarios.py``, next to the runs they declare, which leaves a
``__main__.py`` that is only a declaration.
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence

from sim_common import config
from sim_common.report import configure_logging, section

from realsim.run import Report, Result, Run

__all__ = ["Console", "Scenario", "Demo"]


# --------------------------------------------------------------------------- #
# The run flags every demo accepts, and what they configure. Private: Demo.main
# is the only caller, and a capability's own flags go through Demo.flags.
#
# The three entry points (``python -m putget_sim``, ``python -m dedup_sim``,
# ``python -m kvcache_sim``) all take the same five, all turn them into the same
# ``config.configure`` call, and all dump a trace the same way. Capability-
# specific flags (which scenario, how many readers) stay with the demo that owns
# them, via :meth:`Demo.flags` -- only the run knobs are shared, because only
# they mean the same thing everywhere.
# --------------------------------------------------------------------------- #


def _add_run_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the run knobs every demo accepts. Returns the parser, for chaining."""
    parser.add_argument(
        "-v", "--verbose", "--debug", action="store_true", dest="verbose",
        help="show the full per-event virtual-time trace (log level DEBUG)",
    )
    parser.add_argument(
        "--fingerprint", action="store_true",
        help="print each run's trace fingerprint (a determinism-debugging digest, "
        "folded from the trace on demand; off by default -- it is not part of the "
        "performance measurement)",
    )
    parser.add_argument(
        "--shim-directory", action="store_true",
        help="back the controller directory with a lightweight dict shim instead "
        "of the real torchstore Trie (opt-in; skips the per-key trie tax on scale "
        "runs). Metrics are byte-identical either way; the real directory is the "
        "default.",
    )
    parser.add_argument(
        "--contention", choices=("none", "serialize", "progressive"), default=None,
        help="network/storage contention model (default: none -- independent, "
        "full-bandwidth transfers, the historical behavior). 'serialize' serves a "
        "resource one transfer at a time; 'progressive' shares a resource's "
        "bandwidth max-min fairly among concurrent transfers. Non-default modes "
        "change timing (they are a fidelity model, not byte-identical to none).",
    )
    parser.add_argument(
        "--collapse-charges", action="store_true",
        help="coalesce each transport op's per-component charges (a get's "
        "storage+mem+network; a put's network+storage) into one virtual-clock "
        "sleep, cutting the per-op event-loop bounces on the non-contended path. "
        "Same total time in isolation and fabric bytes are unchanged; not "
        "byte-identical (the sub-charge instants collapse). Inert when "
        "--contention is not none.",
    )
    parser.add_argument(
        "--controller-rtt", type=float, default=None, metavar="SECONDS",
        help="one-way latency of the hop to the directory service (default: 0 -- "
        "the in-process call). Every capability pays it on every directory read, "
        "and so does every consultation of a policy installed there; it is a "
        "fidelity model, not byte-identical to 0.",
    )
    parser.add_argument(
        "--coordinator-rtt", type=float, default=None, metavar="SECONDS",
        help="one-way latency of the hop to the control plane's coordinator "
        "service (default: 0 -- the in-process call, byte-identical to holding "
        "the object). A request pays it out and back before prefill can start, "
        "so a non-zero value lands in TTFT; it is a fidelity model, not "
        "byte-identical to 0.",
    )
    return parser


def _apply_run_flags(
    args: argparse.Namespace, logger: Optional[logging.Logger] = None
) -> None:
    """Set the process config from ``args``, once, and configure logging.

    An unset flag defers to the ``TOSO_*`` env var / default, so this never
    overrides an ambient setting with "off". Pass ``logger`` when the demo's own
    logger must drop to DEBUG independently of the root (``putget_sim`` keeps the
    root at INFO so torchstore's own debug logging stays quiet).
    """
    config.configure(
        fingerprint=args.fingerprint or None,
        real_directory=False if args.shim_directory else None,
        contention=args.contention,
        collapse_charges=args.collapse_charges or None,
        coordinator_rtt=args.coordinator_rtt,
        controller_rtt=args.controller_rtt,
    )
    if logger is None:
        configure_logging(logging.DEBUG if args.verbose else logging.INFO)
        return
    # Keep the ROOT logger at INFO so the real torchstore code's own DEBUG latency
    # logs stay quiet; then let the stdout handler and this demo's logger drop to
    # DEBUG so -v shows the virtual-time event trace.
    configure_logging(logging.INFO)
    if args.verbose:
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)


def _log_trace(
    logger: logging.Logger, trace: Any, *, limit: Optional[int] = None
) -> None:
    """Emit a trace at DEBUG (shown only under ``-v``).

    ``limit`` keeps the first N events, for a run with thousands of them. The
    fingerprint line stays with the demo that prints it: some demos print several
    per run and label them, so the wording is theirs, not this helper's.
    """
    lines = trace.render_lines()
    capped = limit is not None and len(lines) > limit
    logger.debug(
        "(a) event trace (%d events%s)",
        len(lines),
        f", first {limit} shown" if capped else "",
    )
    for line in lines[:limit] if capped else lines:
        logger.debug(line)
    if capped:
        logger.debug("... (%d more)", len(lines) - limit)
    logger.debug("")


# --------------------------------------------------------------------------- #
# The demo itself.
# --------------------------------------------------------------------------- #


class Console:
    """How a demo prints: headers, prose, traces, summaries. One implementation.

    Handed to every scenario, so the ``_section`` / ``_log_trace`` pair that each
    demo used to define for itself exists once.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def section(self, title: str) -> None:
        """A run of ``=`` and a title, at INFO."""
        section(self.logger, title)

    def info(self, message: str, *args) -> None:
        """One line of prose, at INFO."""
        self.logger.info(message, *args)

    def trace(
        self, trace, *, limit: Optional[int] = None, label: str = "run"
    ) -> None:
        """The per-event trace at DEBUG (so it shows only under ``-v``).

        Follows it with the fingerprint line when ``--fingerprint`` is set. A demo
        that prints several fingerprints per scenario labels them with ``label``.
        """
        _log_trace(self.logger, trace, limit=limit)
        if config.current().fingerprint:
            self.logger.info("%s fingerprint: %s", label, trace.fingerprint())

    def summary(self, report: Report) -> None:
        """The ``(b) summary`` header and the report's rendering."""
        self.logger.info("(b) summary")
        self.logger.info(report.render())


class Scenario(ABC):
    """One named comparison: which configurations, and how to narrate them.

    The two halves are deliberately separate methods. :meth:`runs` *declares* --
    it returns :class:`~realsim.run.Run` values and executes nothing;
    :meth:`Demo.main` executes them, in one place, and hands the results to
    :meth:`show`, which only prints. Before this split a scenario was one
    callback that did all three, so every one of them silently ran simulations
    inside a function called "show".

    Subclass it next to the runs it declares -- a capability's
    ``workload/scenarios.py`` -- and hand instances to
    :meth:`Demo.scenarios`::

        class Hotspot(Scenario):
            name = "hotspot"

            def __init__(self, seed: int = 0):
                self.seed = seed

            def runs(self, args=None):
                return [configure(...), configure(...)]

            def show(self, console, results):
                console.section("HOTSPOT: ...")
                console.summary(HotspotReport(results))
    """

    #: The CLI's positional choice for this scenario.
    name: str = ""

    @abstractmethod
    def runs(self, args: Optional[argparse.Namespace] = None) -> Sequence[Run]:
        """The configurations to compare. Declares; executes nothing.

        ``args`` is the parsed command line, for a scenario whose shape a flag
        varies. One parameterized by construction instead ignores it, so a test
        can ask for its runs with no command line at all.
        """

    @abstractmethod
    def show(self, console: Console, results: Sequence[Result]) -> None:
        """Narrate the results, in the order :meth:`runs` declared them."""


class Demo(ABC):
    """A sim's command line: flags, logging, dispatch. Subclass and declare.

    Subclasses set :attr:`name` and :attr:`description` and implement
    :meth:`scenarios`. :meth:`flags` and :meth:`takeaway` are optional hooks with
    working defaults -- override only if the sim has extra knobs or a closing
    word.
    """

    #: Package name; also the logger name and the ``python -m`` program name.
    name: str = ""
    #: One-paragraph ``--help`` description.
    description: str = ""
    #: Whether the demo's own logger drops to DEBUG under ``-v`` independently of
    #: the root, keeping torchstore's own debug logging quiet.
    own_logger: bool = False

    @abstractmethod
    def scenarios(self) -> Sequence[Scenario]:
        """The scenarios this demo can run, in the order ``--help`` lists them."""

    def flags(self, parser: argparse.ArgumentParser) -> None:
        """Add capability-specific CLI flags. Default: none."""

    def takeaway(self, console: Console) -> None:
        """Closing prose, printed after a run-everything pass. Default: none."""

    def run_knobs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Per-invocation knobs for :meth:`~realsim.run.Run.execute`.

        For a sim whose own flags reach the engine rather than the scenario --
        ``putget_sim``'s ``--seed``. Default: none.
        """
        return {}

    # -- the shared plumbing ------------------------------------------------- #
    def _play(
        self, scenario: Scenario, console: Console, args: argparse.Namespace
    ) -> None:
        """Declare, execute, narrate -- the three steps, visibly in that order."""
        runs = scenario.runs(args)
        results = [run.execute(**self.run_knobs(args)) for run in runs]
        scenario.show(console, results)

    def main(self, argv: Optional[Sequence[str]] = None) -> None:
        """Parse, configure, dispatch. The same for every sim."""
        logger = logging.getLogger(self.name)
        console = Console(logger)
        scenarios = list(self.scenarios())
        by_name = {s.name: s for s in scenarios}

        parser = argparse.ArgumentParser(
            prog=f"python -m {self.name}", description=self.description
        )
        if len(scenarios) > 1:
            parser.add_argument(
                "scenario", nargs="?", choices=sorted(by_name),
                help="scenario to run (default: all). one of: "
                + ", ".join(sorted(by_name)),
            )
        self.flags(parser)
        _add_run_flags(parser)
        args = parser.parse_args(list(argv) if argv is not None else None)

        # Set the process config once from the CLI flags (an unset flag defers to
        # the TOSO_* env / default). Every Trace, controller adapter, resource
        # registry and the transport's collapse decision reads it ambiently.
        _apply_run_flags(args, logger if self.own_logger else None)

        chosen = getattr(args, "scenario", None)
        if chosen is not None:
            self._play(by_name[chosen], console, args)
            return
        for scenario in scenarios:
            self._play(scenario, console, args)
        self.takeaway(console)
