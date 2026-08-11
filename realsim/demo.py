"""A sim's command line, declared rather than hand-rolled: :class:`Demo`.

Three ``__main__.py`` files had grown the same skeleton -- a parser, the five
shared run flags, ``apply_run_flags``, a ``_section`` helper (byte-identical in
two of them), a ``_log_trace`` helper, a ``SCENARIOS`` dict, a run-all-or-run-one
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

Scenario narration stays a function. A scenario's prose is genuinely bespoke --
which comparison it drew, what the reader should notice -- and pressing it into
declarative fields would contort it for no gain. What that function is *handed*
is a :class:`Console`, so section headers, trace dumps and the summary line are
one implementation rather than three.
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from sim_common import config
from sim_common.report import section

from realsim.cli import add_run_flags, apply_run_flags, log_trace
from realsim.reporting import Report

__all__ = ["Console", "Scenario", "Demo"]


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
        log_trace(self.logger, trace, limit=limit)
        if config.current().fingerprint:
            self.logger.info("%s fingerprint: %s", label, trace.fingerprint())

    def summary(self, report: Report) -> None:
        """The ``(b) summary`` header and the report's rendering."""
        self.logger.info("(b) summary")
        self.logger.info(report.render())


@dataclass
class Scenario:
    """One named thing a demo can run.

    Args:
        name: the CLI's positional choice for it.
        show: draws the comparison -- runs it and narrates it, given the
            :class:`Console` and the parsed arguments.
    """

    name: str
    show: Callable[[Console, argparse.Namespace], None]


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

    # -- the shared plumbing ------------------------------------------------- #
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
        add_run_flags(parser)
        args = parser.parse_args(list(argv) if argv is not None else None)

        # Set the process config once from the CLI flags (an unset flag defers to
        # the TOSO_* env / default). Every Trace, controller adapter, resource
        # registry and the transport's collapse decision reads it ambiently.
        apply_run_flags(args, logger if self.own_logger else None)

        chosen = getattr(args, "scenario", None)
        if chosen is not None:
            by_name[chosen].show(console, args)
            return
        for scenario in scenarios:
            scenario.show(console, args)
        self.takeaway(console)
