"""What every demo's ``__main__`` shares: flags, config, logging, traces.

The three entry points (``python -m dedup_sim``, ``python -m kvcache_sim``,
``python -m realsim``) all take the same five run flags, all turn them
into the same ``config.configure`` call, and all dump a trace the same way. That
was copied three times; it lives here instead, so a new capability's demo is a
parser, a scenario and a renderer.

    parser = argparse.ArgumentParser(...)
    add_run_flags(parser)
    args = parser.parse_args(argv)
    apply_run_flags(args, logger)

Capability-specific flags (which scenario, how many readers) stay in the demo that
owns them -- only the run knobs are shared, because only they mean the same thing
everywhere.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from sim_common import config
from sim_common.report import configure_logging

__all__ = ["add_run_flags", "apply_run_flags", "log_trace"]


def add_run_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
    return parser


def apply_run_flags(
    args: argparse.Namespace, logger: Optional[logging.Logger] = None
) -> None:
    """Set the process config from ``args``, once, and configure logging.

    An unset flag defers to the ``TOSO_*`` env var / default, so this never
    overrides an ambient setting with "off". Pass ``logger`` when the demo's own
    logger must drop to DEBUG independently of the root (``realsim`` keeps the
    root at INFO so torchstore's own debug logging stays quiet).
    """
    config.configure(
        fingerprint=args.fingerprint or None,
        real_directory=False if args.shim_directory else None,
        contention=args.contention,
        collapse_charges=args.collapse_charges or None,
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


def log_trace(
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
