"""Small helpers for rendering a simulation's console report.

Keeps the logging setup and section-header formatting in one place so a demo
entrypoint can route its digest through the ``logging`` module and print
banner-delimited sections consistently.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int) -> None:
    """Route log records to stdout as bare messages.

    ``force=True`` resets any root handlers a dependency may have installed at
    import time; otherwise ``basicConfig`` would no-op and the output would be
    silently dropped. The report is the product, so it goes to stdout.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def section(logger: logging.Logger, title: str) -> None:
    """Log a banner-delimited section header at INFO."""
    logger.info("\n%s", "=" * 72)
    logger.info(title)
    logger.info("=" * 72)
