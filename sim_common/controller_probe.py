"""Silenced probe for the real TorchStore ``Controller``.

Attempts to import the real controller for documentation/faithfulness. The
import (via Monarch) is noisy on stderr, so it is silenced -- the real
controller is not driven (its endpoints are ``@endpoint async`` Monarch-actor
methods that a single-threaded simulation cannot run); this only records
whether it is importable via :data:`HAVE_REAL`.
"""

from __future__ import annotations

import contextlib
import io
import os


def _probe_real_controller() -> bool:
    """Attempt the (silenced) real-controller import; return whether it succeeds."""
    try:  # pragma: no cover - depends on Monarch being installed
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            with open(os.devnull, "w") as _devnull:
                _saved = os.dup(2)
                os.dup2(_devnull.fileno(), 2)
                try:
                    from torchstore.controller import Controller as _RealController  # noqa: F401,E501
                finally:
                    os.dup2(_saved, 2)
                    os.close(_saved)
        return True
    except Exception:  # pragma: no cover
        return False


HAVE_REAL: bool = _probe_real_controller()
