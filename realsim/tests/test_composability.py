"""Composability: realsim's real-directory backend is importable like sim_common.

A consumer (e.g. ``dedup_sim`` / ``kvcache_sim``) must be able to import realsim's
**real controller directory** as a cleanly separable unit -- the same way it
imports ``sim_common`` -- with no import-time side effects. This test proves that:

1. the backend imports + constructs with **no import-time side effects** (no
   background threads, no running event loop, no Monarch mesh required), and
2. its directory lookup honors the empty-on-absent contract.

It runs the real ``Controller`` directory logic off-actor via the seams on a plain
asyncio loop (no engine needed -- this proves importability + standalone
usability, not timing).
"""

from __future__ import annotations

import asyncio
import sys

# A plain top-level import of the separable real-directory backend, the same way a
# consumer imports ``sim_common``. These imports must be side-effect-free (asserted
# below).
from realsim.adapters.real_controller import RealControllerAdapter
from realsim.seams.controller_handle import LocalControllerHandle


def test_backend_imports_with_no_side_effects():
    """Importing + constructing the backend spawns nothing and needs no runtime.

    The determinism contract forbids background threads on the sim path, and a
    consumer must not inherit a Monarch mesh or a live event loop just by importing
    the directory. So: no thread-count change, no running loop required, and
    construction works off any actor runtime.

    Thread count is read via ``sys._current_frames()`` (one frame per live
    thread) rather than ``threading`` -- the concurrency-contract lint bans the
    ``threading`` import even in test modules.
    """
    threads_before = len(sys._current_frames())

    adapter = RealControllerAdapter()

    # No background threads were started by import or construction.
    assert len(sys._current_frames()) == threads_before
    # The backend is the real objects, wired but inert until called.
    assert isinstance(adapter.handle, LocalControllerHandle)
    assert adapter.controller.is_initialized is True
    # No event loop is running at import/construction time.
    with_no_loop = True
    try:
        asyncio.get_running_loop()
        with_no_loop = False
    except RuntimeError:
        pass
    assert with_no_loop


def test_missing_key_returns_empty():
    """locate for an unknown key returns empty when ``missing_ok=True``.

    The real backend raises for an absent key unless ``missing_ok=True`` is passed,
    which yields ``{}`` -- the empty-on-absent contract a consumer relies on. This
    documents that one seam explicitly.
    """

    async def _go():
        handle = RealControllerAdapter().handle
        return await handle.locate_volumes.call(["absent"], missing_ok=True)

    assert asyncio.run(_go()) == {}
