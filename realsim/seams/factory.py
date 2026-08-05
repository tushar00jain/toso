"""The single substitution point for ``create_transport_buffer``.

The real ``LocalClient`` reaches the transport through a **process-wide module
global**: it does ``from torchstore.transport import create_transport_buffer`` at
import time, so the only place a sim can substitute the in-memory transport is
the bound name on the ``torchstore.client`` module object (see
``docs/realsim_design.md`` recommendation 2 for the upstream fix).

Because that global is process-wide, *every* substitution in this repo must go
through this module. Three call sites used to patch it independently -- the
single-client :class:`~realsim.adapters.real_client.RealClientAdapter`, the read
coordinator, and ``kvcache_sim``'s cluster -- each with its own save/restore and
(for the latter two) its own :class:`~contextvars.ContextVar` for the calling
client's source endpoint. Nothing stopped two of them being active at once, and
the failure was silent: the inner install wins, the outer's contextvar is never
read, and transfers are charged the wrong source locality (or
:meth:`current_source` raises ``LookupError``) while the outer owner keeps
recording metrics as if it were still driving. Consolidating here gives:

* **one** ``_current_src`` contextvar, so any owner's ``bind_source`` is the one
  the installed factory reads; and
* **one** owner at a time, enforced by :func:`installed` -- a second install
  raises instead of silently shadowing the first.

Determinism: ``asyncio`` copies the current context into each task it creates, so
a source bound inside a reader/instance coroutine is task-local and the lookup is
deterministic under the virtual-clock engine.
"""

from __future__ import annotations

import contextvars
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import torchstore.client  # noqa: F401  (ensure the submodule is in sys.modules)

from sim_common.topology import Endpoint

# The real ``torchstore.client`` submodule object. It is shadowed on the
# ``torchstore`` package by a ``client`` *function*, so it must be fetched from
# ``sys.modules`` rather than by attribute access on the package.
_CLIENT_MODULE = sys.modules["torchstore.client"]

# The source endpoint of the client whose operation is currently running. The
# installed factory reads it to charge the right locality; owners set it with
# :func:`bind_source` inside the calling coroutine.
_current_src: "contextvars.ContextVar[Endpoint]" = contextvars.ContextVar(
    "realsim_current_src_endpoint"
)

# The object currently holding the process-wide patch (``None`` == nobody).
_owner: Optional[Any] = None


def bind_source(endpoint: Endpoint) -> None:
    """Bind the source endpoint for the calling coroutine's transfers."""
    _current_src.set(endpoint)


def current_source() -> Endpoint:
    """Return the source endpoint bound for the calling coroutine.

    Raises ``LookupError`` if no owner bound one -- a bug in the caller, not a
    condition to paper over with a default (a wrong default would silently
    misprice every transfer).
    """
    return _current_src.get()


def current_owner() -> Optional[Any]:
    """Return the object currently holding the patch, or ``None``."""
    return _owner


@contextmanager
def installed(
    factory: Callable[[Any], Any], *, owner: Any
) -> Iterator[None]:
    """Substitute ``create_transport_buffer`` with ``factory`` for the block.

    Args:
        factory: called with a ``StorageVolumeRef``, returns a transport buffer.
        owner: the object claiming the patch, used to report a conflict.

    Raises:
        RuntimeError: if another owner already holds the substitution. The
            global is process-wide, so overlapping installs cannot both work --
            surfacing the conflict beats silently charging the wrong source.
    """
    global _owner
    if _owner is not None:
        raise RuntimeError(
            "create_transport_buffer is already substituted by "
            f"{type(_owner).__name__}; {type(owner).__name__} cannot install "
            "over it. The binding is a process-wide module global, so installs "
            "must be sequential, not nested -- scope the outer block to end "
            "before the inner one begins, or drive both clients from one "
            "realsim.mesh.Mesh."
        )
    _owner = owner
    original = _CLIENT_MODULE.create_transport_buffer
    _CLIENT_MODULE.create_transport_buffer = factory
    try:
        yield
    finally:
        _CLIENT_MODULE.create_transport_buffer = original
        _owner = None
