"""The single substitution point for ``create_transport_buffer``.

The real ``LocalClient`` reaches the transport through a **process-wide module
global**: it does ``from torchstore.transport import create_transport_buffer`` at
import time, so the only place a sim can substitute the in-memory transport is
the bound name on the ``torchstore.client`` module object (see
``docs/realsim_design.md`` recommendation 2 for the upstream fix).

Because that global is process-wide, *every* substitution in this repo must go
through this module. Three call sites used to patch it independently -- the
single-client :class:`~realsim.adapters.real_client.RealClientAdapter`, a read
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
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

import torchstore.client  # noqa: F401  (ensure the submodule is in sys.modules)

from sim_common.topology import Endpoint

__all__ = [
    "bind_source",
    "bind_prefer",
    "current_prefer",
    "current_source",
    "current_owner",
    "installed",
]

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

# Which volumes the calling client would rather be served by, best first --
# the source preference ``locate_volumes`` applies to the answer it is about to
# give (:func:`proposed.selector.prefer`). ``None`` is no preference, hence the
# directory's own order.
#
# A binding rather than an argument because the real ``LocalClient.get`` has no
# such parameter: upstream this is one optional argument on the read path, applied
# to the located map before ``_build_volume_requests`` picks a volume per key. The
# simulator carries it here so an *unmodified* client can be handed a preference
# at all -- see :func:`bind_prefer`.
_current_prefer: "contextvars.ContextVar[Optional[Tuple[str, ...]]]" = (
    contextvars.ContextVar("realsim_current_prefer", default=None)
)

# The object currently holding the process-wide patch (``None`` == nobody).
_owner: Optional[Any] = None


def bind_source(endpoint: Endpoint) -> None:
    """Bind the source endpoint for the calling coroutine's transfers."""
    _current_src.set(endpoint)


def bind_prefer(sources: Optional[Sequence[str]]) -> None:
    """Prefer ``sources``, best first, for the reads the calling coroutine makes.

    A *value* the caller was handed by whoever decides -- a data plane asks its
    control plane who should serve a key and binds the answer here before driving the
    client (:meth:`realsim.mesh.Mesh.client_for`). Nothing in the store consults
    anybody as a result: ``locate_volumes`` applies what it finds bound and no more.

    ``None`` clears it, so a client vended without a preference reads exactly as
    an unrouted one does.
    """
    _current_prefer.set(tuple(sources) if sources is not None else None)


def current_prefer() -> Optional[Tuple[str, ...]]:
    """The sources the calling coroutine prefers, or ``None`` if it named none.

    Unlike :func:`current_source` this returns ``None`` rather than raising: a
    read with no preference is the ordinary read, and the directory's own order is
    always a correct answer -- whereas an unbound *source* would silently misprice
    a transfer.
    """
    return _current_prefer.get()


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
