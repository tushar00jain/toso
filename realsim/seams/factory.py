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
from typing import Any, Callable, Iterator, Optional

import torchstore.client  # noqa: F401  (ensure the submodule is in sys.modules)

from sim_common.topology import Endpoint

__all__ = [
    "bind_choice",
    "current_choice",
    "bind_source",
    "bind_requester",
    "current_requester",
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

# The *directory* identity (volume id) of that same client, for the controller's
# routing hook: a policy is asked "which source for this requester". Defaults to
# ``None`` so an unrouted drive simply gets the directory's own answer.
_current_requester: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "realsim_current_requester", default=None
)

# The object currently holding the process-wide patch (``None`` == nobody).
_owner: Optional[Any] = None


#: The source an application already chose for the current operation (GAP 1/2:
#: the controller cannot otherwise be told what the caller decided).
_current_choice: contextvars.ContextVar = contextvars.ContextVar(
    "toso_current_choice", default=None
)


def bind_source(endpoint: Endpoint) -> None:
    """Bind the source endpoint for the calling coroutine's transfers."""
    _current_src.set(endpoint)


def bind_requester(volume_id: Optional[str]) -> None:
    """Bind the *directory* identity of the client whose operation is running.

    The endpoint bound by :func:`bind_source` is the locality the cost model
    prices against; this is the same client's volume id in the real directory,
    which is what a routing policy needs to know who is asking. They are
    separate ids (a topology may name a node ``"r0"`` and its endpoint
    ``"volr0"``), so both are bound together by
    :meth:`realsim.mesh.Mesh.bind_source`.
    """
    _current_requester.set(volume_id)


def current_requester() -> Optional[str]:
    """The directory identity of the calling client, or ``None`` if unbound.

    Unlike :func:`current_source` this returns ``None`` rather than raising: a
    routing policy that cannot tell who is asking must fall back to the
    directory's own answer, which is always correct, whereas an unbound *source*
    would silently misprice a transfer.
    """
    return _current_requester.get()


def bind_choice(volume_id: Optional[str]) -> None:
    """Bind the source the caller already chose for this operation.

    The third identity a client call carries, alongside the locality endpoint and
    the requester volume id: *which holder the application decided to read from*.
    An installed policy receives it as ``chosen``. ``None`` clears it, so an
    operation that made no choice cannot inherit the previous one's.
    """
    _current_choice.set(volume_id)


def current_choice() -> Optional[str]:
    """The source bound by :func:`bind_choice`, or ``None`` if the caller had none."""
    return _current_choice.get()


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
