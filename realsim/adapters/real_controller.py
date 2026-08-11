"""Construct + drive a real ``Controller`` off-actor.

The real ``Controller`` (``torchstore/controller.py``) subclasses
``monarch.actor.Actor`` but its ``__init__`` only sets plain attributes (a
``Trie`` directory + flags) and does not touch Monarch, so ``Controller()``
constructs fine off-actor. We deliberately **do not** call the real ``init``
``@endpoint``: it needs a live Monarch storage-volume mesh
(``strategy.set_storage_volumes`` calls ``storage_volumes.get_id.call()``). The
only state the directory logic requires is ``is_initialized``, which the real
``init`` sets last; we set it directly (mirroring the tail of ``Controller.init``)
so ``assert_initialized`` passes. All directory state and logic remain the real
object's.

Two directory backings are offered behind an identical handle surface:

* :class:`RealControllerAdapter` (the DEFAULT) keeps the real ``Trie`` directory
  -- maximum fidelity.
* :class:`ShimControllerAdapter` swaps only ``keys_to_storage_volumes`` for a
  :class:`~realsim.seams.dict_directory.DictDirectory` (a plain ``dict`` with the
  same surface), so large runs skip the per-key trie tax while **every** bit of
  ``Controller`` decision logic still runs the real code. Metrics are
  byte-identical between the two (the divergence-gate tests assert this), because
  the only difference is the container behind the same ``Mapping`` interface, and
  directory *iteration order* is never consumed by any measured metric.

:func:`make_controller_adapter` selects between them from the ambient
:data:`sim_common.config.SimConfig.real_directory` flag (with an explicit
override), so the choice threads through every construction site the same way
the other cross-cutting knobs do.
"""

from __future__ import annotations

from typing import Optional, Union

from sim_common import config

from realsim.seams.controller_handle import FakeControllerHandle
from realsim.seams.link import ServiceHop
from realsim.seams.dict_directory import DictDirectory
from torchstore.controller import Controller

__all__ = [
    "RealControllerAdapter",
    "ShimControllerAdapter",
    "make_controller_adapter",
]


class RealControllerAdapter:
    """Owns a real ``Controller`` and its off-actor handle.

    Attributes:
        controller: the real ``Controller`` instance (real ``Trie`` directory).
        handle: a :class:`FakeControllerHandle` exposing the controller's actor
            surface to the client.
    """

    def __init__(self, hop: Optional[ServiceHop] = None) -> None:
        self.controller = Controller()
        # Mirror the tail of Controller.init: we skip the @endpoint (it needs a
        # Monarch mesh) and only need the directory marked initialized.
        self.controller.is_initialized = True
        self.handle = FakeControllerHandle(self.controller, hop=hop)


class ShimControllerAdapter:
    """Owns a real ``Controller`` whose directory is a lightweight ``dict``.

    Mirrors :class:`RealControllerAdapter`'s attributes (``.controller`` +
    ``.handle``) exactly, so it is interchangeable at every construction site. The
    only change from the real adapter is that ``keys_to_storage_volumes`` is a
    :class:`~realsim.seams.dict_directory.DictDirectory` instead of a ``Trie``:
    the real ``Controller`` decision logic (``_notify_put`` / ``_notify_delete`` /
    ``_is_dtensor_fully_committed`` / the ``locate_volumes`` body mirrored in
    :class:`FakeControllerHandle`) runs unchanged over it, so payoff metrics stay
    byte-identical while the run skips the per-key trie tax.

    Attributes:
        controller: the real ``Controller`` instance (dict-backed directory).
        handle: a :class:`FakeControllerHandle` exposing the controller's actor
            surface to the client (identical surface to the real adapter).
    """

    def __init__(self, hop: Optional[ServiceHop] = None) -> None:
        self.controller = Controller()
        # Swap only the directory container -- everything else stays real. The
        # Trie built by Controller.__init__ is discarded here (a one-time
        # construction cost, not the per-key tax the shim removes).
        self.controller.keys_to_storage_volumes = DictDirectory()
        # Mirror the tail of Controller.init (see RealControllerAdapter).
        self.controller.is_initialized = True
        self.handle = FakeControllerHandle(self.controller, hop=hop)


def make_controller_adapter(
    real_directory: Optional[bool] = None,
) -> Union[RealControllerAdapter, ShimControllerAdapter]:
    """Build the controller adapter selected by the directory-backing flag.

    Args:
        real_directory: explicit override. ``True`` -> real ``Trie`` directory
            (:class:`RealControllerAdapter`); ``False`` -> lightweight dict shim
            (:class:`ShimControllerAdapter`). ``None`` (default) defers to the
            ambient :data:`sim_common.config.SimConfig.real_directory` flag, which
            itself defaults to ``True`` -- so the faithful real directory is the
            default everywhere unless a run explicitly opts into the shim.

    Returns:
        A controller adapter exposing the same ``.controller`` / ``.handle``
        attributes regardless of backing.
    """
    use_real = config.current().real_directory if real_directory is None else real_directory
    # What reaching this controller costs. Resolved once, here, because this is
    # the one place a run's controller is built.
    hop = ServiceHop(config.current().controller_rtt)
    return RealControllerAdapter(hop) if use_real else ShimControllerAdapter(hop)
