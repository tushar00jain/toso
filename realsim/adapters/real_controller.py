"""Construct + drive a TorchStore controller off-actor.

Both controller implementations subclass ``monarch.actor.Actor`` but their
constructors only set plain attributes and do not touch Monarch, so they construct
off-actor. We deliberately **do not** call the ``init`` ``@endpoint``: it needs a
live Monarch storage-volume mesh
(``strategy.set_storage_volumes`` calls ``storage_volumes.get_id.call()``). The
directory logic only requires ``is_initialized``, which ``init`` sets last; the
adapter sets it directly so ``assert_initialized`` passes.

The ambient ``controller_backend`` selects the controller class at this module's
single construction seam. ``legacy`` retains two directory backings chosen by
``shim=``:

* the DEFAULT keeps the real ``Trie`` directory -- maximum fidelity;
* ``shim=True`` swaps only ``keys_to_storage_volumes`` for a
  :class:`~realsim.seams.dict_directory.DictDirectory` (a plain ``dict`` with the
  same surface), so large runs skip the per-key trie tax while **every** bit of
  ``Controller`` decision logic still runs the real code. Metrics are
  byte-identical between the two (the divergence-gate tests assert this), because
  the only difference is the container behind the same ``Mapping`` interface, and
  directory *iteration order* is never consumed by any measured metric.

The ``indexed`` controller owns its logical-region directory, so the legacy dict
shim does not apply to it. :func:`make_controller_adapter` reads both choices from
ambient config at construction.
"""

from __future__ import annotations

from typing import Optional

from sim_common import config

from realsim.seams.controller_handle import LocalControllerHandle
from realsim.seams.controller_service import ControllerService
from realsim.seams.link import ServiceHop
from realsim.seams.dict_directory import DictDirectory
from torchstore.controllers import get_controller_class

__all__ = ["RealControllerAdapter", "make_controller_adapter"]


class RealControllerAdapter:
    """Owns the selected controller and the endpoint handle in front of it.

    Args:
        hop: what reaching this controller costs (see
            :class:`~realsim.seams.link.ServiceHop`). ``None`` is a free hop.
        shim: swap the ``Controller``'s ``keys_to_storage_volumes`` for a
            :class:`~realsim.seams.dict_directory.DictDirectory` instead of the
            real ``Trie``. Everything else stays real -- the ``Controller``'s own
            decision logic (``_notify_put`` / ``_notify_delete`` /
            ``_is_dtensor_fully_committed``, and the ``locate_volumes`` body
            mirrored in :class:`LocalControllerHandle`) runs unchanged over it, so
            metrics stay byte-identical while the run skips the per-key trie tax.
            Ignored when ``controller_backend == "indexed"`` because that
            controller owns its directory structure.

    Attributes:
        controller: the real ``Controller`` instance.
        service: the :class:`ControllerService` around it -- the server side, which
            holds the endpoint bodies.
        handle: a :class:`LocalControllerHandle` referring to that service -- the
            client side, which is what a caller holds.
    """

    def __init__(self, hop: Optional[ServiceHop] = None, *, shim: bool = False) -> None:
        self.backend = config.current().controller_backend
        controller_class = get_controller_class(self.backend)
        self.controller = controller_class()
        if shim and self.backend == "legacy":
            # The Trie built by Controller.__init__ is discarded here: a one-time
            # construction cost, not the per-key tax the shim removes.
            self.controller.keys_to_storage_volumes = DictDirectory()
        # Mirror the tail of controller init: we skip the @endpoint (it needs a
        # Monarch mesh) and only need the directory marked initialized.
        self.controller.is_initialized = True
        self.service = ControllerService(self.controller)
        self.handle = LocalControllerHandle(self.service, hop=hop)

    @property
    def shimmed(self) -> bool:
        """Whether this controller's directory is the dict shim."""
        return isinstance(
            getattr(self.controller, "keys_to_storage_volumes", None), DictDirectory
        )


def make_controller_adapter(
    real_directory: Optional[bool] = None,
) -> RealControllerAdapter:
    """Build the controller adapter with the directory backing a run selects.

    Args:
        real_directory: explicit override. ``True`` -> the real ``Trie``;
            ``False`` -> the dict shim. ``None`` (default) defers to the ambient
            :data:`sim_common.config.SimConfig.real_directory` flag, which itself
            defaults to ``True`` -- so the faithful real directory is the default
            everywhere unless a run explicitly opts into the shim. This setting
            applies only to the ``legacy`` backend.
    """
    use_real = (
        config.current().real_directory if real_directory is None else real_directory
    )
    # What reaching this controller costs. Resolved once, here, because this is
    # the one place a run's controller is built.
    hop = ServiceHop(config.current().controller_rtt)
    return RealControllerAdapter(hop, shim=not use_real)
