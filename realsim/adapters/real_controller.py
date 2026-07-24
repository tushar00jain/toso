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
"""

from __future__ import annotations

from realsim.seams.controller_handle import FakeControllerHandle
from torchstore.controller import Controller


class RealControllerAdapter:
    """Owns a real ``Controller`` and its off-actor handle.

    Attributes:
        controller: the real ``Controller`` instance (real ``Trie`` directory).
        handle: a :class:`FakeControllerHandle` exposing the controller's actor
            surface to the client.
    """

    def __init__(self) -> None:
        self.controller = Controller()
        # Mirror the tail of Controller.init: we skip the @endpoint (it needs a
        # Monarch mesh) and only need the directory marked initialized.
        self.controller.is_initialized = True
        self.handle = FakeControllerHandle(self.controller)
