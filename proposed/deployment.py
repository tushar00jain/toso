"""How application code reaches the store it is deployed against.

A capability's data plane calls ordinary torchstore APIs, but it has to get the
client from somewhere, and in a simulation "somewhere" is a harness object holding
many clients at once. Depending on that harness directly would make the data plane
unliftable: real code cannot import the simulator.

So the data plane asks for a :class:`Deployment` instead. In production this is
one process with one client and one controller; in the simulator it is the mesh,
which resolves the node and does the bookkeeping a real deployment would not need.
Either way the application code is the same.

The returned objects are deliberately untyped here: they are torchstore's own
``LocalClient`` and controller handle, and this package cannot import torchstore --
it is what torchstore would gain, not something layered on top of it. The
application typing them is the right place for that.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

__all__ = ["Deployment"]


class Deployment(Protocol):
    """The store, as application code sees it."""

    def client_for(self, node_id: str, *, source: Optional[str] = None) -> Any:
        """The torchstore client for ``node_id``, ready to be driven.

        A deployment that runs one node per process ignores the argument and
        returns its own client. A harness running many nodes in one process
        resolves the node and attributes the work to it.

        ``source`` names the volume the caller already chose to read from, for an
        application that routes before it fetches. It reaches an installed
        :class:`~proposed.policy.Policy` as ``chosen``; without it the directory
        answers for itself and the client takes whichever holder comes first,
        which need not be the one the caller priced.
        """
        ...

    @property
    def controller_handle(self) -> Any:
        """The controller's endpoint surface: the directory calls."""
        ...
