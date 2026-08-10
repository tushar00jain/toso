"""The dedup data plane: the read-through write that makes the routing true.

Dedup moves no bytes of its own -- the transfer is the reader's ordinary
``client.get``. All this plane does is the step *after* it: store what was just
fetched into the reader's own volume, which through the real ``client.put`` path
also registers the reader in the real directory. That registration is what
releases the next reader's withheld answer, so the chain/tree is a consequence of
this one call rather than a loop anywhere in the control plane.
"""

from .read_through import make_plane, ReadThroughPlane

__all__ = ["ReadThroughPlane", "make_plane"]
