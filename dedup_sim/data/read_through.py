"""Read-through: a finished reader becomes a real directory source.

:class:`ReadThroughPlane` overrides one method of
:class:`realsim.plane.DataPlane`. After a reader's ``get`` lands, it drives the
real ``client.put`` into that reader's co-located volume: a zero-fabric local
write plus the real ``notify_put_batch``. The reader is now a genuine source in
the real directory, which is what
:class:`~dedup_sim.control.routing.DedupPolicy` was waiting on.

Nothing here decides anything -- the source assignment already happened in
``control/``; this is the actuation, and it is one ordinary store call.
"""

from __future__ import annotations

from typing import Any

from realsim.plane import DataPlane

__all__ = ["ReadThroughPlane", "make_plane"]


class ReadThroughPlane(DataPlane):
    """Publish what a reader just fetched into the reader's own volume.

    Args:
        mesh: the :class:`realsim.mesh.Mesh` holding the per-node real clients.
        key: the key the burst moves.
        value: the payload carrier to re-``put`` (a ``device="meta"`` tensor or a
            :class:`~realsim.seams.transport.TensorDescriptor`) -- allocation-free
            either way, exactly as the producer's put was.
    """

    def __init__(self, mesh: Any, key: str, value: Any) -> None:
        self.mesh = mesh
        self.key = key
        self.value = value

    async def after(self, item: Any, result: Any) -> None:
        """Real read-through: the reader stores the key into its own volume."""
        self.mesh.bind_source(item.id)
        await self.mesh.client(item.id).put(self.key, self.value)


def make_plane(mesh: Any, key: str, value: Any) -> ReadThroughPlane:
    """Factory matching ``realsim.scenarios.put_get.MakePlane``."""
    return ReadThroughPlane(mesh, key, value)
