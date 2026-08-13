"""Read-through: a finished reader becomes a real directory source.

:class:`ReadThroughPlane` overrides one method of
:class:`proposed.plane.DataPlane`. After a reader's ``get`` lands, it drives the
real ``client.put`` into that reader's co-located volume: a zero-fabric local
write plus the real ``notify_put_batch``. The reader is now a genuine source in
the real directory, which is what
:class:`~dedup_sim.control.routing.DedupKeySelector` was waiting on.

Nothing here decides anything -- the source assignment already happened in
``control/``; this is the actuation, and it is one ordinary store call.
"""

from __future__ import annotations

from typing import Any

from proposed import DataPlane, Deployment

__all__ = ["ReadThroughPlane"]


class ReadThroughPlane(DataPlane):
    """Publish what a reader just fetched into the reader's own volume.

    Args:
        deployment: the :class:`~proposed.deployment.Deployment` the readers run
            against; it vends the client co-located with a reader.
        key: the key the burst moves.
        value: the payload carrier to re-``put`` (a ``device="meta"`` tensor or a
            allocation-free descriptor) -- whatever the producer put, put back
            unchanged, so the read-through stores exactly what was read.
    """

    def __init__(self, deployment: Deployment, key: str, value: Any) -> None:
        self.deployment = deployment
        self.key = key
        self.value = value

    async def after(self, requester: str, result: Any) -> None:
        """Real read-through: the reader stores the key into its own volume."""
        await self.deployment.client_for(requester).put(self.key, self.value)
