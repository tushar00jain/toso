"""Read-through: a finished reader becomes a real directory source.

:class:`ReadThroughPlane` is dedup's whole executing half: a reader's ``get``,
then the real ``client.put`` into that reader's co-located volume -- a zero-fabric
local write plus the real ``notify_put_batch``. The reader is now a genuine source
in the real directory, which is what
:class:`~dedup_sim.control.routing.DedupKeySelector` was waiting on.

Both calls in one member, which is the shape ``kvcache_sim``'s serving host has
too: a data plane makes its own store calls and owns the order of them. Nothing is
handed in but the reader and the key -- ``client_for`` binds the caller before it
vends the client, so a plane needs nothing from the harness to read as that
reader.

Nothing here decides anything -- the source assignment already happened in
``control/``; this is the actuation, and it is one ordinary store call.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from proposed import DataPlane, Deployment

__all__ = ["ReadThroughPlane"]


class ReadThroughPlane(DataPlane):
    """Publish what a reader just fetched into the reader's own volume.

    The deployment arrives at :meth:`attach`, not in the constructor: a plane is
    built with its knobs and handed its ports once they exist, which is
    :class:`~proposed.plane.DataPlane`'s half of the two-phase shape the deciding
    half has too.

    Args:
        key: the key the burst moves, when a caller does not name one.
        value: the payload carrier to re-``put`` (a ``device="meta"`` tensor or a
            allocation-free descriptor) -- whatever the producer put, put back
            unchanged, so the read-through stores exactly what was read.
        trace: where to record that a reader finished. Optional and never
            load-bearing, the same terms a control plane keeps a
            :class:`~proposed.selector.DecisionLog` on.
    """

    def __init__(self, key: str, value: Any, *, trace: Any = None) -> None:
        self.key = key
        self.value = value
        self.trace = trace
        # Filled by attach(): the readers' clients do not exist before the
        # deployment does.
        self.deployment: Optional[Deployment] = None

    def attach(self, deployment: Deployment) -> None:
        """Keep the deployment whose clients this plane puts through."""
        self.deployment = deployment

    async def read_through(self, requester: str, key: Optional[str] = None) -> Any:
        """``requester`` reads ``key``, then keeps what it read; answers with the read.

        Two ordinary client calls in the order that makes the second one matter: the
        ``get`` is routed by whatever selector the controller has installed, and the
        ``put`` is what registers this reader as a source for the next one.
        """
        key = key if key is not None else self.key
        client = self.deployment.client_for(requester)
        result = await client.get(key)
        if self.trace is not None:
            self.trace.record(
                asyncio.get_running_loop().time(), "burst", f"reader {requester} done"
            )
        await client.put(key, self.value)
        return result
