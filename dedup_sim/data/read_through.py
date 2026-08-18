"""Read-through: a finished reader becomes a real directory source.

Ask, read, put, then commit, and the order is load-bearing. Asking first keeps the read
an unmodified ``client.get_batch``: the only thing routing it is the preference handed
to ``client_for``. The put is awaited before the commit so the directory observation
and the action agree that the batch landed. The commit settles the debt the fan-out
recorded and satisfies gates waiting on it; nothing else in the run knows the put
happened.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Optional

from proposed import ControlPlane, DataPlane, Deployment, endpoint
from torchstore.transport import Request

from ..control._sensor import Published

__all__ = ["ReadThroughPlane"]


class ReadThroughPlane(DataPlane):
    """Publish what a reader just fetched into the reader's own volume.

    Args:
        trace: where to record that a reader finished. Records only; no metric turns
            on it.
    """

    def __init__(self, *, trace: Any = None) -> None:
        self.trace = trace
        # Filled by attach().
        self.deployment: Optional[Deployment] = None

    def attach(self, deployment: Deployment) -> None:
        """Keep the deployment whose clients this plane puts through."""
        self.deployment = deployment

    @endpoint
    async def read_through(
        self,
        requester: str,
        entries: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``requester`` reads ``entries``, then keeps them; answers with the batch.

        Control answers only once the volume it named is usable, so the ``get`` has
        nothing to wait for here.
        """
        batch = dict(entries)
        if not batch:
            raise ValueError("read_through requires at least one entry")
        requests = tuple(
            Request.from_any(key, value).meta_only()
            for key, value in batch.items()
        )
        control: ControlPlane = self.deployment.control_plane_handle
        selection = await control.sources.call_one(requests, requester)
        results = await self.deployment.client_for(
            requester, prefer=selection.sources
        ).get_batch(batch)
        if self.trace is not None:
            self.trace.record(
                asyncio.get_running_loop().time(), "burst", f"reader {requester} done"
            )
        # A second vend, with no preference: a put chooses its own volume (the
        # co-located one), and leaving the read's preference bound would say otherwise.
        await self.deployment.client_for(requester).put_batch(results)
        await self.deployment.dispatcher_handle.dispatch.call_one(
            Published(requester)
        )
        return results
