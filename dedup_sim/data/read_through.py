"""Read-through: a finished reader becomes a real directory source.

Ask, read, put, then commit, and the order is load-bearing. Asking first keeps the read
an unmodified ``client.get``: the only thing routing it is the preference handed to
``client_for``. The put is awaited before the commit so the directory observation and
the action agree that the key landed. The commit
(:class:`~proposed.dispatch.Stored`) settles the debt the fan-out recorded and
satisfies gates waiting on it; nothing else in the run knows the put happened.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from proposed import ControlPlane, DataPlane, Deployment, endpoint, Stored

__all__ = ["ReadThroughPlane"]


class ReadThroughPlane(DataPlane):
    """Publish what a reader just fetched into the reader's own volume.

    Args:
        key: the key the burst moves, when a caller does not name one.
        value: the payload carrier to re-``put``.
        trace: where to record that a reader finished. Records only; no metric turns
            on it.
    """

    def __init__(self, key: str, value: Any, *, trace: Any = None) -> None:
        self.key = key
        # Put back unchanged, so the read-through stores exactly what was read: a
        # ``device="meta"`` tensor or an allocation-free descriptor, never a copy.
        self.value = value
        self.trace = trace
        # Filled by attach().
        self.deployment: Optional[Deployment] = None

    def attach(self, deployment: Deployment) -> None:
        """Keep the deployment whose clients this plane puts through."""
        self.deployment = deployment

    @endpoint
    async def read_through(self, requester: str, key: Optional[str] = None) -> Any:
        """``requester`` reads ``key``, then keeps what it read; answers with the read.

        Control answers only once the volume it named is usable, so the ``get`` has
        nothing to wait for here.
        """
        key = key if key is not None else self.key
        control: ControlPlane = self.deployment.control_plane_handle
        selection = await control.sources.call_one([key], requester)
        result = await self.deployment.client_for(
            requester, prefer=selection.sources
        ).get(key)
        if self.trace is not None:
            self.trace.record(
                asyncio.get_running_loop().time(), "burst", f"reader {requester} done"
            )
        # A second vend, with no preference: a put chooses its own volume (the
        # co-located one), and leaving the read's preference bound would say otherwise.
        await self.deployment.client_for(requester).put(key, self.value)
        await self.deployment.dispatcher_handle.dispatch.call_one(
            Stored(requester, key)
        )
        return result
