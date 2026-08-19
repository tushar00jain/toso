"""Read-through: a finished reader becomes a real directory source.

Ask, read, put, then commit, and the order is load-bearing. ``get_batch`` runs the
fetch planner control planned with (:class:`~proposed.planner.GreedyClient`) against
the per-key sources control selected. The put is awaited before the commit so the
directory observation and the action agree that the batch landed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Optional

from proposed import ControlPlane, DataPlane, Deployment, endpoint, GreedyClient
from proposed.selector import prefer
from torchstore.transport import Request

from ..control._sensor import Published

__all__ = ["ReadThroughPlane"]


class _Locate:
    def __init__(self, locate, by_key) -> None:
        self._locate = locate
        self._by_key = by_key

    async def call_one(self, keys):
        located = await self._locate.call_one(keys)
        return {
            key: prefer({key: located[key]}, self._by_key.get(key, ()))[key]
            for key in keys
        }


class _ScopedController:
    def __init__(self, controller, by_key) -> None:
        self.locate_volumes = _Locate(controller.locate_volumes, by_key)


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
            Request.from_any(key, value).meta_only() for key, value in batch.items()
        )
        control: ControlPlane = self.deployment.control_plane_handle
        plan = await control.sources.call_one(requests, requester)
        client = self.deployment.client_for(requester)
        # The same planner control dry-ran the plan with, so the two agree on which
        # source serves which region given the same located map.
        routed = GreedyClient(
            _ScopedController(client._controller, plan.by_key), client.strategy
        )
        results = await routed.get_batch(batch)
        if self.trace is not None:
            self.trace.record(
                asyncio.get_running_loop().time(), "burst", f"reader {requester} done"
            )
        # A second vend, with no preference: a put chooses its own volume (the
        # co-located one), and leaving the read's preference bound would say otherwise.
        await self.deployment.client_for(requester).put_batch(results)
        # What landed, not what was asked for: a reader whose fetch came back short
        # still publishes, and a peer waiting on the missing keys must go on waiting.
        await self.deployment.dispatcher_handle.dispatch.call_one(
            Published(requester, frozenset(results))
        )
        return results
