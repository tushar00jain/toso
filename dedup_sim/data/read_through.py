"""Read a batch through ranked sources and publish it locally."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Optional

from proposed import ControlPlane, DataPlane, Deployment, endpoint
from torchstore.transport import Request

from ..control._sensor import Published

__all__ = ["ReadThroughPlane"]


class ReadThroughPlane(DataPlane):
    """Publish what a reader fetched into its own volume."""

    def __init__(self, *, trace: Any = None) -> None:
        self.trace = trace
        self.deployment: Optional[Deployment] = None

    def attach(self, deployment: Deployment) -> None:
        self.deployment = deployment

    @endpoint
    async def read_through(
        self,
        requester: str,
        entries: Mapping[str, Any],
    ) -> dict[str, Any]:
        batch = dict(entries)
        if not batch:
            raise ValueError("read_through requires at least one entry")
        requests = tuple(
            Request.from_any(key, value).meta_only() for key, value in batch.items()
        )
        control: ControlPlane = self.deployment.control_plane_handle
        while True:
            plan = await control.sources.call_one(requests, requester)
            try:
                results = await self.deployment.client_for(
                    requester, prefer=plan.sources
                ).get_batch(batch)
            except KeyError:
                await self.deployment.dispatcher_handle.dispatch.call_one(
                    Published(plan.publication)
                )
                continue
            if self.trace is not None:
                self.trace.record(
                    asyncio.get_running_loop().time(),
                    "burst",
                    f"reader {requester} done",
                )
            await self.deployment.client_for(requester).put_batch(results)
            await self.deployment.dispatcher_handle.dispatch.call_one(
                Published(plan.publication)
            )
            return results
