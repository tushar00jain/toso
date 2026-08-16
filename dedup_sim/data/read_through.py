"""Read-through: a finished reader becomes a real directory source.

:class:`ReadThroughPlane` is dedup's whole executing half, and it is three calls in
one member: ask the control plane who should serve this key, read from what it named
(an ordinary ``client.get``), and put what came back into this reader's own volume --
an ordinary ``client.put``, a zero-fabric local write. What that put made true is then
one action this reader commits (:class:`~proposed.dispatch.Stored`): the plane's own
fan-out settles the debt it owed, and whoever was parked on it wakes and re-reads.

The order is the whole design. Asking first is what makes the store's part of it a
value it was handed rather than a decision it makes: the read is an unmodified
``client.get`` and the only thing routing it is the preference passed to
``client_for``. Awaiting the put before dispatching is what makes a woken reader's
re-read find this one: the directory is written inside the put, so the commit that
wakes anybody happens after it. Committing at all is what makes the next reader's
answer arrive -- nothing else in the run knows that this put happened.

Both halves in one member, which is the shape ``kvcache_sim``'s serving host has
too: a data plane makes its own store calls and owns the order of them. Nothing is
handed in but the reader and the key.

Nothing here decides anything -- which source, and when, is ``control/``'s answer;
this is the actuation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from proposed import ControlPlane, DataPlane, Deployment, Stored

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
        # Filled by attach(): neither the readers' clients nor the ports they reach
        # exist before the deployment does.
        self.deployment: Optional[Deployment] = None

    def attach(self, deployment: Deployment) -> None:
        """Keep the deployment whose clients this plane puts through.

        Two ports are read off it where they are used, which are the two things a host
        says to control: the question goes to the plane, and the action it commits goes
        to the dispatcher.
        """
        self.deployment = deployment

    async def read_through(self, requester: str, key: Optional[str] = None) -> Any:
        """``requester`` reads ``key``, then keeps what it read; answers with the read.

        The ``get`` is served by whichever volume control named -- it answers only
        once that volume is usable, so there is nothing to wait for here -- and the
        ``put`` is what makes this reader a source for the next one, readable from the
        moment it returns and waited on until the action after it is committed.
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
