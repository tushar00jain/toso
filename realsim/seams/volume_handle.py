"""What a caller holds to reach a storage volume: :class:`LocalVolumeHandle`.

The **client** side, and only that -- the third of the pair this package builds for
each service (see :mod:`realsim.seams.controller_handle`,
:mod:`realsim.seams.coordinator_handle`). A caller does not hold a storage volume; it
holds a reference to one, and Monarch's reference is not the same shape as the actor:
``@endpoint`` turns each method into an ``EndpointProperty``, so the attribute
resolves to an endpoint object and the call reads

    await volume.put.call(transport_buffer, requests)

which is what real client and transport code writes (``torchstore/client.py``,
``torchstore/transport/``). This class is that reference, for a service living in
this process instead of another one: an endpoint per method of
:class:`realsim.seams.volume_service.VolumeService`.

In a deployment there is nothing here to keep. ``StorageVolume.spawn`` returns
Monarch's own handle, whose endpoints do the same thing over the wire; this class is
the `[S]` that disappears. Which is also why it does not implement
:class:`proposed.deployment.StorageVolume`: the *service* implements that, and a
reference to a service is a different shape. Fusing the two -- one object holding the
store, the residency, the capacity rule *and* offering endpoints -- is what this file
used to be, and it made "is this the client side or the server side?" unanswerable
about the volume in exactly the way it once was about the directory.

Cost
----
The hop is free (``ServiceHop()`` with no rtt), so awaiting one of these endpoints
never suspends and a run is byte-identical to calling the service directly. That is
deliberate and not an oversight: what a client↔volume call really costs is *fabric*,
and the transport seam already charges it per put/get off the run's
``MachineProfile``. Charging a service hop here as well would price the same wire
twice.
"""

from __future__ import annotations

from typing import Any

from realsim.seams.link import LocalEndpoint, ServiceHop

__all__ = ["LocalVolumeHandle"]


class LocalVolumeHandle:
    """A reference to a :class:`VolumeService` living in this process.

    Args:
        service: the storage service this refers to.
    """

    def __init__(self, service: Any) -> None:
        self.service = service
        # One hop shared by every endpoint: they are all the same boundary. Free,
        # for the reason in this module's docstring -- the fabric is charged by the
        # transport seam, not here.
        self.hop = ServiceHop()
        self.put = LocalEndpoint(service.put, self.hop)
        self.get = LocalEndpoint(service.get, self.hop)
        self.handshake = LocalEndpoint(service.handshake, self.hop)
        self.touch = LocalEndpoint(service.touch, self.hop)
        self.delete = LocalEndpoint(service.delete, self.hop)
        self.delete_batch = LocalEndpoint(service.delete_batch, self.hop)
        self.reset = LocalEndpoint(service.reset, self.hop)

    @property
    def volume_id(self) -> str:
        """Which volume this refers to -- a handle knows its own target."""
        return self.service.volume_id
