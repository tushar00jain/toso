"""A sensor an application's hosts report into, off-actor: :class:`SensorService`.

The **server** side of a sensor a control plane decides against, and the fourth of
the pair this package builds for each service
(:mod:`realsim.seams.controller_service`,
:mod:`realsim.seams.control_plane_service`,
:mod:`realsim.seams.volume_service`). In a deployment this is a Monarch actor: a
process holding the sensor, receiving the facts its hosts report, and folding them.
Here it is a plain object holding that same sensor in this process, with the same
member, receiving ordinary calls instead of messages.

It implements :class:`proposed.deployment.NotifiedSensor` -- the surface declared
there, which is one write -- by holding the application's sensor and forwarding to
it. The split is the same one the directory makes: the thing that *is* a sensor is
the sensor, and the thing a host *holds* is
:class:`realsim.seams.sensor_handle.LocalSensorHandle`, a different shape (endpoints)
for a different reason (it stands in for the process boundary).

What crosses here and what does not
-----------------------------------
This endpoint carries what a *host* reports, which is why only a
``NotifiedSensor`` is fronted at all: a sensor the deciding plane alone writes needs
no service and gets none. The control plane deciding against this one reads it, and
writes its own decisions into it, by plain local call -- the same co-location a
:class:`~proposed.selector.KeySelector` has with the directory it senses through
``locate_raw``, and for the same reason: a decision formed against a read that could
suspend is a decision formed against a picture that changed halfway through.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SensorService"]


class SensorService:
    """An application's sensor, as a service another process can reach.

    Args:
        sensor: the application's :class:`proposed.deployment.NotifiedSensor` -- the
            object the facts are folded into. This service holds it and forwards;
            it holds nothing itself.
    """

    def __init__(self, sensor: Any) -> None:
        self.sensor = sensor

    # -- proposed.deployment.NotifiedSensor --------------------------------- #
    async def notify(self, fact: Any) -> None:
        await self.sensor.notify(fact)
