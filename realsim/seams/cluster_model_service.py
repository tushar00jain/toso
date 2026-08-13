"""An application's model of its cluster, off-actor: :class:`ClusterModelService`.

The **server** side of the model a control plane decides against, and the fourth of
the pair this package builds for each service
(:mod:`realsim.seams.controller_service`, :mod:`realsim.seams.placement_service`,
:mod:`realsim.seams.volume_service`). In a deployment this is a Monarch actor: a
process holding the model, receiving the facts its hosts report, and folding them.
Here it is a plain object holding that same model in this process, with the same
member, receiving ordinary calls instead of messages.

It implements :class:`proposed.deployment.ClusterModel` -- the surface declared
there, which is one write -- by holding the application's model and forwarding to
it. The split is the same one the directory makes: the thing that *is* a model is
the model, and the thing a host *holds* is
:class:`realsim.seams.cluster_model_handle.LocalClusterModelHandle`, a different
shape (endpoints) for a different reason (it stands in for the process boundary).

What crosses here and what does not
-----------------------------------
This endpoint carries what a *host* reports. The control plane holding the model
reads it, and writes its own decisions into it, by plain local call -- the same
co-location an installed :class:`~proposed.policy.KeySelector` has with the directory it
senses through ``locate_raw``, and for the same reason: a decision formed against a
read that could suspend is a decision formed against a picture that changed halfway
through.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ClusterModelService"]


class ClusterModelService:
    """An application's cluster model, as a service another process can reach.

    Args:
        model: the application's :class:`proposed.deployment.ClusterModel` -- the
            object the facts are folded into. This service holds it and forwards;
            it models nothing.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    # -- proposed.deployment.ClusterModel ----------------------------------- #
    async def notify(self, fact: Any) -> None:
        await self.model.notify(fact)
