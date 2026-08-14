"""What one dedup decision senses: :class:`FanoutView`.

Dedup senses one thing beside the directory, so there is no view to assemble out of
several: ``view.derived(FanoutView, fanout=...)``
(:meth:`~proposed.view.View.derived`) is the whole of it. The keyword is claimed here
and nowhere else -- one no view claims reaches :class:`~proposed.view.View`, which
takes none, and raises there.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from proposed import View

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._sensor import FanoutSensor

__all__ = ["FanoutView"]


class FanoutView(View):
    """The tree this plane has planned and the puts it is owed: :attr:`fanout`.

    Observed state as much as the directory is -- this plane's own record of its own
    decisions -- so whatever routes against it senses it here instead of being handed
    the sensor, and the plane writes a landed put back the same way.
    """

    def __init__(
        self,
        *ports: Any,
        fanout: Optional["FanoutSensor"] = None,
        **sensors: Any,
    ) -> None:
        super().__init__(*ports, **sensors)
        self._fanout = fanout

    @property
    def fanout(self) -> "FanoutSensor":
        """The planned tree and the outstanding debt.

        Raises like :meth:`~proposed.view.View.transfer_cost` does and for its reason:
        a view composed without that sensor cannot say who is folded in behind whom,
        and an empty tree is not one to invent -- it would route every requester to
        the origin and report no error.
        """
        if self._fanout is None:
            raise RuntimeError(
                "this view was composed without a fan-out sensor, so nothing here "
                "can read the read-through tree this plane has planned"
            )
        return self._fanout
