"""What a transfer will cost: the estimate a routing decision needs.

A control plane cannot choose between "pull this from a peer" and "recompute it
locally" without pricing the pull. That estimate is not something a capability
should hard-code, and it is not something the simulator should hand it directly --
otherwise every scheduler is written against one cost model and cannot be lifted
out of the harness.

So it is a protocol. :class:`TransferCost` is the whole surface: given two volume
ids and a byte count, how long. ``sim_common`` implements it from a
``MachineProfile`` and a topology; a deployment would implement it from measured
fabric numbers.

It pairs with :mod:`proposed.topology` — knowing *where* a volume is (gap 4) is
what makes an estimate possible at all.
"""

from __future__ import annotations

from typing import Protocol


class TransferCost(Protocol):
    """Predicts the time to move ``nbytes`` between two volumes."""

    def get_time(self, src_id: str, dst_id: str, nbytes: int) -> float:
        """Seconds to move ``nbytes`` from ``src_id`` to ``dst_id``.

        Must agree with what the data plane is actually charged, or a scheduler
        will route against a fiction. ``kvcache_sim/tests/test_cost_parity.py``
        pins that agreement for the simulator's implementation.
        """
        ...
