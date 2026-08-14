"""What a transfer will cost: the estimate a routing decision needs.

A control plane cannot choose between "pull this from a peer" and "recompute it
locally" without pricing the pull. That estimate is not something a capability
should hard-code, and it is not something the simulator should hand it directly --
otherwise every scheduler is written against one cost model and cannot be lifted
out of the harness.

So the run supplies it. :data:`TransferCost` is the whole surface: given two volume
ids and a byte count, how long. ``sim_common`` builds one from a ``MachineProfile``
and a topology; a deployment would build one from measured fabric numbers. A control
plane reaches it off the view it was attached to
(:meth:`proposed.view.View.transfer_cost`), which is where every other run-supplied
read lives.

It pairs with :mod:`proposed.topology` — knowing *where* a volume is (gap 4) is
what makes an estimate possible at all.
"""

from __future__ import annotations

from typing import Callable

from proposed.deployment import VolumeId

__all__ = ["TransferCost"]

#: Seconds to move ``nbytes`` from one volume to another.
#:
#: One function, so it is one: a class around a single call would be a place for a
#: second one to accrete. Whatever answers it must agree with what the data plane is
#: actually charged, or a scheduler routes against a fiction --
#: ``kvcache_sim/tests/test_cost_parity.py`` pins that agreement for the simulator's.
TransferCost = Callable[[VolumeId, VolumeId, int], float]
