"""The dedup control plane: which peer serves a reader, and when.

One module, one decision. :class:`~dedup_sim.control.routing.DedupPolicy` is a
:class:`proposed.policy.Policy`: it is handed a read-only
:class:`~proposed.view.View` and returns a ranked source plus a readiness gate. It
holds no client, no volume and no deployment, and it never executes anything -- the
read-through write that makes its answer come true lives in
:mod:`dedup_sim.data`.
"""

from .routing import DedupPolicy

__all__ = ["DedupPolicy"]
