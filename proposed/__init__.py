"""The surface proposed for torchstore itself.

Everything in this package is a component a real deployment would need but
torchstore does not have today. It is kept apart from ``realsim`` so the upstream
ask is legible at a glance: ``realsim`` is scaffolding that disappears outside the
simulator, ``proposed`` is the design being argued for.

* :mod:`proposed.policy` -- :class:`~proposed.policy.Policy`
  (``select`` / ``notice``) and :class:`~proposed.policy.Selection`. A controller
  consults a policy inside ``locate_volumes`` to decide *which* volume serves a
  requester, and may withhold the answer until that volume is usable. The default
  :class:`~proposed.policy.NaivePolicy` returns the directory's own answer, so an
  installed policy changes nothing until one is written.
* :mod:`proposed.coordinator` -- :class:`~proposed.coordinator.Coordinator`, the
  other kind of control plane: an *application's*, deciding about its own resources
  from facts only it can see. Two members, ``decide`` and ``observe``, with the
  questions carried as payloads the application defines -- so a second application
  reuses the shape (and the seam that carries it) instead of inventing one.
* :mod:`proposed.deployment` -- :class:`~proposed.deployment.Controller`, the
  directory surface a caller reaches (torchstore names this type but never declares
  it: ``api.py`` annotates the spawned handle as the actor class and ``LocalClient``
  takes it unannotated). The difference between it and torchstore's class is the
  ask: the policy hook inside ``locate_volumes``, and ``locate_raw``, the same read
  without it. Beside it, :class:`~proposed.deployment.Coordinator` declares the
  *caller's* side of the surface above -- what a reference to a running coordinator
  offers, endpoint per member. Two shapes, so two types with one name each in their
  own module: ``from proposed import Coordinator`` is the one you *write*, and
  ``proposed.deployment.Coordinator`` the one you *call*;
* :mod:`proposed.view` -- :class:`~proposed.view.View`, the read-only observation
  a controller hands a policy: who holds a key, where volumes are, what time it is.
  It reads a :class:`~proposed.deployment.Controller`, through ``locate_raw``.

Import rule, enforced by ``realsim/tools/check_contract.py``: **this package may
not import anything at all** -- not ``realsim``, not a capability, not even
``sim_common``. That is what keeps it honest: a contract that needed the simulator
underneath it could not survive outside the harness. The locality types live here
rather than in ``sim_common`` for the same reason; only the *cost* of a tier is
simulation.

The gaps each piece answers are listed in the design doc's "What torchstore is
missing" section.
"""

# Re-export the contract surface so callers import from the package directly.
from .coordinator import Coordinator
from .cost import TransferCost
from .deployment import Controller, Deployment
from .plane import ControlPlane, DataPlane
from .policy import DecisionLog, Policy, Selection
from .topology import Endpoint, locality, Tier, TIER_LABEL
from .view import View

__all__ = [
    # the torchstore ask
    "Policy",
    "Selection",
    "DecisionLog",
    "View",
    "Controller",
    "Coordinator",
    "Endpoint",
    "Tier",
    "TIER_LABEL",
    "locality",
    # ports the application depends on
    "Deployment",
    "ControlPlane",
    "DataPlane",
    "TransferCost",
]
