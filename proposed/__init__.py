"""The surface proposed for torchstore itself.

Everything in this package is a component a real deployment would need but
torchstore does not have today. It is kept apart from ``realsim`` so the upstream
ask is legible at a glance: ``realsim`` is scaffolding that disappears outside the
simulator, ``proposed`` is the design being argued for.

* :mod:`proposed.selector` -- :class:`~proposed.selector.Selection` and the selection
  contracts that answer with one, each naming the subject it takes
  (``subject_type``). A :class:`~proposed.selector.KeySelector` takes *keys*, and is
  the only kind a controller installs inside ``locate_volumes``, where it decides
  which volume serves a requester and may withhold the answer until that volume is
  usable; a :class:`~proposed.selector.AnySelector` takes an application's own
  subject (which host prefills, which peer a prefix comes from) and is never
  installed -- an application's hosts ask one as a service of its own. A selector
  that is neither is consulted by another selector and reached from nowhere. The
  default :class:`~proposed.selector.NaiveKeySelector` returns the directory's own
  answer, so an installed selector changes nothing until one is written. It, the
  combinators beside it -- :class:`~proposed.selector.FirstMatch` (try selectors in
  order), :class:`~proposed.selector.KeySelectorChain` (the same over keys, so
  installable) and :class:`~proposed.selector.Refine` (one selector's ranking,
  narrowed by the tests behind it) -- and the
  :class:`~proposed.selector.Selector` base they are typed on are reached
  through the module rather than re-exported here: what a deployment has to
  *implement* is one of the two named subjects above.
* :mod:`proposed.deployment` -- :class:`~proposed.deployment.Controller`, the
  directory surface a caller reaches (torchstore names this type but never declares
  it: ``api.py`` annotates the spawned handle as the actor class and ``LocalClient``
  takes it unannotated). The difference between it and torchstore's class is the
  ask: the selector hook inside ``locate_volumes``, and ``locate_raw`` -- the same
  read without it, and what a sensor reads. Beside it,
  :class:`~proposed.deployment.ClusterModel`: the directory's peer on the
  application's side, holding the load a store cannot see and written by one
  ``notify(fact)``, the way the directory is written by ``notify_put_batch``;
* :mod:`proposed.view` -- :class:`~proposed.view.View`, the read-only observation
  a controller hands a selector: who holds a key, where volumes are, what time it is.
  It reads a :class:`~proposed.deployment.Controller` through ``locate_raw``,
  synchronously.

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
from .cost import TransferCost
from .deployment import (
    ClusterModel, Controller, Deployment, Key, StorageFull, StorageVolume,
    VolumeId,
)
from .plane import ControlPlane, DataPlane
from .selector import DecisionLog, AnySelector, KeySelector, Selection
from .topology import Endpoint, locality, Tier, TIER_LABEL
from .view import View

__all__ = [
    # the torchstore ask
    "KeySelector",
    "AnySelector",
    "Selection",
    "Key",
    "VolumeId",
    "DecisionLog",
    "View",
    "Controller",
    "ClusterModel",
    "StorageVolume",
    "StorageFull",
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
