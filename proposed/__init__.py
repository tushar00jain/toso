"""The surface proposed for torchstore itself.

Everything in this package is a component a real deployment would need but
torchstore does not have today. It is kept apart from ``realsim`` so the upstream
ask is legible at a glance: ``realsim`` is scaffolding that disappears outside the
simulator, ``proposed`` is the design being argued for.

* :mod:`proposed.selector` -- :class:`~proposed.selector.Selection` and the selection
  contracts that answer with one, each naming the subject it takes
  (``subject_type``). Utilities, not planes: a capability's
  :class:`~proposed.plane.ControlPlane` is what a run holds and a caller asks, and a
  selector is one of the things it works an answer out with. A
  :class:`~proposed.selector.KeySelector` takes *keys* -- which volumes should serve
  this read, the store's own question, and what a data plane hands the store as a
  source preference. A :class:`~proposed.selector.AnySelector` takes an application's
  own subject (which host prefills, which peer a prefix comes from). The default
  :class:`~proposed.selector.NaiveKeySelector` answers with the directory's own order,
  so preferring what it names changes nothing until a real one is written. It,
  :func:`~proposed.selector.prefer` (what the store does with a preference), the two
  combinators -- :class:`~proposed.selector.FirstMatch` (try key selectors in order)
  and :class:`~proposed.selector.Discount` (re-rank any one ranking by how much it has
  lately named each source) -- and the :class:`~proposed.selector.Selector` base they
  are typed on are reached through the module rather than re-exported here: what a
  deployment has to *implement* is one of the two named subjects above.
* :mod:`proposed.deployment` -- :class:`~proposed.deployment.Controller`, the
  directory surface a caller reaches (torchstore names this type but never declares
  it: ``api.py`` annotates the spawned handle as the actor class and ``LocalClient``
  takes it unannotated). The difference between it and torchstore's class is the
  ask: an optional **source preference** on ``locate_volumes``, which the store
  applies to its own answer without consulting anybody, and ``locate_raw`` -- the
  same read with nothing applied, which is what a view reads. Beside it,
  :class:`~proposed.deployment.Sensor`: the directory's peer on the application's
  side, holding the load a store cannot see, and
  :class:`~proposed.deployment.NotifiedSensor` -- one of those that a host writes
  from another process, by one ``notify(fact)``, the way the directory is written by
  ``notify_put_batch``. A capability declares the reads on its own sensor and exposes
  it through a view;
* :mod:`proposed.view` -- :class:`~proposed.view.View`, the read-only observation a
  control plane senses through: who holds a key, where volumes are, what time it is,
  and whatever sensors a capability composes onto it -- one
  :class:`~proposed.view.Sensed` attribute per sensor, on a
  :class:`~proposed.view.SensorView`. It reads a
  :class:`~proposed.deployment.Controller` through ``locate_raw``, synchronously.

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
    Controller, Deployment, Key, NotifiedSensor, Sensor, StorageFull,
    StorageVolume, VolumeId,
)
from .plane import ControlPlane, DataPlane
from .selector import DecisionLog, AnySelector, KeySelector, Selection
from .topology import Endpoint, locality, nearest, Tier, TIER_LABEL
from .view import Sensed, SensorView, View

__all__ = [
    # the torchstore ask
    "KeySelector",
    "AnySelector",
    "Selection",
    "Key",
    "VolumeId",
    "DecisionLog",
    "View",
    "SensorView",
    "Sensed",
    "Controller",
    "Sensor",
    "NotifiedSensor",
    "StorageVolume",
    "StorageFull",
    "Endpoint",
    "Tier",
    "TIER_LABEL",
    "locality",
    "nearest",
    # ports the application depends on
    "Deployment",
    "ControlPlane",
    "DataPlane",
    "TransferCost",
]
