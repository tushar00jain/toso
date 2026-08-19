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
  source preference. A ``Selector[Subject]`` takes an application's own subject
  instead (which host prefills, which peer a prefix comes from). The default
  :class:`~proposed.selector.NaiveKeySelector` answers with the directory's own order,
  so preferring what it names changes nothing until a real one is written. The
  combinators a decision is declared out of -- :class:`~proposed.selector.FirstMatch`
  (try selectors in order), :class:`~proposed.selector.Annotate` and its
  :data:`~proposed.selector.Balance` (append a dimension),
  :class:`~proposed.selector.WithFold` (say how the key is read) and
  :func:`~proposed.selector.Ordered` / :func:`~proposed.selector.Best` (order or cut) --
  and the :class:`~proposed.selector.Selector` base they are typed on are reached
  through the module rather than re-exported here: what a deployment has to *implement*
  is the store's own subject above.
* :mod:`proposed.deployment` -- :class:`~proposed.deployment.Controller`, the
  directory surface a caller reaches (torchstore names this type but never declares
  it: ``api.py`` annotates the spawned handle as the actor class and ``LocalClient``
  takes it unannotated). The difference between it and torchstore's class is the
  ask: pending publications, an optional source preference on ``locate_volumes``,
  and a synchronous local read for directory sensors. Beside it,
  :class:`~proposed.deployment.Sensor`: the directory's peer on the application's
  side, holding the load a store cannot see. Selectors declare the sensor types they
  read and resolve them when attached;
* :mod:`proposed.dispatch` -- :class:`~proposed.dispatch.Dispatcher`, where a fact a
  host reports arrives and the one way a fact is announced: an
  :class:`~proposed.dispatch.Action` folded by every
  :class:`~proposed.dispatch.Reducer` that folds its type, and one commit, at which
  that action satisfies gates waiting on it. A gate wakes after every action it
  named has committed. No reducer can reach another's state;
* :mod:`proposed.routed` -- :func:`~proposed.routed.routed`, how a data plane says
  that a member may answer with the *address* of the host a request belongs on and
  where in that answer it is, and :class:`~proposed.routed.RoutedPlane`, a caller that
  goes there over :meth:`~proposed.deployment.Deployment.plane_handle`. The reroute is
  the server's decision; following it is nobody's to write twice;
* :mod:`proposed.environment` -- :class:`~proposed.environment.Environment`, the
  stable facts and calculations for one run;
* :mod:`proposed.sensors` -- :class:`~proposed.sensors.DirectorySensor`, the coherent
  directory read, and
  :class:`~proposed.sensors.LoadSensor`, the common load reading.

Import rule, enforced by ``realsim/tools/check_contract.py``: this package may not
import the simulator or a capability. TorchStore and Monarch actor primitives are part
of the deployment surface; simulator scaffolding is not.

The gaps each piece answers are listed in the design doc's "What torchstore is
missing" section.
"""

# Re-export the contract surface so callers import from the package directly.
from .deployment import (
    Controller, Deployment, Key, Sensor, StorageFull, StorageVolume, VolumeId,
)
from .dispatch import Action, Dispatcher, Reducer
from .endpoint import endpoint
from .plane import ControlPlane, DataPlane
from .routed import routed, RoutedPlane
from .selector import declares, DecisionLog, KeySelector, Selection
from .topology import Endpoint, locality, nearest, Tier, TIER_LABEL
from .environment import Environment
from .sensors import DirectorySensor, LoadSensor, Sensing

__all__ = [
    # the torchstore ask
    "KeySelector",
    "Selection",
    "Key",
    "VolumeId",
    "DecisionLog",
    "declares",
    "Environment",
    "DirectorySensor",
    "LoadSensor",
    "Sensing",
    "Action",
    "Dispatcher",
    "Reducer",
    "endpoint",
    "Controller",
    "Sensor",
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
    "routed",
    "RoutedPlane",
]
