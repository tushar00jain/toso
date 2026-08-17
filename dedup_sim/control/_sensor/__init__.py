"""What this capability's decisions read: the fan-out this plane has planned.

One sensor here: :class:`FanoutSensor`, the tree and the puts it is owed, moved by the
two actions that move a debt -- :class:`Asked`, which this plane dispatches to itself as
a reader asks, and :class:`proposed.dispatch.Stored`, which that reader dispatches once
its put lands. No waiting is recorded: a requester parks on the commit of an action this
folds, and nothing here knows who is parked.

The sensor is written only by dispatching, and has no service of its own: a second seam
for the same fact would not be ordered against the directory's, and a stale fan-out is a
readiness gate nothing opens.

Folder-private: selectors declare the sensor type rather than importing it outside control.
"""

from ._action import Asked
from ._fanout import FanoutSensor

__all__ = ["Asked", "FanoutSensor"]
