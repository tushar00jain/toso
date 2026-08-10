"""One question, one interface: which volume serves these keys, and when?

Both capabilities in this repo ask the store the same thing. ``dedup_sim`` wants
a reader routed to a *peer* that is about to hold the key; ``kvcache_sim`` wants
the peer holding the longest reusable prefix, priced against recomputing it. The
answer in both cases is a ranked list of sources plus the moment they become
usable, so there is one interface:

    Policy.select(view, keys, requester) -> Selection

and two places it is invoked:

* **inside the controller's ``locate_volumes`` body**, after it reads the real
  directory and before it answers -- so a scenario that just calls
  ``client.get(K)`` is routed without knowing a policy exists. This is where the
  "and when" matters: a :class:`Selection` may carry a readiness gate, and the
  controller withholds its answer until that gate opens. Blocking the response is
  something a real controller can do (it is the same shape as waiting for a
  shard to commit), and it needs no client change;
* **directly from an app**, when the app wants to *price* the alternatives rather
  than be handed one -- ``kvcache_sim`` compares "pull from the best peer"
  against "recompute locally" before it commits to either.

What deliberately does not go through it: compute placement, admission and SLO
gates. Those are decisions the store knows nothing about; the moment ``select``
answers them it becomes a union type serving neither caller.

The default implementation is **naive**: every holder, in directory order --
which is exactly the answer the real ``Controller`` already gives, so the naive
selection is the empty one and the directory's own answer is returned untouched.
That is what makes "no policy installed" and "``NaivePolicy`` installed"
byte-identical runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Tuple

__all__ = ["Selection", "Policy", "NaivePolicy"]

# A readiness gate: called with no arguments, awaited until the chosen source is
# usable. ``None`` means "usable now".
Ready = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Selection:
    """Ranked sources for a set of keys, plus when they become usable.

    Args:
        sources: volume ids, best first. ``None`` -- the default -- means *every
            holder, in directory order*: the naive answer, and also what the real
            directory returns on its own, so a ``None`` selection leaves the
            controller's answer untouched.
        ready: optional gate awaited before the answer is released. A policy that
            routes a requester to a peer which has not registered yet returns the
            peer here plus a gate that opens when it does.
    """

    sources: Optional[Tuple[str, ...]] = None
    ready: Optional[Ready] = None

    @classmethod
    def of(cls, sources: Sequence[str], *, ready: Optional[Ready] = None) -> "Selection":
        """A selection ranking ``sources`` best-first."""
        return cls(sources=tuple(sources), ready=ready)

    async def wait(self) -> None:
        """Block until the chosen sources are usable (returns at once if ready)."""
        if self.ready is not None:
            await self.ready()

    def narrow(
        self, located: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Restrict a directory answer to the selected sources, in rank order.

        A key none of the selected sources holds is left untouched: the selection
        is a preference, not a filter that can make data disappear.
        """
        if self.sources is None:
            return located
        scoped: Dict[str, Dict[str, Any]] = {}
        for key, volume_map in located.items():
            ranked = {
                vid: volume_map[vid] for vid in self.sources if vid in volume_map
            }
            scoped[key] = ranked if ranked else volume_map
        return scoped


class Policy:
    """Source-selection policy. The base implementation is the naive one.

    Subclass and override :meth:`select` to route; override :meth:`notice` too if
    the routing has to wait for a source to appear.
    """

    name = "naive"

    async def select(
        self, view: Any, keys: Sequence[str], requester: str
    ) -> Selection:
        """Rank the volumes that should serve ``keys`` for ``requester``.

        Naive: every holder, in directory order, usable now. That is precisely
        the real directory's own answer, so this returns the empty
        :class:`Selection` rather than re-deriving it -- an installed
        ``NaivePolicy`` is therefore free and byte-identical to no policy at all.
        A caller that wants the list spelled out can read it off
        ``view.holders(await view.locate(keys), key)``.
        """
        return Selection()

    def notice(self, volume_id: str, keys: Sequence[str]) -> None:
        """The real directory just gained ``keys`` on ``volume_id``.

        Called by the controller on every real registration
        (``notify_put_batch``). Default: nothing. A policy whose answer is
        withheld until a planned peer registers opens its readiness gate here.
        """


class NaivePolicy(Policy):
    """Every holder, directory order -- the default, spelled out."""
