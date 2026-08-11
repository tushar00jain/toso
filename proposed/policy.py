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

:class:`NaivePolicy` is **naive**: every holder, in directory order -- which is
exactly the answer the real ``Controller`` already gives, so its selection is the
empty one and the directory's own answer is returned untouched. That is what makes
"no policy installed" and "``NaivePolicy`` installed" byte-identical runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any, Awaitable, Callable, Dict, Optional, Protocol, Sequence, Tuple,
)

from proposed.plane import ControlPlane
from proposed.view import View

__all__ = ["Ready", "Selection", "DecisionLog", "Policy", "NaivePolicy"]

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


class DecisionLog(Protocol):
    """Somewhere a policy can explain itself.

    Optional and never load-bearing: a policy must behave identically with none
    attached. Declared here so a policy can be handed one without naming the
    simulator's trace -- a deployment would pass its own logger.
    """

    def record(self, at: float, kind: str, message: str) -> None:
        ...


class Policy(ControlPlane, ABC):
    """Source-selection policy: the interface a controller consults.

    Abstract, so what a policy *is* and what the naive answer *does* are two
    things: :class:`NaivePolicy` is the implementation of "the directory answers
    for itself". Override :meth:`notice` too if the routing has to wait for a
    source to appear.
    """

    name = "policy"

    @abstractmethod
    async def select(
        self, view: View, keys: Sequence[str], requester: str
    ) -> Selection:
        """Rank the volumes that should serve ``keys`` for ``requester``."""

    def notice(self, volume_id: str, keys: Sequence[str]) -> None:
        """The real directory just gained ``keys`` on ``volume_id``.

        Called by the controller on every real registration
        (``notify_put_batch``). Default: nothing. A policy whose answer is
        withheld until a planned peer registers opens its readiness gate here.
        """

    async def evict(
        self, view: View, volume_id: str, need_bytes: int
    ) -> Sequence[str]:
        """``volume_id`` needs ``need_bytes`` freed: which keys should go?

        Asked by the *store*, at the one moment it knows it is out of room -- a put
        that would push a volume past its capacity. Which copies are worth keeping is
        not something the store can answer: it sees bytes, not what a caller will
        want next. So it asks, drops what it is told, and lets the put land.

        Default: nothing to drop, which leaves the store to refuse the put as it does
        with no policy installed. A capability that models per-volume residency
        (``kvcache_sim``'s LRU) overrides this and returns its coldest keys.

        The answer is advisory in one direction only: naming a key the volume does
        not hold is harmless, but naming too few to fit the put leaves the store to
        refuse it. Answering is not a promise that the store *had* to ask -- the same
        keys may already have been dropped for another reason.

        This is the half of write placement that exists today. The other half --
        *where* a new copy should go, so the store asks before writing rather than
        when it is already full -- has no seam yet, and the two belong together: you
        cannot choose where a copy lands without choosing what it displaces.
        """
        return ()


class NaivePolicy(Policy):
    """Every holder, in directory order, usable now.

    That is precisely the real directory's own answer, so this returns the empty
    :class:`Selection` rather than re-deriving it -- installing it is free and
    byte-identical to installing no policy at all. A caller that wants the list
    spelled out can read it off ``view.holders(await view.locate(keys), key)``.
    """

    name = "naive"

    async def select(
        self, view: View, keys: Sequence[str], requester: str
    ) -> Selection:
        return Selection()
