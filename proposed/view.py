"""The read-only observation a control plane is handed: :class:`View`.

A :class:`~proposed.selector.KeySelector` never touches a client, a volume or the
mesh -- it is given a ``View`` and returns a decision. The ``View`` is the read-only
half of that contract: *reads* of state that already exists, and no mutation of any
kind.

What the base view offers, and how a capability adds to it
----------------------------------------------------------
* :meth:`locate` -- the real directory answer for a set of keys, read straight
  from the real ``Controller`` body, with no caller's source preference applied to
  it (see :mod:`proposed.selector`): a view must report the directory as it *is*,
  and a selector ranking an answer somebody has already reordered would be reading
  its own output back.
* :meth:`holders` / :meth:`topology` / :meth:`endpoint` / :meth:`locality` --
  who holds a key and how far away they are, the two inputs every source
  decision has needed so far.
* :meth:`now` -- the running loop's clock, because a decision has to be timed
  where it is *made*: a selector is handed a subject and no timestamp, since
  over a non-zero hop the sender's stamp is already stale and comparing it against
  every instance's queue would read the cluster in the past.
* :meth:`pinned` -- one decision, one directory. It is the *port* read that is
  pinned, so anything derived from it (``kvcache_sim``'s prefix runs) is coherent
  for the whole decision without knowing the pin exists. What is deliberately not
  pinned: a sensor, whose reads are live by design, and :meth:`locate_live`, for a
  caller whose correctness is freshness (a gate on a fact yet to land).

Anything more specific stays out, and is *composed* on instead: a capability adds
one subclass of this base per read it senses -- a sensor it holds is one line,
``cluster = Sensed()`` on a :class:`SensorView` -- and builds a view by naming them
(:meth:`View.derived`), so a selector takes the one view carrying the read it needs,
any one of them alone is already a view, and the class statement is the list of what
that capability's decisions sense. ``kvcache_sim``
composes leading-prefix-run lengths, which are a KV-cache notion (a block key
chain); ``dedup_sim`` composes the fan-out tree it has planned, which is one plane's
own bookkeeping and no directory's. Folding either into the base type would make it a
union serving neither caller -- and per-node *load* is the same trap twice over: the
KV-cache scheduler's load signal is its own predicted prefill queue (a control-plane
model, not an observation) and dedup's is its planned tree, so a base
``load()`` would be a stub with two incompatible meanings. It is left out until a
caller can observe one: an application ranking by load exposes its own
:class:`~proposed.deployment.Sensor` through a view of its own.

Construction
------------
A view is built from a :class:`~proposed.deployment.Controller` and a topology,
and reads the first through ``locate_raw`` alone. That surface is *declared* rather
than left as ``Any``, so the proposal states what it needs instead of silently
depending on the shape of whatever the simulator happens to pass. ``realsim``
builds one via ``Mesh.view``; a real controller would build one over itself.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Dict, FrozenSet, Iterator, Optional, Sequence

from proposed.cost import TransferCost
from proposed.deployment import Controller
from proposed.topology import Endpoint

__all__ = ["Sensed", "SensorView", "View"]


class _Pin:
    """The directory answer one decision pinned: which keys, and what they said.

    ``keys is None`` outside a decision that pinned, which is when :meth:`View.locate`
    is live. One cell per root view, shared *by reference* with everything derived from
    it (:meth:`View.derived`), so a read on any of those views is inside the root's pin
    rather than beside it -- a per-view cell would let a second derived view walk the
    directory again and answer for a different moment.
    """

    def __init__(self) -> None:
        self.keys: Optional[FrozenSet[str]] = None
        self.located: Dict[str, Dict[str, Any]] = {}


class View:
    """Everything a control plane senses and prices with, and nothing else.

    A container of run-supplied reads: who holds a key, where the volumes are, what
    time it is, what a transfer would cost. This base holds what a plane could not
    otherwise reach and nothing a capability made itself; a capability whose decisions
    also read state it holds itself composes a subclass carrying that
    :class:`~proposed.deployment.Sensor`, over these same ports (:meth:`derived`).

    What it reads are the members: :attr:`directory`, :attr:`topology`. Of the
    five the directory answers, one is safe for a decision to read, and that one is
    what :meth:`locate_live` calls -- the read spelled out here rather than left to
    each caller, since a decision made against an answer somebody has already
    reordered would be ranking its own output back.

    Args:
        directory: the directory to read -- anything satisfying
            :class:`~proposed.deployment.Controller`. In the simulator that is the
            controller service, in a deployment the controller itself.
        topology: ``volume_id -> Endpoint``; the volume id is the directory
            identity, the endpoint is what locality is priced against.
        cost: a :data:`~proposed.cost.TransferCost`. ``None`` for a run whose
            decisions price nothing, which is every capability but ``kvcache_sim``.
    """

    def __init__(
        self,
        directory: Controller,
        topology: Dict[str, Endpoint],
        cost: Optional[TransferCost] = None,
    ) -> None:
        self._directory = directory
        self._topology = dict(topology)
        self._cost = cost
        self._pin = _Pin()

    def derived(self, cls: type, **sensors: Any) -> "View":
        """A view of type ``cls`` over these same ports, carrying ``sensors``.

        How a capability composes its own reads in: ``cls`` names the views the
        capability adds -- ``kvcache_sim`` adds prefix runs, its model of the cluster,
        and the pulls it has priced -- rather than a read handle it assembles out of
        ports it would have to be handed one by one. Each derives this base, so
        composing a subset of them is a class statement and this base enters the
        result's MRO once.

        ``sensors`` is what the capability *already holds* and wants read through the
        same view as the run's ports, and nothing here supplies any of it (see the load
        discussion above). Passed as keywords, each claimed by the :class:`Sensed`
        attribute of that name, so this base names none of them -- and one no attribute
        claimed reaches this base's own ``__init__``, which takes none, and fails there
        rather than being silently absent.

        The pin (:meth:`pinned`) travels here rather than through the ports or the
        keywords: a keyword would be one this base has to name, which is exactly what
        makes a misspelled sensor raise, and the pin is not a port a run supplies.
        """
        view = cls(self._directory, self._topology, self._cost, **sensors)
        view._pin = self._pin
        return view

    @property
    def directory(self) -> Controller:
        """The directory this view senses. :meth:`locate` is the read to make of it."""
        return self._directory

    def locate(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``key -> {volume_id -> StorageInfo}``: off the pin if one is held, else live.

        Asserts on a key the pin does not cover: answering it live would be the
        incoherence the pin exists to rule out, and answering it absent would say
        nobody holds it -- so a read past the pin is a bug in the decision, not a miss
        to serve.
        """
        pin = self._pin
        if pin.keys is None:
            return self.locate_live(keys)
        assert all(key in pin.keys for key in keys), (
            "a pinned view answers for the keys it was pinned to; one decision reads "
            "one directory"
        )
        return {key: pin.located[key] for key in keys if key in pin.located}

    def locate_live(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``key -> {volume_id -> StorageInfo}`` from the REAL directory, now.

        Missing keys are simply absent (``missing_ok=True``): a read reports
        what is there, it does not raise at the observer. Reads the raw controller
        body, so no caller's preference is folded into what a decision is made
        against -- and it cannot suspend, which is what a pin relies on
        (:meth:`~proposed.deployment.Controller.locate_raw`).

        For a caller whose correctness is freshness rather than coherence: a gate on a
        fact that has not happened yet would park its waiter forever behind a read
        taken before the fact landed. Everything deciding *against* the directory reads
        :meth:`locate` instead.
        """
        if not keys:
            return {}
        return self._directory.locate_raw(list(keys), missing_ok=True)

    @contextmanager
    def pinned(self, keys: Sequence[str]) -> Iterator[None]:
        """Walk the directory once for ``keys``, and serve that walk for the block.

        Scoped state on the view rather than an answer passed around, because every
        selector a decision consults senses through a view derived from this one
        (:meth:`~proposed.selector.Selector.attach`) and would otherwise read past it
        into the live directory.

        Sound because one decision cannot be interleaved with another: the read
        underneath it is a plain synchronous method
        (:meth:`~proposed.deployment.Controller.locate_raw`), so there is no
        suspension point between the pin and its release. Should one ever appear, the
        assertions fire -- here on a second decision entering, in :meth:`locate` on a
        read of other keys arriving inside one.
        """
        pin = self._pin
        assert pin.keys is None, "a decision already holds this view's directory read"
        # Copied per key: the directory answers with its own volume map for each key
        # (:meth:`~proposed.deployment.Controller.locate_raw`), so a delete landing
        # inside the decision would otherwise rewrite what it pinned.
        located = {
            key: dict(volumes) for key, volumes in self.locate_live(keys).items()
        }
        pin.keys, pin.located = frozenset(keys), located
        try:
            yield
        finally:
            pin.keys, pin.located = None, {}

    # -- topology ----------------------------------------------------------- #
    @property
    def topology(self) -> Dict[str, Endpoint]:
        """``volume_id -> Endpoint`` for the whole run."""
        return self._topology

    # -- price -------------------------------------------------------------- #
    def transfer_cost(self, src_id: str, dst_id: str, nbytes: int) -> float:
        """Seconds to move ``nbytes`` from ``src_id`` to ``dst_id``.

        The run's estimate, not this capability's, so a prediction and the charge the
        transport makes cannot diverge. Raises for a run that supplied none: pricing
        a transfer a run cannot price is a scheduler in the wrong harness, not a
        number to invent.
        """
        if self._cost is None:
            raise RuntimeError(
                "this run supplied no transfer cost, so nothing here can be priced"
            )
        return self._cost(src_id, dst_id, nbytes)

    # -- clock -------------------------------------------------------------- #
    def now(self) -> float:
        """The running loop's clock, in seconds.

        Stock ``asyncio``, and the whole of what makes this liftable: under a plain
        loop it is ``time.monotonic()`` (real seconds), under a simulation engine
        whose loop overrides ``time()`` it is that run's virtual seconds. Same line
        either way -- which is why time is read here and never ``time.time()``,
        whose value no loop controls and no run can reproduce.
        """
        return asyncio.get_running_loop().time()


class Sensed:
    """One sensor a view carries, declared as the attribute it is read through::

        class ClusterView(SensorView):
            cluster = Sensed()

    The attribute name *is* the keyword :meth:`View.derived` takes, so the two cannot
    drift apart.

    Args:
        noun: what the raise calls the sensor, for a view whose attribute is not its
            name (``reserved`` reads a reservation sensor).
    """

    def __init__(self, noun: Optional[str] = None) -> None:
        self._noun = noun

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.noun = self._noun or name

    def __get__(
        self, view: Optional["SensorView"], owner: Optional[type] = None
    ) -> Any:
        """The sensor, or raise: an absent sensor is not an empty one.

        A run that composed none would otherwise read every host as idle, nothing as
        promised and no tree as planned -- answers that look healthy and are wrong.
        """
        if view is None:
            return self
        sensor = view._sensors.get(self.name)
        if sensor is None:
            raise RuntimeError(
                f"this view was composed without a {self.noun} sensor: an absent "
                f"sensor answers nothing, so compose it in ({self.name}=...)"
            )
        return sensor


def _sensed(cls: type) -> Sequence[str]:
    """Every :class:`Sensed` attribute reachable from ``cls``, each named once."""
    return tuple(dict.fromkeys(
        name
        for klass in cls.__mro__
        for name, attr in vars(klass).items()
        if isinstance(attr, Sensed)
    ))


class SensorView(View):
    """A view whose :class:`Sensed` attributes name the sensors it is composed with.

    Cooperative: it takes the keywords its own descriptors claim and hands the rest
    up, so several compose (``class KVView(PrefixView, ClusterView, ...)``) with this
    base and :class:`View` each entering the MRO once, and a keyword no descriptor
    claims arrives at :meth:`View.__init__`, which takes none.
    """

    def __init__(self, *ports: Any, **sensors: Any) -> None:
        self._sensors = {
            name: sensors.pop(name, None) for name in _sensed(type(self))
        }
        super().__init__(*ports, **sensors)
