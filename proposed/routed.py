"""A member that answers "not here, there", and the caller that goes.

A data-plane member may answer with the *address* of the host the request belongs on
rather than serving it (:mod:`kvcache_sim.data.serving`). :func:`routed` is how the
plane says where in its answer that address is; :class:`RoutedPlane` is a caller
holding those hosts, calling the member again wherever it is sent, up to a cap.

One concept, and it is the **answer's**: the extractor reads the host out of what the
member returned, so a reroute is the server's decision and no argument of the caller's
can produce one. The call itself never changes -- the member is called at the new host
with exactly what a first-time caller passed -- so a rerouted member has one signature
and no parameter only a reroute fills in.

:func:`routed` **does not wrap the member**, for two reasons. Monarch's ``@endpoint``
turns a method into a descriptor holding the function, so a wrapper would have to sit
*under* it -- following the reroute inside the actor, which is the server-side
forwarding this exists instead of. And a service reads its surface off the coroutines
a plane declares, so a non-``async def`` wrapper would drop the member out of it.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, Optional, TypeVar
from weakref import WeakKeyDictionary

__all__ = ["Where", "routed", "declared", "RoutedPlane", "peerless"]

_M = TypeVar("_M")

#: Where a rerouted call goes, read off the answer that reroutes it. ``None`` -- and a
#: member carrying no declaration at all -- is an answer that is the result.
Where = Callable[[Any], Optional[str]]

#: Every declaration made, keyed two ways so it survives ``@endpoint`` in either
#: order: by the member's own qualified name, which the class attribute is named after
#: when ``@endpoint`` is applied *over* :func:`routed`, and against the decorated
#: object, which *is* the class attribute when it is applied under. Written while a
#: module is imported, read when its classes are created: both before anything runs.
_BY_NAME: Dict[str, Where] = {}
_BY_OBJECT: "WeakKeyDictionary[Any, Where]" = WeakKeyDictionary()


def routed(*, at: Where) -> Callable[[_M], _M]:
    """Declare where this member's answer sends the call, and return it unchanged.

    ``at`` is read off the answer and names the host to call this same member on again,
    unchanged; ``None`` is an answer nobody has to take anywhere. Inert on the host,
    which reads none of it: what reads it is a caller.
    """

    def declare(member: _M) -> _M:
        if inspect.isfunction(member):
            _BY_NAME[f"{member.__module__}.{member.__qualname__}"] = at
        try:
            _BY_OBJECT[member] = at
        except TypeError:
            pass  # not weak-referenceable; the name above is the record
        return member

    return declare


def declared(cls: type) -> Dict[str, Where]:
    """``member -> where its answer sends the call``, in declaration order.

    What :meth:`proposed.plane.DataPlane.__init_subclass__` puts on the class -- read off
    the class, because the decorator returns the member unchanged and never sees it.
    """
    out: Dict[str, Where] = {}
    for klass in reversed(cls.__mro__):
        stem = f"{klass.__module__}.{klass.__qualname__}"
        for name, value in vars(klass).items():
            at = _BY_NAME.get(f"{stem}.{name}")
            if at is None:
                try:
                    at = _BY_OBJECT.get(value)
                except TypeError:
                    at = None  # unhashable or not weak-referenceable: not a member
            if at is not None:
                out[name] = at
    return out


class RoutedPlane:
    """A data plane's hosts as a caller reaches them, reroutes followed.

    Every member of the plane is a coroutine here, taking that member's own arguments
    plus ``at`` -- the host to call, which is the caller's own choice (a client SDK
    balancing, an ingress proxy, DNS) and the one thing no answer can supply. A
    :func:`routed` member is then called again wherever its answer sends it.

    Args:
        deployment: what resolves an address into a reference -- one
            :class:`~proposed.deployment.Deployment` member, ``plane_handle``.
        plane: the data-plane class whose :attr:`~proposed.plane.DataPlane.routes`
            say which members reroute.
        max_hops: how many times one call may be sent on. The answers terminate it --
            an address that came back ``None``, or no answer at all -- so this bounds
            only a plane that keeps sending a call somewhere: a cycle, or a plane whose
            answer about one subject is not settled by having been given. Small on
            purpose, because a chain that is long is a plane that does not converge, and
            that is a fault to see rather than a limit to raise.
    """

    def __init__(self, deployment: Any, plane: type, *, max_hops: int = 8) -> None:
        self.deployment = deployment
        self.routes: Dict[str, Where] = dict(getattr(plane, "routes", {}))
        self.max_hops = max_hops

    def __getattr__(self, member: str) -> Any:
        """Any member of the plane, reached the way this object reaches one."""
        if member.startswith("_"):
            raise AttributeError(member)
        return partial(self._call, member)

    async def _call(self, member: str, *args: Any, at: str) -> Any:
        """Call ``member`` at ``at``, and again wherever its answer sends it.

        Every call goes through the reference, so the boundary is paid once per hop --
        the charge a caller reaching each host itself would pay.
        """
        where = self.routes.get(member)
        host = at
        for _ in range(self.max_hops):
            reference = self.deployment.plane_handle(host)
            answered = await getattr(reference, member).call_one(*args)
            if where is None or answered is None:
                return answered  # nothing to read an address out of
            host = where(answered)
            if host is None:
                return answered
        raise RuntimeError(
            f"{member!r} is still being sent on after {self.max_hops} hops: nothing in "
            f"that chain answered without an address"
        )


def peerless(plane: Any) -> None:
    """Refuse a routed data plane that holds a way to reach another host.

    Engines xor a peer table: a plane that answers with an address *and* could call it
    is a forwarder, which is what answering with one exists instead of. Checked where a
    plane is fronted, because the structure lint cannot see it -- a peer is not a control
    port, so nothing in the import graph says a host grew one. What a way to reach a host
    looks like is what this host *is*: an object offering every member the plane offers,
    or one of the plane's own type.
    """
    if not getattr(type(plane), "routes", None):
        return
    surface = {
        name for name in dir(type(plane))
        if not name.startswith("_")
        and inspect.iscoroutinefunction(getattr(type(plane), name, None))
    }
    for name, held in sorted(vars(plane).items()):
        if isinstance(held, Mapping):
            values: Any = held.values()
        elif isinstance(held, (list, tuple, set, frozenset)):
            values = held
        else:
            values = (held,)
        for value in values:
            if value is plane or value is None:
                continue
            if isinstance(value, type(plane)) or surface <= set(dir(value)):
                raise TypeError(
                    f"{type(plane).__name__}.{name} holds another host: a routed "
                    f"member answers with an address, and a plane that can call one "
                    f"forwards instead of answering. Hand the address back"
                )
