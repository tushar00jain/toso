"""Guards on the two places this repo restates torchstore instead of calling it.

Both are deliberate and both can rot silently:

* :class:`proposed.deployment.Controller` declares the directory surface, because
  ``proposed`` must not import torchstore -- it is what torchstore would gain, so
  depending on it would invert the claim. The signatures are therefore a copy;
* :class:`realsim.seams.controller_handle.FakeControllerHandle` mirrors the bodies
  of ``locate_volumes`` and ``keys`` **verbatim**, because ``@endpoint`` makes them
  ``EndpointProperty`` descriptors that cannot be invoked off-actor, and torchstore
  has not extracted sync helpers for them the way it has for ``_notify_put``.

Nothing compared either against the original until this file. Tests here may
import torchstore, so they can: a signature that gains an argument upstream, or a
mirrored body that changes, fails the build with a message saying what to re-copy.

The parity check reads through ``EndpointProperty._method``, the undecorated
function Monarch keeps -- the only way to see a real endpoint's signature from
outside an actor.
"""

from __future__ import annotations

import hashlib
import inspect

from torchstore.controller import Controller as RealController

from proposed import Controller as ProposedController
from proposed import ControllerHandle
from realsim.seams.controller_handle import FakeControllerHandle

#: Endpoints whose bodies are mirrored verbatim, and the digest of the source we
#: mirrored. A failure here is not a bug: it means upstream edited the body and the
#: mirror in ``FakeControllerHandle`` has to be re-copied (then update the digest).
MIRRORED_BODIES = {
    "locate_volumes": "9604ec9210bbb386841f4c0cb447900a",
    "keys": "f3b003d0a74bf6e671bc2242cf2cb114",
}


def _real(name: str):
    """The undecorated function behind a real ``@endpoint``."""
    return RealController.__dict__[name]._method


def _digest(name: str) -> str:
    return hashlib.md5(
        inspect.getsource(_real(name)).encode("utf-8")
    ).hexdigest()[:32]


def test_proposed_controller_declares_the_real_surface():
    """Every member we declare exists upstream, spelled the same way."""
    declared = [
        name for name in vars(ProposedController)
        if not name.startswith("_") and callable(getattr(ProposedController, name))
    ]
    assert sorted(declared) == [
        "keys", "locate_volumes", "notify_delete", "notify_delete_batch",
        "notify_put_batch",
    ], declared
    for name in declared:
        assert name in RealController.__dict__, (
            f"proposed.Controller declares {name!r}, which the real Controller "
            f"does not have -- either it moved upstream or our copy invented it"
        )
        ours = list(inspect.signature(getattr(ProposedController, name)).parameters)
        theirs = list(inspect.signature(_real(name)).parameters)
        assert ours == theirs, (
            f"{name}: proposed.Controller takes {ours} and the real Controller "
            f"takes {theirs}. Upstream changed the surface; re-copy it (the "
            f"difference between the two is meant to be the policy hook, nothing "
            f"else)"
        )


def test_the_mirrored_bodies_are_still_the_ones_we_mirrored():
    """A verbatim copy has to be told when the original changes."""
    for name, expected in MIRRORED_BODIES.items():
        assert _digest(name) == expected, (
            f"Controller.{name}'s body changed upstream. FakeControllerHandle "
            f"mirrors it verbatim, so re-copy the body and update the digest in "
            f"MIRRORED_BODIES."
        )


def test_the_seam_implements_the_handle_and_not_the_service():
    """Two shapes, and the seam is the caller's one.

    ``FakeControllerHandle`` is a handle, so it offers an endpoint per method --
    which is what real ``LocalClient`` code requires
    (``locate_volumes.call_one(...)``). It deliberately does *not* implement
    :class:`proposed.Controller`: that is the service surface, and collapsing the
    endpoints into methods would break every real caller.
    """
    handle = FakeControllerHandle(RealController())
    declared = [n for n in ControllerHandle.__annotations__]
    assert sorted(declared) == sorted(
        n for n in vars(ProposedController) if not n.startswith("_")
    ), "the handle type and the service surface must name the same members"
    for name in declared:
        endpoint = getattr(handle, name)
        assert hasattr(endpoint, "call_one") and hasattr(endpoint, "call"), (
            f"{name} on the handle is not endpoint-shaped: a caller reaches an "
            f"actor method through call_one / call, not by calling it"
        )
        assert not inspect.iscoroutinefunction(endpoint), (
            f"{name} is a coroutine on the handle, i.e. the service shape -- real "
            f"client code calls {name}.call_one(...) and would break"
        )
