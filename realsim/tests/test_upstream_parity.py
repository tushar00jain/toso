"""Guards on the places this repo restates torchstore instead of calling it.

All are deliberate and all can rot silently:

* :class:`proposed.deployment.Controller` declares the directory surface, because
  ``proposed`` must not import torchstore -- it is what torchstore would gain, so
  depending on it would invert the claim. The signatures are therefore a copy,
  except for the members that *are* the ask (``THE_ASK``), which upstream has yet
  to have. The half of the ask that is not a member -- a source preference on the read
  path -- is pinned on the client instead (``CLIENT_READ_PARAMETERS``), since what
  makes the simulator's stand-in necessary is the argument that is *not* there;
* :class:`proposed.deployment.StorageVolume` declares the storage surface the same
  way, and asks for nothing: every member is a copy, and each endpoint left out is
  listed with a reason (``VOLUME_NOT_DECLARED``) so a narrower surface stays a
  decision rather than an oversight;
* :class:`realsim.seams.controller_service.ControllerService` mirrors the bodies of
  ``locate_volumes`` and ``keys`` **verbatim**, because ``@endpoint`` makes them
  ``EndpointProperty`` descriptors that cannot be invoked off-actor, and torchstore
  has not extracted sync helpers for them the way it has for ``_notify_put``;
* :class:`proposed.planner.GreedyClient` mirrors ``_build_volume_requests`` with one
  branch changed, which is the client half of the ask (``CLIENT_THE_ASK``).

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

from torchstore.client import LocalClient as RealClient
from torchstore.controller import Controller as RealController
from torchstore.storage_volume import StorageVolume as RealStorageVolume

from proposed import Controller as ProposedController
from proposed import StorageVolume as ProposedStorageVolume
from realsim.seams.controller_handle import LocalControllerHandle
from realsim.seams.controller_service import ControllerService
from realsim.seams.volume_handle import LocalVolumeHandle
from realsim.seams.volume_service import VolumeService

#: Endpoints whose bodies are mirrored verbatim, and the digest of the source we
#: mirrored. A failure here is not a bug: it means upstream edited the body and the
#: mirror in ``ControllerService`` has to be re-copied (then update the digest).
#:
#: "Verbatim" is qualified in one direction: a mirror may carry insertions that are
#: part of :data:`THE_ASK`, each marked ``OURS`` at the line. The digest is over
#: upstream's source, so it tracks upstream and says nothing about those.
MIRRORED_BODIES = {
    "locate_volumes": "9604ec9210bbb386841f4c0cb447900a",
    "keys": "f3b003d0a74bf6e671bc2242cf2cb114",
}

#: Members of :class:`proposed.Controller` that are a copy of an upstream endpoint,
#: and must stay spelled the way upstream spells them.
COPIED_FROM_UPSTREAM = [
    "keys",
    "locate_volumes",
    "notify_delete",
    "notify_delete_batch",
    "notify_put_batch",
]

#: Members that are the *ask* -- declared here because torchstore would have to gain
#: them. All three are asked for as plain **synchronous local methods**: a directory
#: read or write that cannot suspend is what makes a routing decision atomic without a
#: lock.
#:
#: * ``locate_raw`` is ``locate_volumes`` with no caller's preference applied, which
#:   is what a control plane senses through a directory sensor;
#: * ``project`` / ``clear_projections`` record what a volume *will* hold, in the
#:   directory's own key-to-volume map, so a plane routing readers onto in-flight
#:   writers looks entries up instead of joining a private overlay per request. The
#:   implementation is :mod:`realsim.seams.projection`, mixed into
#:   :class:`~realsim.seams.controller_service.ControllerService`.
#:
#: Not every part of the ask is a member: ``locate_volumes`` takes a source preference
#: and applies it, and ``locate_raw`` takes ``projected=``; both change a signature
#: rather than adding one. The preference half is pinned on the client instead -- see
#: :data:`CLIENT_READ_PARAMETERS`, and the planning half -- see
#: :data:`CLIENT_THE_ASK` -- is a changed branch rather than any signature at all.
THE_ASK = ["clear_projections", "locate_raw", "project"]

#: The client half of the ask: a **source choice for sliced keys**, which is the same
#: ask as upstream's own ``TODO`` inside ``_expand_tensor_slices``. Today
#: ``_build_volume_requests`` takes the first volume listed and stops for an
#: ``OBJECT`` or a ``TENSOR``, so map order chooses; for a ``TENSOR_SLICE`` it walks
#: every volume and fetches every intersecting slice, so map order chooses nothing
#: and a replicated shard is fetched once per replica.
#:
#: :class:`proposed.planner.GreedyClient` is that method with a covered-region set
#: threaded through the sliced branch. It is what lets a control plane express a
#: ranking as a located map and have the store honor it for sliced and whole keys
#: alike; while it lives here, its source is pinned against upstream's by
#: :data:`CLIENT_MIRRORED_BODIES`.
CLIENT_THE_ASK = "greedy source selection for TENSOR_SLICE"

#: ``LocalClient`` bodies :mod:`proposed.planner` copies or calls into, and the digest
#: of the source it was written against. A failure means upstream edited the planner:
#: re-read it, re-copy what ``GreedyClient`` mirrors, and update the digest.
CLIENT_MIRRORED_BODIES = {
    "_build_volume_requests": "f1111e0c3ff1861f5ddffe5a58e51178",
    "_expand_tensor_slices": "9765db95694d7ba0fe6fdfe4d7d83b9c",
}

#: What ``LocalClient``'s read path takes today, in full. Part of the ask is that it
#: takes one thing more: an optional **source preference**, applied to the
#: located map before ``_build_volume_requests`` takes the first volume listed per
#: key. Until it does, the simulator carries that preference in a coroutine binding
#: (:func:`realsim.seams.factory.bind_prefer`), which is the one thing here standing
#: in for something upstream does not have.
#:
#: Spelled as the whole parameter list rather than a name guessed at, so *any* change
#: to the read surface fails here and gets looked at -- including the one being asked
#: for.
CLIENT_READ_PARAMETERS = {
    "get": ["self", "key", "inplace_tensor", "tensor_slice_spec"],
    "get_batch": ["self", "keys"],
}


def _real(name: str):
    """The undecorated function behind a real ``@endpoint``."""
    return RealController.__dict__[name]._method


def _digest(name: str) -> str:
    return hashlib.md5(
        inspect.getsource(_real(name)).encode("utf-8")
    ).hexdigest()[:32]


def _members() -> list[str]:
    return [
        name for name in vars(ProposedController)
        if not name.startswith("_") and callable(getattr(ProposedController, name))
    ]


def _declared() -> list[str]:
    """What ``Controller`` copies from upstream."""
    return [name for name in _members() if name not in THE_ASK]


def _asked() -> list[str]:
    """What it adds: the proposal."""
    return [name for name in _members() if name in THE_ASK]


def test_proposed_controller_declares_the_real_surface():
    """Every copied member exists upstream, spelled the same way."""
    assert sorted(_declared()) == sorted(COPIED_FROM_UPSTREAM), _declared()
    assert sorted(_asked()) == sorted(THE_ASK), _asked()
    for name in COPIED_FROM_UPSTREAM:
        assert name in RealController.__dict__, (
            f"proposed.Controller declares {name!r}, which the real Controller "
            f"does not have -- either it moved upstream or our copy invented it"
        )
        ours = list(inspect.signature(getattr(ProposedController, name)).parameters)
        theirs = list(inspect.signature(_real(name)).parameters)
        assert ours == theirs, (
            f"{name}: proposed.Controller takes {ours} and the real Controller "
            f"takes {theirs}. Upstream changed the surface; re-copy it (the "
            f"difference between the two is meant to be the ask, nothing else)"
        )


def test_the_ask_is_still_an_ask():
    """The members upstream lacks are the proposal; say so if that changes.

    A failure here is good news: torchstore grew the member, so it stops being
    something this repo is asking for and moves into ``COPIED_FROM_UPSTREAM`` --
    where its signature starts being checked against the real one.
    """
    for name in THE_ASK:
        assert name not in RealController.__dict__, (
            f"the real Controller now has {name!r}: move it from THE_ASK to "
            f"COPIED_FROM_UPSTREAM and check the signature still matches ours"
        )


def test_the_client_read_path_takes_no_preference():
    """What the simulator stands in for, pinned on the client's read members.

    ``realsim.seams.factory.bind_prefer`` exists *because* these signatures have
    nowhere to put it: a coroutine binding is how an unmodified client can be handed a
    source preference at all. When ``get`` grows the parameter, this fails -- and that
    binding can go, along with ``client_for``'s keyword.
    """
    for name, expected in CLIENT_READ_PARAMETERS.items():
        ours = list(inspect.signature(getattr(RealClient, name)).parameters)
        assert ours == expected, (
            f"LocalClient.{name} takes {ours}, not {expected}. If that is the source "
            f"preference arriving, the simulator's coroutine binding stops being a "
            f"stand-in and becomes a duplicate: pass it as this argument instead"
        )


def test_the_ask_is_local_and_synchronous():
    """What torchstore is asked for is a method, not an endpoint or a coroutine.

    A directory read that cannot suspend is what a control plane's atomicity rests
    on (``dedup_sim.control._selector``,
    ``kvcache_sim.control.scheduler._Scheduler._select_prefill``). ``async`` on it
    would not be a detail: it would put the interleaving back. Nor is it reached
    across a boundary -- it is absent from the handle, which is what a caller holds.
    """
    service = ControllerService(RealController())
    handle = LocalControllerHandle(service)
    for name in THE_ASK:
        declared = getattr(ProposedController, name)
        assert not inspect.iscoroutinefunction(declared), (
            f"proposed.Controller.{name} is a coroutine: the ask is a plain local "
            f"read, and awaiting it would let a second decision interleave"
        )
        assert not inspect.iscoroutinefunction(getattr(service, name)), (
            f"{name} is a coroutine on the service, so the service no longer "
            f"implements the surface it is meant to"
        )
        assert not hasattr(handle, name), (
            f"the handle offers {name}: the raw read has no caller across the "
            f"boundary, and one reaching it there would be charged a hop for a read "
            f"a control plane makes beside the directory"
        )


def test_the_mirrored_bodies_are_still_the_ones_we_mirrored():
    """A verbatim copy has to be told when the original changes."""
    for name, expected in MIRRORED_BODIES.items():
        assert _digest(name) == expected, (
            f"Controller.{name}'s body changed upstream. ControllerService "
            f"mirrors it verbatim, so re-copy the body and update the digest in "
            f"MIRRORED_BODIES."
        )


def test_the_mirrored_planner_is_still_the_one_we_mirrored():
    """``GreedyClient`` is upstream's planner plus one branch; pin the original."""
    for name, expected in CLIENT_MIRRORED_BODIES.items():
        source = inspect.getsource(getattr(RealClient, name))
        digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:32]
        assert digest == expected, (
            f"LocalClient.{name} changed upstream. proposed.planner.GreedyClient "
            f"mirrors it with one branch of its own, so re-copy the body, keep the "
            f"lines marked OURS, and update the digest in CLIENT_MIRRORED_BODIES."
        )


def test_the_sliced_read_still_has_no_source_choice():
    """The client half of the ask, checked against the method that would answer it.

    A failure is good news the same way :func:`test_the_ask_is_still_an_ask` is:
    upstream started tracking what a sliced read has already covered, so
    ``GreedyClient`` stops being a proposal and becomes a duplicate to delete.
    """
    source = inspect.getsource(RealClient._build_volume_requests)
    assert "covered" not in source, (
        f"LocalClient._build_volume_requests now tracks covered regions: "
        f"{CLIENT_THE_ASK} has landed upstream, so drop proposed.planner."
    )
    todo = inspect.getsource(RealClient._expand_tensor_slices)
    assert "we have already fetched this region" in todo, (
        "upstream's over-fetch TODO is gone from _expand_tensor_slices; check "
        "whether the sliced read now chooses a source, and retire GreedyClient if "
        "it does"
    )


def test_the_service_implements_the_surface_and_the_handle_refers_to_it():
    """Two objects, two shapes, and only one of them is a Controller.

    ``ControllerService`` implements :class:`proposed.Controller` -- plain async
    methods, the server side. ``LocalControllerHandle`` offers an endpoint per
    method instead, which is what real ``LocalClient`` code requires
    (``locate_volumes.call_one(...)``); collapsing those into methods would break
    every real caller, which is why the two are separate objects.

    The handle carries every member a *caller* reaches, and ``locate_raw`` is not
    one: its only reader is a control plane sensing through a directory sensor over the
    service itself, so nothing crosses the boundary the handle stands for. Which is
    why it is asked for as a plain synchronous method -- see
    :func:`test_the_ask_is_local_and_synchronous`.
    """
    service = ControllerService(RealController())
    handle = LocalControllerHandle(service)
    for name in _declared():
        assert inspect.iscoroutinefunction(getattr(service, name)), (
            f"{name} is not a coroutine on the service: that is the surface "
            f"proposed.Controller declares, and the service is what implements it"
        )
    for name in _declared():
        endpoint = getattr(handle, name)
        assert hasattr(endpoint, "call_one") and hasattr(endpoint, "call"), (
            f"{name} on the handle is not endpoint-shaped: a caller reaches an "
            f"actor method through call_one / call, not by calling it"
        )
        assert not inspect.iscoroutinefunction(endpoint), (
            f"{name} is a coroutine on the handle, i.e. the service shape -- real "
            f"client code calls {name}.call_one(...) and would break"
        )


# --------------------------------------------------------------------------
# The same three guards for the storage service. Its surface is a pure copy --
# unlike Controller, the proposal asks it to gain nothing -- so every member must
# exist upstream, spelled the same way.
# --------------------------------------------------------------------------

#: Endpoints the real ``StorageVolume`` has that this proposal does not declare,
#: and why. Listed rather than ignored so a member that starts being needed is a
#: decision someone makes, not a gap nobody noticed.
VOLUME_NOT_DECLARED = {
    "get_meta": "the store's own metadata read; no caller in this proposal",
    "get_id": "actor identity, which Monarch answers, not the surface",
    "spawn": "a constructor, not an endpoint",
    "actor_name": "a class attribute",
}


def _volume_declared() -> list[str]:
    return [
        name for name in vars(ProposedStorageVolume)
        if not name.startswith("_")
        and callable(getattr(ProposedStorageVolume, name))
    ]


#: The volume's half of the ask: declared here because a store that evicts well
#: needs it and torchstore has no equivalent. A volume sees only the accesses that
#: reach it, so a cache whose hits are served from data the caller already holds
#: leaves no trace on the object deciding what to drop.
VOLUME_THE_ASK = ["touch"]


def test_proposed_storage_volume_declares_the_real_surface():
    """Every copied member is upstream's, and every omission is deliberate."""
    declared = _volume_declared()
    assert sorted(declared) == sorted([
        "delete", "delete_batch", "get", "handshake", "put", "reset",
    ] + VOLUME_THE_ASK), declared
    for name in [n for n in declared if n not in VOLUME_THE_ASK]:
        assert name in RealStorageVolume.__dict__, (
            f"proposed.StorageVolume declares {name!r}, which the real "
            f"StorageVolume does not have -- it moved upstream or we invented it"
        )
        ours = list(
            inspect.signature(getattr(ProposedStorageVolume, name)).parameters
        )
        theirs = list(
            inspect.signature(RealStorageVolume.__dict__[name]._method).parameters
        )
        assert ours == theirs, (
            f"{name}: proposed.StorageVolume takes {ours} and the real one takes "
            f"{theirs}. Upstream changed the surface; re-copy it"
        )
    # Anything upstream we leave out is listed with a reason.
    upstream = {
        name for name, v in vars(RealStorageVolume).items()
        if not name.startswith("_")
    }
    for name in VOLUME_THE_ASK:
        assert name not in RealStorageVolume.__dict__, (
            f"the real StorageVolume now has {name!r}: it has stopped being an ask"
        )
    unexplained = upstream - set(declared) - set(VOLUME_NOT_DECLARED)
    assert not unexplained, (
        f"the real StorageVolume has {sorted(unexplained)}, which is neither "
        f"declared nor listed in VOLUME_NOT_DECLARED with a reason"
    )


def test_the_volume_service_implements_it_and_the_handle_refers_to_it():
    """Two shapes again: methods on the service, endpoints on the handle."""
    service = VolumeService()
    handle = LocalVolumeHandle(service)
    for name in _volume_declared():
        assert inspect.iscoroutinefunction(getattr(service, name)), (
            f"{name} is not a coroutine on the service: that is the surface "
            f"proposed.StorageVolume declares, and the service implements it"
        )
        endpoint = getattr(handle, name)
        assert hasattr(endpoint, "call_one") and hasattr(endpoint, "call"), (
            f"{name} on the handle is not endpoint-shaped: a caller reaches an "
            f"actor method through call_one / call, not by calling it"
        )
        assert not inspect.iscoroutinefunction(endpoint), (
            f"{name} is a coroutine on the handle, i.e. the service shape -- real "
            f"client code calls {name}.call(...) and would break"
        )
