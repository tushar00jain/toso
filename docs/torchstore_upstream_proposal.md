# Proposal: make TorchStore client + controller drivable off-actor (sans-IO seams)

> **Status:** proposal / not urgent. `realsim` already drives the real client and
> controller without any of these changes (it works around all four). These are
> optional upstream cleanups that would (a) let `realsim`'s seams call real code
> instead of mirroring a few endpoint bodies, and (b) make TorchStore's own logic
> unit-testable without spawning a Monarch actor mesh. **Nothing here blocks the
> simulation.**
>
> Scope: changes to the sibling `../torchstore` repo. This doc only *proposes*
> them; it does not apply them.

## Why

TorchStore's client planning core and controller directory logic are already
plain, Monarch-free Python behind a thin actor/`@endpoint` shell. The only
things that force a live actor runtime today are (1) `@endpoint` read bodies that
aren't extracted into callable sync helpers, (2) a module-global transport
factory, and (3) actor-only initialization. Removing those seams lets the real
logic run under a single-threaded harness (like `realsim`) and under ordinary
`pytest` — a general testability win, independent of toso.

Each item below states the current state, the proposal, the benefit, and
backward-compat. All four are additive/refactors with no behavior change on the
actor path.

## 1. Extract `locate_volumes` / `keys` endpoint bodies into sync helpers

**Current.** `Controller.notify_put_batch` already delegates to a sync
`_notify_put`, but `locate_volumes` and `keys` keep their ~5-line `Trie` reads
*inside* the `@endpoint async` methods. Since `@endpoint` methods are descriptors
that aren't callable off-actor, a caller without a mesh must re-implement those
bodies (which `realsim`'s `FakeControllerHandle` does, mirroring them verbatim).

**Proposal.** Move the bodies into sync helpers and have the endpoints delegate,
mirroring the existing `_notify_put` pattern:

```python
def _locate_volumes(self, keys): ...   # the current locate_volumes body
def _keys(self, prefix): ...           # the current keys body

@endpoint
async def locate_volumes(self, keys):
    return self._locate_volumes(keys)

@endpoint
async def keys(self, prefix):
    return self._keys(prefix)
```

**Benefit.** Off-actor callers (sims, unit tests) invoke real code; no mirrored
duplication that can silently drift from the endpoint.

**Compat.** Pure refactor; the endpoint signatures/behavior are unchanged.

## 2. Make the transport factory injectable per client (not a module global)

**Current.** `LocalClient` resolves transport via the module-global
`torchstore.client.create_transport_buffer`. To substitute an in-memory
transport, `realsim` monkeypatches that global (`sys.modules["torchstore.client"]
.create_transport_buffer`), which is process-wide — awkward for tests and a
blocker for running multiple clients with different transports in one process.

**Proposal.** Let `LocalClient` accept an optional transport factory (constructor
arg or attribute), defaulting to the current global:

```python
class LocalClient:
    def __init__(self, ..., transport_factory=create_transport_buffer):
        self._make_transport = transport_factory
    # use self._make_transport(...) instead of the module global
```

**Benefit.** Per-client injection; no global monkeypatch; enables multi-client
scenarios (e.g. a read burst from many readers) cleanly.

**Compat.** Default preserves today's behavior; existing callers unaffected.

## 3. Add a non-actor `Controller` init path

**Current.** `Controller.init` needs a Monarch mesh (it calls
`storage_volumes.reset.call()` etc.), so off-actor use sets `is_initialized =
True` directly — reaching past the intended API.

**Proposal.** Factor the mesh-free part of initialization into a plain method (or
allow constructing an already-initialized controller for the directory-only use
case), so callers don't poke internal flags:

```python
def _init_local(self):
    # the Monarch-free portion of init (directory state, strategy wiring)
    self.is_initialized = True
```

**Benefit.** A supported way to get a directory-only controller for tests/sims;
no reliance on internal state.

**Compat.** Additive; the actor `init` endpoint keeps calling the shared logic.

## 4. Minor: `torchstore.client` submodule shadowed by a `client` function

**Current.** The package exposes a `client()` function whose name shadows the
`torchstore.client` submodule attribute, so tooling must reach the submodule via
`sys.modules["torchstore.client"]`.

**Proposal.** Rename the function (e.g. `get_client`) or the accessor to avoid
the collision.

**Benefit.** `torchstore.client` resolves to the submodule as expected;
removes a footgun.

**Compat.** A rename — needs a deprecation shim if the function is public API.

## Priority

- **Nice-to-have for `realsim`:** #1 (kills mirrored code) and #2 (kills the
  global monkeypatch, unblocks multi-client scenarios).
- **Cleanup:** #3 and #4.

None are required today. Revisit if/when `realsim` grows multi-client scenarios
(#2 becomes the most valuable) or if TorchStore wants off-actor unit tests of its
directory/planning logic (#1 + #3).
