# Proposal: TorchStore testability and directory consistency

> **Status:** proposal. These changes target the sibling `../torchstore` repo;
> this document does not apply them. `realsim` already works around the testability
> gaps, so none of the changes block simulation. The off-actor seams preserve actor
> behavior; the directory work adds a consistency guarantee.

The proposal addresses two related problems: TorchStore logic is difficult to drive
without a Monarch actor mesh, and its directory is an unchecked notification mirror.
The first group exposes existing logic to ordinary tests and simulations. The second
makes directory divergence harmless, self-correcting, and time-bounded.

## Off-actor test seams

TorchStore's client planning and controller directory logic are mostly plain Python
behind `@endpoint` and actor initialization. Four small changes would let an
off-actor harness invoke that logic directly.

### 1. Extract directory reads into synchronous helpers

`Controller.notify_put_batch` delegates to synchronous `_notify_put`, but
`locate_volumes` and `keys` keep their `Trie` reads inside `@endpoint` methods.
Because endpoint descriptors are not callable off-actor, `realsim` mirrors those
bodies in `LocalControllerHandle`.

Move the bodies into helpers and delegate from the endpoints:

```python
def _locate_volumes(self, keys): ...
def _keys(self, prefix): ...

@endpoint
async def locate_volumes(self, keys):
    return self._locate_volumes(keys)

@endpoint
async def keys(self, prefix):
    return self._keys(prefix)
```

Tests and simulations could then call the real implementation. This is a pure
refactor; endpoint signatures and behavior remain unchanged.

### 2. Inject the transport factory per client

`LocalClient` resolves transport through the module-global
`torchstore.client.create_transport_buffer`. An in-memory transport therefore
requires a process-wide monkeypatch, which also prevents clients with different
transports from sharing a process.

Accept an optional factory on `LocalClient`, defaulting to the existing factory,
and store it on the instance:

```python
class LocalClient:
    def __init__(self, ..., transport_factory=create_transport_buffer):
        self._make_transport = transport_factory
```

This removes global patching and supports multi-client tests without changing
existing callers.

### 3. Add a non-actor `Controller` initialization path

`Controller.init` calls actor operations such as `storage_volumes.reset.call()`.
Directory-only users must consequently set `is_initialized = True` themselves.

Factor the mesh-free state and strategy setup into a plain method used by the actor
endpoint, or support construction in an initialized directory-only state. This is
additive and gives tests a supported path without exposing internal flags.

### 4. Avoid shadowing the `torchstore.client` submodule

The package-level `client()` function shadows the `torchstore.client` submodule,
forcing tooling to retrieve the module through `sys.modules`. Rename the accessor,
for example to `get_client`, and retain a deprecation shim if it is public API.

## Directory consistency

The directory (`key -> {volume_id -> StorageInfo}`) is a mirror of what N volumes
hold, maintained only by notifications. Nothing establishes, checks, or repairs
it: there is no version, incarnation, acknowledgement path, or way to read a
volume's contents back. The following design gives it a stated guarantee.

### Terms

For each volume `V`, over units `u = (key, object_type, slice)` -- the granularity
the index stores, not the key name:

- `G_V(t)` -- what `V` holds at `t` (ground truth).
- `I_V(t)` -- what the directory says `V` holds.
- `Δ⁻ = I_V \ G_V` -- **dangling**: the index names data that is not there.
- `Δ⁺ = G_V \ I_V` -- **orphan**: data that the index does not name.
- `D_V = Δ⁻ ∪ Δ⁺` -- desync.

The ideal invariant is `I_V = G_V`. The design does not preserve it at every
instant; it makes violations harmless, self-correcting, and time-bounded.

### Failure model

- **A1 (channel).** Messages from one sender to the directory arrive in order and
  at most once; duplicates are suppressed and a gap terminates the session rather
  than being skipped. The actor transport provides this per sender, not across
  senders.
- **A2 (visibility).** A message that cannot be delivered is reported to its sender.
- **A3 (single writer).** For each `V`, only `V` sends messages that mutate `I_V`.
- **A4 (local order).** `V` applies its mutations one at a time and emits their
  messages in the same order.
- **A5 (atomic apply).** The directory applies one message indivisibly with respect
  to reads.

A1, A2, and A5 already hold. P1 establishes A3 and A4.

### Design

**P1 -- one writer per volume.** The volume emits its own registrations and
deregistrations instead of splitting them between clients and the volume. Two
senders writing one subtree have no relative order; one sender inherits A1's order.

**P2 -- read repair at the source.** A read for a unit the volume does not hold is
the volume observing `Δ⁻`: it answers "absent" and deregisters the unit. Repair from
the volume preserves A3 and needs no fencing because the volume is authoritative
for its contents and can suppress the repair if the unit arrived meanwhile. Repair
must follow a store-level miss, never a delivery failure: "not here" and "cannot
reach" are opposite facts.

**P3 -- treat a located source as a hint.** A caller that misses at the named source
tries other holders, then the origin or recomputation. This turns `Δ⁻` from an error
into one wasted round trip.

**P4 -- total re-announce.** Periodically, and after an undeliverable message,
session break, restart, or other reason to doubt the directory, the volume sends
its whole unit set with a watermark; the directory atomically replaces that
volume's subtree. A total snapshot is idempotent, repairs dangling entries and
orphans together, and needs neither diff logic nor a volume enumeration endpoint.

**P5 -- incarnation.** Every message carries the volume's incarnation. A new
incarnation purges that volume's subtree before any other message applies. Restart
is otherwise invisible to the directory and is the one divergence P1--P4 cannot
bound.

**P6 (optional) -- Bloom localization.** A total announcement costs `O(|G_V|)` per
period. As an optimization, the directory sends `BF(I_V)` with a watermark and the
volume tests its units: *definitely absent* identifies a certain orphan; *maybe* is
skipped. This direction is additive only, so a stale filter causes a redundant
announcement, never a deletion.

### Correctness

**Lemma 1 (desync is the in-flight window).** Under A1, A3, and A4, every mutation
of `I_V` originates at `V` (A3), is emitted in local mutation order (A4), and
arrives in that order exactly once (A1). Thus `I_V` is a prefix of `V`'s mutation
log, and `D_V(t)` is exactly the set of units whose mutation is in flight at `t`.
Without failures, desync is bounded by one one-way latency. ∎

This is the load-bearing result and why P1 comes first. Without A3, two senders can
interleave: a delayed registration may arrive after the deregistration that
superseded it. `I_V` is then not a prefix of any mutation log, and repair cannot
prevent the race from recurring.

**Lemma 2 (failures are observable).** Every event that can break Lemma 1's premise
-- an undeliverable message, a session break, or a restart -- is reported to `V`
(A2) or is `V`'s own restart (P5). Therefore `V` can set a dirty bit whenever `I_V`
may no longer be a prefix of its log. ∎

This observability is necessary: a system that may be incomplete must know when to
degrade and re-announce.

**Theorem 1 (safety: uncertainty is one-sided).** Under P2 and P3, no state of
`D_V` produces a wrong answer.

- `u ∈ Δ⁺`: the directory omits a copy, so the caller uses another holder or goes
  upstream. The cost is an extra hop or recomputation; the result is unchanged.
- `u ∈ Δ⁻`: the caller reaches `V`, which answers absent and repairs the directory
  (P2); the caller falls back (P3). The cost is one wasted round trip; the result is
  unchanged.

The directory can therefore cause extra work, not a wrong answer. This requires a
miss to be retryable rather than fatal and requires "absent" to remain distinct
from "unreachable." Treating unreachable as absent can discard a live copy,
turning `Δ⁻` into `Δ⁺`; the theorem still preserves correctness but not that copy. ∎

**Theorem 2 (bounded desync).** Under P1, P4, and P5 with period `T`, if
`u ∈ D_V(t)`, then `u ∉ D_V(t')` for some `t' ≤ t + T + δ`, where `δ` is one
delivery. The re-announcement snapshots `G_V` at one instant and applies atomically
(A5). Immediately afterward, `D_V` contains only mutations later than the snapshot,
which Lemma 1 bounds. The total snapshot covers every unit. After a restart, P5
empties the subtree so replay begins from `∅`. ∎

Together, Theorems 1 and 2 give the caller-facing guarantee: *divergence costs
work, not correctness, and no divergence outlives `T + δ`.*

**Theorem 3 (P6 is sound).** A Bloom filter has no false negatives, so "definitely
absent from `I_V`" is certain for the snapshot; the watermark preserves that
certainty at apply time by excluding units written after the snapshot. False
positives only delay detection. With independent hashing each round, an orphan
survives `r` rounds with probability `≤ ε^r`. P6 gives a geometric expected bound,
while P4 retains the hard bound. ∎

### Limits

- **Version selection.** The directory says where, not which version. Version the
  key when two writes must be distinguished.
- **Complete multi-unit queries.** A possibly incomplete directory cannot prove it
  has every shard or every key under a prefix. Such queries must gate on the
  volume's watermark or go upstream. This is the case a prefix-matching cache
  depends on, so bounded desync alone is insufficient.
- **Change notification.** `T + δ` bounds repair, not awareness; callers are not
  notified that a unit they already located has changed.

## Suggested order

1. Implement P1 before any directory repair; the proof depends on one ordered
   writer per volume.
2. Pair P2 with P3 for safe reads, then add P4 and P5 for the hard repair bound.
   Treat P6 only as a bandwidth optimization.
3. For off-actor testing, prioritize seams 1 and 2. They remove mirrored endpoint
   logic and global transport patching. Seams 3 and 4 are lower-impact cleanup.
