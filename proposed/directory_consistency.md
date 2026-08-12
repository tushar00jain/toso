# Directory consistency: proposals and proof

The directory (`key -> {volume_id -> StorageInfo}`) is a mirror of what N volumes
hold, maintained only by notifications. Nothing establishes, checks or repairs it:
there is no version, no incarnation, no acknowledgement path, and no way to read a
volume's contents back. This is the design that would give it a stated guarantee.

## Terms

For each volume `V`, over units `u = (key, object_type, slice)` -- the granularity
the index actually stores, not the key name:

* `G_V(t)` -- what `V` holds at `t` (ground truth).
* `I_V(t)` -- what the directory says `V` holds.
* `Δ⁻ = I_V \ G_V` -- **dangling**: the index names data that is not there.
* `Δ⁺ = G_V \ I_V` -- **orphan**: data that the index does not name.
* `D_V = Δ⁻ ∪ Δ⁺` -- desync.

The invariant is `I_V = G_V`. The proposals do not preserve it; they make its
violation harmless, self-correcting and time-bounded, which is achievable where the
invariant is not.

## Failure model

* **A1 (channel).** Messages from one sender to the directory arrive in order and
  at most once; duplicates are suppressed and a gap terminates the session rather
  than being skipped. This is what the actor transport already provides *per
  sender*, and it says nothing about two senders.
* **A2 (visibility).** A message that cannot be delivered is reported to its sender.
* **A3 (single writer).** For each `V`, only `V` sends messages that mutate `I_V`.
* **A4 (local order).** `V` applies its own mutations one at a time and emits the
  corresponding message in that order.
* **A5 (atomic apply).** The directory applies one message indivisibly with respect
  to reads.

A1, A2 and A5 hold today. A3 and A4 are P1.

## Proposals

**P1 -- one writer per volume.** The volume emits its own registrations and
deregistrations, rather than its clients emitting the registrations and the volume
the deregistrations. Two senders writing one subtree have no relative order; one
sender inherits A1's.

**P2 -- read repair at the source.** A read for a unit the volume does not hold is
the volume observing `Δ⁻` directly: it answers "absent" and deregisters the unit.
Repair from the volume, not the reader, keeps A3 and needs no fencing -- the volume
is the authority on its own contents and can suppress the repair if the unit
arrived meanwhile. It must fire only on a store-level miss, never on a delivery
failure: "not here" and "cannot reach" are opposite facts.

**P3 -- a located source is a hint.** A caller that misses at the named source
falls back to the other holders, and to the origin (or recomputation) if there are
none. Without this, `Δ⁻` is an error; with it, `Δ⁻` is a wasted round trip.

**P4 -- total re-announce.** The volume periodically, and on every event that makes
it doubt itself (an undeliverable, a session break, a restart), sends its **whole**
unit set with a watermark; the directory replaces that volume's subtree. Total, not
a diff: one idempotent message repairs both hazards and needs no comparison logic
or enumerate endpoint on the volume.

**P5 -- incarnation.** Every message carries the volume's incarnation; a new
incarnation purges that volume's subtree before anything else applies. Restart is
otherwise invisible to the index, and is the one divergence P1--P4 cannot bound.

**P6 (optional) -- Bloom localization.** Shipping the whole unit set costs
`O(|G_V|)` per period. Instead the directory sends `BF(I_V)` plus a watermark and
the volume tests its own units: *definitely absent* is a certain orphan, *maybe*
is skipped. Direction matters -- this one is additive only, so a stale filter
causes a redundant re-announce, never a deletion.

## Correctness

**Lemma 1 (desync is the in-flight window).** Under A1, A3, A4: every mutation of
`I_V` originates at `V` (A3), arrives in emission order exactly once (A1), and is
emitted in the order it was performed (A4). So `I_V` is a *prefix* of `V`'s
mutation log, and `D_V(t)` is exactly the set of units whose mutation is in flight
at `t`. Absent failures, desync is bounded by one one-way latency. ∎

This is the load-bearing result, and it is why P1 comes first: without A3 the two
senders interleave and `I_V` is not a prefix of anything -- a delayed registration
can land after the deregistration that superseded it, and no amount of repair
prevents it recurring.

**Lemma 2 (failures are observable).** Every event that can break Lemma 1's
premise -- an undeliverable message, a session break, a restart -- is reported to
`V` (A2) or is `V`'s own restart (P5). So `V` can hold a *dirty* bit that is set
whenever `I_V` may no longer be a prefix of its log. ∎

Lemma 2 is what makes the rest honest: a system that may be incomplete must know
that it may be incomplete, or it cannot degrade deliberately.

**Theorem 1 (safety: uncertainty is one-sided).** Under P2 and P3, no state of
`D_V` yields a wrong answer.

* `u ∈ Δ⁺`: the index omits a copy, so the caller is routed to another holder or
  upstream. Cost: one extra hop, or a recomputation. Same result.
* `u ∈ Δ⁻`: the caller is routed to `V`, `V` answers absent, the caller falls back
  (P3) and `V` repairs (P2). Cost: one wasted round trip. Same result.

So the index can only cause *work*, never a wrong answer, and every divergence has
a bounded price. ∎

Two conditions this rests on, both easy to violate: a miss must be retryable
rather than fatal, and "absent" must be distinguished from "unreachable" -- with
the latter, repairing turns `Δ⁻` into `Δ⁺`, which is safe by the theorem but
discards a live copy.

**Theorem 2 (bounded desync).** Under P1, P4 and P5 with period `T`: if
`u ∈ D_V(t)` then `u ∉ D_V(t')` for some `t' ≤ t + T + δ`, `δ` one delivery. The
re-announce is a snapshot of `G_V` at a single instant, applied atomically (A5), so
immediately afterwards `D_V` contains only mutations later than the snapshot --
which Lemma 1 bounds. Every unit is covered because the announcement is total. A
restart is covered by P5, which empties the subtree so the replay starts from `∅`.
∎

Together: *divergence costs work, not correctness (T1), and no divergence outlives
`T + δ` (T2).* That is the guarantee to state to callers.

**Theorem 3 (P6 is sound).** A Bloom filter has no false negatives, so "definitely
absent from `I_V`" is certain with respect to the snapshot, and the watermark makes
it certain at apply time (the volume ignores units written after the snapshot).
False positives delay detection only: with independent hashing per round, an orphan
survives `r` rounds with probability `≤ ε^r`. P6 therefore gives a geometric
expected bound; P4's total re-announce keeps the hard one. ∎

## Not guaranteed

* **Which version.** The index says *where*, never *which*. Two writes of one key
  are indistinguishable to it; version the key if that matters.
* **Many-unit queries.** A query needing a *complete* set -- every shard of a
  sharded value, or a prefix range -- cannot be answered from a possibly-incomplete
  index, because completeness is exactly what is uncertain. Those must gate on the
  volume's watermark or go upstream. This is the one case bounded desync does not
  cover, and it is the case a prefix-matching cache lives on.
* **Notification.** `T + δ` bounds repair, not awareness: nothing tells a caller
  that a unit it already located has changed.
