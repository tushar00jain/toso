"""There is no KV a host holds that its volume has not been told about.

This model charges **events** and forgets **states**. A put is charged, a get is
charged, a decode step is charged, a service hop is charged -- every one of them
is a moment, and every one of them has a call site somebody had to write. Bytes
merely *sitting* on a host have no moment: nothing fires, no method is entered,
and the run reports exactly the same numbers whether an instance is holding one
block or ten thousand. Residency is therefore the one quantity here that can be
wrong without anything going wrong, and it has been wrong four times:

* the prefill -> decode handoff was a free Python method call, so the KV that
  crossed between two machines crossed nothing and landed nowhere;
* a decode host that had itself prefilled the request fetched its own KV back out
  of the store and billed itself for a transfer that never happened;
* the KV a decode host **generated** occupied nothing -- an instance could
  generate forever inside a bounded volume without once pressuring it;
* the chain a decode host **fetched in order to attend over it** occupied nothing
  either.

Every one of those was found by accident, refactors after it was introduced, and
every fix was the same sentence: a host that holds KV registers it on its volume.
The change that closed the last two did not claim the sweep was exhaustive, and
it was right not to -- nothing in the repo could have told it. This module is
that something.

What is reconciled, and against what
------------------------------------
:class:`~realsim.seams.volume_service.VolumeService` already tracks
``resident_bytes`` exactly, per key, maintained on put / delete / evict / reset.
The volume is not where the bug lives. The bug lives in the gap between *holding*
KV and *saying so*, which means a reconciliation is only worth anything if one of
its two sides does not come from a store call. Two sides that both derive from
``put`` share the blind spot and agree with each other about nothing.

So the reconciled quantity is the set of **(host, block key) pairs the run's own
outcomes imply a host must have held**, derived from three facts no store ever
sees:

* which instance **prefilled** a request. It computed or pulled every block of
  that request's chain in order to run the forward pass, so it held all of them
  (:attr:`~kvcache_sim.report.metrics.RequestResult.prefill`);
* which instance **decoded** it. Unless that is the same instance, a decode step
  attends over every token of the prompt, so it had the whole chain
  (:attr:`~kvcache_sim.report.metrics.RequestResult.decode`);
* how many tokens the **client received**. Every generated token past the first
  leaves a position of KV behind, under keys continuing the same chain
  (:meth:`~kvcache_sim.control.request.Request.continuation_keys`), so the decode
  host wrote ``blocks_for(output_tokens - 1)`` more blocks
  (:attr:`~kvcache_sim.report.metrics.RequestResult.output_tokens`).

None of those three is a store call, and none of them cares *how* the bytes
travelled. That is the property the whole test rests on, and it is what makes it
retroactive: the very first bug in the list moved KV host-to-host as a Python
object, so no store, transport or directory instrumentation anywhere could have
seen it -- but a decode host still had to be holding the chain, because it
decoded, and this derivation says so from the outcome row alone.

What multi-turn changed, and what it did not
--------------------------------------------
A request is now a **turn** of a conversation, and turn N+1's chain is turn N's
chain plus the keys turn N's *generation* left behind plus a new message
(:mod:`kvcache_sim.workload._generator`). Two consequences, and the second is the
reason nothing above had to be rewritten.

Blocks a decode host generated are now blocks a *prefill* host attends over on the
next turn -- so the same key is legitimately implied on two hosts by two different
facts, one from each side of the derivation. The pairs are a set and replication
was already in scope (see below), so that composes: a generated block held by the
decode host that made it and by the prefill host that reused it is two pairs, both
told, both true.

And the prefill side's sentence -- "it ran a forward pass over the whole prompt,
so it held every block of the chain" -- did not need a clause added for generated
blocks, because it was never phrased in terms of where a block came from. That is
worth stating rather than noticing: a derivation written from the physics ("a
forward pass attends over its whole prompt") extended itself to a workload it
predates, where one written from the call sites ("a prefill publishes what it
computed") would have had to be amended.
:func:`test_the_prefill_side_now_carries_generated_blocks` pins that it is really
exercised, since a derivation that silently covered nothing new would look
identical from here.

Against that: the set of keys each volume was ever **told about** -- every key of
every ``put`` that reached it, recorded by :class:`_RecordingVolume` for the
duration of the run.

Told, and deliberately not resident. Whether a volume *keeps* a block is a
capacity decision that belongs to the volume, is already reported
(:attr:`~kvcache_sim.report.metrics.RequestResult.published`,
:attr:`~kvcache_sim.report.metrics.RequestResult.decode_unpublished`), and is
supposed to drop things -- the eviction sweep's smallest capacity ends the run
holding 134 of the 1195 keys it was handed, and a resident-set comparison would
fail there for the most ordinary reason there is. What must never happen is that
the volume was never given the chance to decide. "Was told" is exactly that line,
and it is where the four bugs sit: in each of them the volume was not told.

The two sets are asserted **equal**, not merely one-sided, and they are equal in
every run today. The reverse inclusion is worth as much as the forward one: a key
a volume was told about by a host the outcomes say never held it would be a
registration for bytes nobody produced, which routes later reads at a phantom.

Alternatives considered, and why they are weaker
------------------------------------------------
**Published bytes minus evicted bytes, versus the sum of ``resident_bytes``.**
This reconciles the volume with itself. Both sides are ``put``-derived, so all
four bugs above are invisible to it -- the KV that was never published is missing
from the total *and* from the residency, and the arithmetic balances. It would
have been green throughout. (The much cheaper half of it is still asserted below,
as :func:`test_a_volumes_aggregate_is_exactly_its_per_key_sum`, because it costs
one line and catches a different thing: an aggregate drifting from the per-key
map it is supposed to be the sum of.)

**The directory's per-key holder sets versus each volume's ``_resident_by_key``.**
Two genuinely independent records of one fact, and they should agree, so
:func:`test_the_directory_and_the_volumes_agree_on_who_holds_what` checks it. But
it is not the check this module exists for, because both records are written by
the same event: a decode host that never published is absent from the directory
*and* absent from the volume, and the two agree perfectly about a host that is
holding an entire block chain. It catches the opposite failure -- an entry that
outlived its bytes, an eviction that deregistered without deleting -- which is
worth having and is not the recurring one -- and it has now caught one, which is
why it is worth keeping a check whose stated purpose is not this module's. Growing
the workload made a publish bigger than a small volume's slack, the volume evicted
a key out of the very ``put_batch`` that was landing it, reported the drop before
the batch was registered, and was registered for it anyway. See the note beside
``EVICTION_CAPACITIES`` in :mod:`kvcache_sim.workload.scenarios` for why that is an
ordering hole upstream of this repo rather than something the sweep can fix.

**Instrumenting the seams KV physically flows through** (``KVStore.fetch``, the
prefill and decode engines) and asserting a publish follows. Closer, and it would
have caught the two most recent bugs, but it catches only what flows through the
seams it wraps. The handoff-as-method-call bug bypassed the store entirely, and a
future one will bypass whatever is wrapped here -- that is what "held without
going through a put" means. Instrumentation follows the code; a derivation from
the request stream follows the physics.

**Checking at the instant of acquisition** rather than at the end of the run.
Same objection: it needs the same hooks, so it re-asks the question above with
more machinery.

Why this is not a lint
----------------------
It was considered for :mod:`realsim.tools.check_contract` and
:mod:`realsim.tools.check_structure` and does not fit either. Those decide
questions the AST answers -- which module imports which, whether a name is
public, whether a ``data/`` module reads a control port -- and they are strong
precisely because the answer is in the text. "Every byte a host holds is a byte
its volume heard about" is a statement about a *run*: it depends on which host
prefilled, which host decoded, how many tokens came back, and what the volume was
asked. The only static approximation available is a rule keyed to today's call
sites ("a ``fetch`` must be followed by a ``publish``"), and every one of the four
bugs was a *path that did not exist* when such a rule would have been written. A
lint would have been silent on all four and would have to be edited by whoever
introduced the fifth.

What does not reconcile, stated rather than smoothed over
---------------------------------------------------------
A handoff that found no chain in the store
(:attr:`~kvcache_sim.report.metrics.RequestResult.handoff_missed`) is excluded
from the implied set, because the model says outright that nothing arrived and
the request decodes over KV it does not account for -- which is a documented hole
in :meth:`kvcache_sim.data.serving.ServingHost.decode`, not one this test can
close: closing it means deciding what a decode with no KV *is*. No scenario here
produces one today, so the exclusion is currently inert, and if that changes this
test goes quiet about exactly the case worth noticing. Hence
:func:`test_no_scenario_hides_behind_a_missed_handoff`, which fails when the
exclusion starts carrying weight.

De-replication is out of scope and the invariant touches it only to confirm it:
the pairs below are a set, so a chain held by three hosts is three pairs, all of
them told, all of them legitimate. Nothing in this model ever decides a copy is
surplus (see :mod:`realsim.seams._retention`), and that is a control-plane
capability question rather than an accounting one.

Intra-generation residency is deferred by design: generated KV is published after
the last token rather than as blocks fill, so a generation long enough to fill a
block mid-flight is under-charged until it ends. Since this test reconciles end
states it cannot see that, and no scenario generates the 512 tokens it would
take.

Cost
----
Six scenarios, eighteen runs, a few seconds. The recording subclass adds a set
insert per put and changes no behaviour, so nothing here can move a metric.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List, Set, Tuple

import pytest

import realsim.mesh
from realsim.run import Result
from realsim.seams.volume_service import VolumeService

from kvcache_sim.__main__ import KVCacheDemo
from kvcache_sim.data.serving import ServingHost
from kvcache_sim.workload._accelerator import SimulatedAccelerator
from kvcache_sim.workload._serving import BLOCK_TOKENS
from kvcache_sim.workload.scenarios import Disaggregation

from ._run import results

#: The demo's own scenario list, so a seventh scenario is covered the day it is
#: added rather than the day somebody remembers to add it here.
SCENARIOS = KVCacheDemo().scenarios()
SCENARIO_IDS = [s.name for s in SCENARIOS]

#: How a generation's token count becomes a block count. The accelerator's own
#: answer, not a second copy of the arithmetic: it is the object that produced
#: the blocks (:meth:`~kvcache_sim.data._compute.Accelerator.generated_kv` is
#: ``kv_blocks(blocks_for(n))``), and a test that re-derived the geometry could
#: agree with a run that had it wrong.
_GEOMETRY = SimulatedAccelerator(block_tokens=BLOCK_TOKENS)


class _RecordingVolume(VolumeService):
    """A volume that also remembers every key it was ever handed.

    The whole of the observation side. Recorded **before** the capacity decision
    and regardless of its outcome, because the question is whether the volume was
    given the chance to account for these bytes -- a refusal is an answer, and a
    volume that never heard of the key gave none.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Every key any put on this volume named, kept for the run's lifetime.
        self.told: Set[str] = set()

    async def put(self, transport_buffer, requests) -> None:
        self.told.update(r.key for r in requests)
        await super().put(transport_buffer, requests)


@contextmanager
def _recording_volumes() -> Iterator[None]:
    """Build this process's volumes as :class:`_RecordingVolume` for the block.

    Patched on :mod:`realsim.mesh` rather than injected, because a mesh is built
    inside :meth:`realsim.run.Run.execute` from a scenario's declaration and
    there is no seam between the two -- and inventing one would put a test's
    needs into the run lifecycle every capability shares. The substitution is a
    subclass that only records, so the runs underneath it stay byte-identical.
    """
    original = realsim.mesh.VolumeService
    realsim.mesh.VolumeService = _RecordingVolume
    try:
        yield
    finally:
        realsim.mesh.VolumeService = original


def _execute(scenario) -> List[Result]:
    """Run every configuration of ``scenario`` with recording volumes."""
    with _recording_volumes():
        return results(scenario.runs())


#: Executed scenarios, by name. These runs are deterministic and every assertion
#: below only reads them, so the eighteen runs happen once for the module instead
#: of once per rule -- which is the difference between a few seconds and a
#: multiple of it. Keyed by name so the mutation test at the bottom can compare
#: its regressed runs against the very same baseline the other tests assert on.
_EXECUTED: Dict[str, List[Result]] = {}


def _runs(scenario) -> List[Result]:
    """``scenario``'s runs, executed once per module."""
    if scenario.name not in _EXECUTED:
        _EXECUTED[scenario.name] = _execute(scenario)
    return _EXECUTED[scenario.name]


# -- the two sides ---------------------------------------------------------- #
def _prefill_side(result: Result) -> Dict[str, Set[str]]:
    """``host -> keys`` each instance held because it prefilled a request.

    A prefill host runs a forward pass over the request's whole prompt, so it
    held every block of the chain: the leading run it already had, whatever it
    pulled from a peer, and the suffix it computed. ``prefill`` is set only by
    :meth:`kvcache_sim.data.serving.ServingHost.prefill`, so an empty one is a
    request refused at the door, which held nothing anywhere.
    """
    requests = {r.id: r for r in result.workload.requests}
    held: Dict[str, Set[str]] = defaultdict(set)
    for row in result.ledger.results:
        if row.prefill:
            held[row.prefill] |= set(requests[row.id].block_keys)
    return held


def _decode_side(result: Result) -> Dict[str, Set[str]]:
    """``host -> keys`` each instance held because it decoded a request.

    Two contributions, and the second is the one with no store call anywhere near
    it in the derivation: the chain the host had to have in order to attend over
    the prompt, and the blocks its generation appended. A request decoded where it
    was prefilled contributes no chain here -- the blocks are already on that
    volume under those keys -- but it still contributes its generated blocks,
    which are new wherever they were made.
    """
    requests = {r.id: r for r in result.workload.requests}
    held: Dict[str, Set[str]] = defaultdict(set)
    for row in result.ledger.results:
        if not row.decode:
            continue
        request = requests[row.id]
        if row.decode != row.prefill and not row.handoff_missed:
            held[row.decode] |= set(request.block_keys)
        generated = max(row.output_tokens - 1, 0)
        if generated:
            held[row.decode] |= set(
                request.continuation_keys(_GEOMETRY.blocks_for(generated))
            )
    return held


def _implied(result: Result) -> Dict[str, Set[str]]:
    """``host -> keys`` the run's outcomes say that instance must have held."""
    held: Dict[str, Set[str]] = defaultdict(set)
    for side in (_prefill_side(result), _decode_side(result)):
        for host, keys in side.items():
            held[host] |= keys
    return held


def _told(result: Result) -> Dict[str, Set[str]]:
    """``host -> keys`` each volume was ever handed in a put."""
    return {
        vid: handle.service.told for vid, handle in result.sim.mesh.volumes.items()
    }


def _reconcile(result: Result) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """``(untold, unimplied)`` -- the two directions' disagreements, per host.

    Both empty is the invariant. ``untold`` is the bug class this module exists
    for (a host holds KV its volume never heard of); ``unimplied`` is its mirror
    (a volume registered KV nobody produced).
    """
    implied, told = _implied(result), _told(result)
    untold = {
        host: keys - told.get(host, set())
        for host, keys in implied.items()
        if keys - told.get(host, set())
    }
    unimplied = {
        host: keys - implied.get(host, set())
        for host, keys in told.items()
        if keys - implied.get(host, set())
    }
    return untold, unimplied


def _sample(gaps: Dict[str, Set[str]]) -> str:
    """A readable few of a disagreement, for a failure message."""
    return "; ".join(
        f"{host}: {len(keys)} key(s), e.g. {sorted(keys)[:2]}"
        for host, keys in sorted(gaps.items())
    )


# -- the invariant ---------------------------------------------------------- #
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_every_block_a_host_holds_is_a_block_its_volume_was_told_about(scenario):
    """The invariant, over every configuration of every scenario.

    Equality, both ways, per host. See the module docstring for what each side is
    and why neither is the other's mirror image.
    """
    for result in _runs(scenario):
        where = f"{scenario.name}/{result.label}"
        untold, unimplied = _reconcile(result)
        assert not untold, (
            f"{where}: KV is held by a host whose volume was never told about it "
            f"-- {_sample(untold)}. Something moved or produced KV without "
            f"publishing it, which is free memory: no capacity consumed, no "
            f"eviction pressure, no directory entry, for blocks the host "
            f"demonstrably had to be holding"
        )
        assert not unimplied, (
            f"{where}: a volume was told about KV no host is accounted as holding "
            f"-- {_sample(unimplied)}. Either a publish named the wrong instance "
            f"or the outcome rows no longer describe where the run put things"
        )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_the_directory_and_the_volumes_agree_on_who_holds_what(scenario):
    """Two independent records of one fact, reconciled key by key.

    The directory says ``key -> {volume}``; each volume keeps its own
    ``_resident_by_key``. They are written by different objects at different
    moments -- the client registers the put after the volume has landed it, and a
    volume that evicts deregisters itself afterwards -- so a divergence is a real
    ordering bug: an entry routing reads to bytes that are gone, or bytes nothing
    can find.

    Reaching into ``_resident_by_key`` is deliberate. It is the record under test,
    not an implementation detail being leaned on: ``resident_bytes`` alone cannot
    say *which* keys, and which keys is the whole question here.
    """
    for result in _runs(scenario):
        where = f"{scenario.name}/{result.label}"
        directory: Dict[str, Set[str]] = {
            key: set(volumes)
            for key, volumes in
            result.sim.mesh.directory.service.controller.keys_to_storage_volumes.items()
        }
        resident: Dict[str, Set[str]] = defaultdict(set)
        for vid, handle in result.sim.mesh.volumes.items():
            for key in handle.service._resident_by_key:
                resident[key].add(vid)
        disagree = {
            key: (directory.get(key, set()), resident.get(key, set()))
            for key in set(directory) | set(resident)
            if directory.get(key, set()) != resident.get(key, set())
        }
        assert not disagree, (
            f"{where}: the directory and the volumes disagree about who holds "
            f"{len(disagree)} key(s) -- e.g. "
            f"{list(sorted(disagree.items()))[:2]} (directory, resident)"
        )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_a_volumes_aggregate_is_exactly_its_per_key_sum(scenario):
    """``resident_bytes`` is the sum of the per-key map it summarises.

    Cheap, and the one thing the volume can get wrong entirely on its own: the
    aggregate is maintained by increment and decrement across put, delete, evict
    and reset, and a single missed decrement makes a bounded volume refuse puts it
    had room for.
    """
    for result in _runs(scenario):
        for vid, handle in result.sim.mesh.volumes.items():
            volume = handle.service
            assert volume.resident_bytes == sum(volume._resident_by_key.values()), (
                f"{scenario.name}/{result.label} {vid}: resident_bytes "
                f"{volume.resident_bytes} != {sum(volume._resident_by_key.values())} "
                f"summed over {len(volume._resident_by_key)} keys"
            )
            assert volume.peak_resident_bytes >= volume.resident_bytes


def test_no_scenario_hides_behind_a_missed_handoff():
    """The one exclusion the invariant makes must stay unused.

    A decode host whose chain was gone from the store decodes anyway, and the
    model records that nothing arrived, so those keys are left out of the implied
    set. That is honest about a hole the model already documents -- but it is also
    the one way this test could go quiet while a decode host attends over KV
    nobody accounts for. It is inert today (no scenario produces a miss), and this
    is what says so out loud when it stops being.
    """
    for scenario in SCENARIOS:
        for result in _runs(scenario):
            assert result.ledger.handoff_misses == 0, (
                f"{scenario.name}/{result.label} now has "
                f"{result.ledger.handoff_misses} handoff miss(es), so the "
                f"residency invariant is skipping those chains. Decide what a "
                f"decode with no KV holds before trusting this test again"
            )


def test_the_decode_side_actually_contributes_something_to_check():
    """A test that checks nothing passes; make sure this one is checking.

    The decode-simulating scenarios are the reason this module exists, so the
    decode-side derivation must be non-empty in them -- and it must include
    instances that never prefilled anything, which is precisely the residency the
    model was blind to. Without this, deleting :func:`_decode_side` would leave
    the invariant green.
    """
    for result in _runs(Disaggregation(0)):
        decode_side = _decode_side(result)
        assert decode_side, f"{result.label}: no decode-side residency derived"
        prefill_side = _prefill_side(result)
        fresh = {
            host: keys - prefill_side.get(host, set())
            for host, keys in decode_side.items()
        }
        assert any(fresh.values()), (
            f"{result.label}: every decoded block was already held by its "
            f"prefill host, so this run cannot demonstrate the invariant"
        )


def test_the_prefill_side_now_carries_generated_blocks():
    """The multi-turn half of the derivation, and that it is not vacuous.

    A conversation's later turns walk through the blocks its earlier turns
    generated, so a host that prefills turn N+1 holds KV that a *generation*
    produced -- keys named by
    :meth:`~kvcache_sim.control.request.Request.continuation_keys`, which before
    the workload grew could only ever be implied on a decode host. Both halves are
    checked: that such keys are in the prefill side's implied set at all, and that
    at least one of them is implied on a host that did **not** decode the turn that
    made it, which is the case that only a chain crossing turns can produce.

    Without this, a workload change that quietly stopped splicing generated keys
    into later turns would leave every assertion above green while the thing they
    are supposed to be reconciling had gone away.
    """
    def generated(keys):
        return {k for k in keys if k.rsplit("|", 1)[1].startswith("g")}

    for scenario in SCENARIOS:
        for result in _runs(scenario):
            prefill_side = _prefill_side(result)
            decode_side = _decode_side(result)
            carried = {
                host: generated(keys) for host, keys in prefill_side.items()
            }
            assert any(carried.values()), (
                f"{scenario.name}/{result.label}: no prefill host is implied to "
                f"hold a single generated block, so later turns are not walking "
                f"earlier turns' output and the workload is single-turn again"
            )
            elsewhere = {
                host: keys - decode_side.get(host, set())
                for host, keys in carried.items()
            }
            assert any(elsewhere.values()), (
                f"{scenario.name}/{result.label}: every generated block a prefill "
                f"host holds was also decoded there, so nothing crossed turns"
            )


def test_the_invariant_catches_a_decode_host_that_holds_kv_silently():
    """Reintroduce the bug and watch it fail. The mutation this test is for.

    :meth:`kvcache_sim.data.serving.ServingHost._reside` is the whole of the
    decode side's registration -- the chain it pulled in and the blocks its
    generation appended, both through the ordinary publish. Making it a no-op is
    exactly the state of this model one commit ago, when a decode pool had
    unbounded free memory. The reconciliation must reject it, and it must reject
    it on the decode hosts specifically.

    A test asserting that a green thing is green tells you nothing about whether
    it would go red. This is the half that does.
    """
    async def _hold_it_silently(self, keys, blocks, request_id, why) -> None:
        return None

    original = ServingHost._reside
    ServingHost._reside = _hold_it_silently
    try:
        regressed = _execute(Disaggregation(0))
    finally:
        ServingHost._reside = original

    for result in regressed:
        untold, unimplied = _reconcile(result)
        assert untold, (
            f"{result.label}: a decode host published none of the KV it held and "
            f"the reconciliation still passed -- the invariant is not checking "
            f"the thing it claims to"
        )
        # ...and it is caught for the right reason: every key in the gap is one
        # the *decode* derivation contributed. A regression that showed up on the
        # prefill side would mean the two sides are entangled and the mutation is
        # not isolating what it claims to.
        decode_side = _decode_side(result)
        assert all(
            keys <= decode_side.get(host, set()) for host, keys in untold.items()
        ), f"{result.label}: the gap is not confined to the decode side: {untold}"
        assert not unimplied, (
            f"{result.label}: the regression should only ever make a volume know "
            f"*less*, so the reverse direction must stay clean"
        )
    # ...and the real code passes the same reconciliation, so the failure above
    # is the mutation and not the scenario.
    for result in _runs(Disaggregation(0)):
        assert _reconcile(result) == ({}, {})
