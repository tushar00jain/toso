"""What the coordinator decided and the cluster has not done yet.

Every entry in :mod:`kvcache_sim.control._sensor._pending` stands for one thing that is
going to happen, and expires when it does -- folded from the action that says so, while a
read of the reservations filters at its own clock besides. These assert both directly,
because the scenarios that would exercise them do so rarely: a measured run of the
early-rejection comparison reads the reservations 800 times and only 3 of those reads see
an entry that has come true. The rule has to hold on all 800.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest kvcache_sim/tests/test_pending.py -q
"""

from __future__ import annotations

from kvcache_sim.control._answer import Plan, Response
from kvcache_sim.control._sensor import (
    Committed, FetchAnswered, PrefillFinished, Reservation, ReservationSensor,
    RoutedPullSensor, SourceLoad,
)


# --------------------------------------------------------------------------
# ReservationSensor: prefills promised, until they land.
# --------------------------------------------------------------------------


def test_a_reservation_that_has_come_true_is_not_pending():
    """Otherwise the request is counted twice.

    Once its prefill lands, the data plane reports the decode batch it joined, so
    the request is in the observed decode state. A reservation still standing for
    it would be added on top of that -- the same request predicted as two.
    """
    reserved = ReservationSensor()
    reserved.reserve(prefill_done=10.0, decode_id="d0", output_tokens=4)
    assert [r.decode_id for r in reserved.pending(now=9.0)] == ["d0"]
    assert list(reserved.pending(now=10.5)) == []


def test_a_reservation_is_pending_up_to_the_instant_it_lands():
    """The boundary is inclusive: at exactly its completion it still counts."""
    reserved = ReservationSensor()
    reserved.reserve(prefill_done=10.0, decode_id="d0", output_tokens=4)
    assert len(reserved.pending(now=10.0)) == 1


def test_a_read_filters_at_its_own_clock_and_not_at_the_last_write():
    """The property the extraction exists for.

    A routing decision reads this sensor before anything reports, so a sensor that only
    dropped what it was told about would serve entries whose prefill has since completed.
    The read filters, so it is never stale however long it has been since the last write
    or the last report.
    """
    reserved = ReservationSensor()
    reserved.reserve(prefill_done=1.0, decode_id="d0", output_tokens=4)
    reserved.reserve(prefill_done=2.0, decode_id="d1", output_tokens=4)
    # Nothing reported -- and the read is still correct at every instant.
    assert len(reserved.pending(now=0.0)) == 2
    assert [r.decode_id for r in reserved.pending(now=1.5)] == ["d1"]
    assert list(reserved.pending(now=99.0)) == []


def test_a_landed_prefill_drops_the_reservations_it_made_stale():
    """What keeps a long run from carrying every prefill it ever promised.

    A host's report is a clock it has reached, so what it retires is what no later read
    could return anyway -- asserted at a clock earlier than the report, where the entries
    would still be pending if the fold had kept them.
    """
    reserved = ReservationSensor()
    reserved.reserve(prefill_done=1.0, decode_id="d0", output_tokens=4)
    reserved.reserve(prefill_done=9.0, decode_id="d1", output_tokens=4)
    reserved.folds[PrefillFinished](PrefillFinished("s0", 5.0))
    assert [r.decode_id for r in reserved.pending(now=0.0)] == ["d1"]


# --------------------------------------------------------------------------
# RoutedPullSensor: a peer priced at routing, until the store asks about it.
# --------------------------------------------------------------------------


def _answered(requester="s0", keys=("a",)):
    """The action a plane dispatches as it answers one fetch."""
    return FetchAnswered(requester, tuple(keys))


def test_a_routed_pull_is_answered_once():
    """A pull is fetched once, so the answer spends the memo it came from.

    An entry left behind would answer some later fetch from the same instance with a
    peer chosen for a different request -- and charge a locality tier nobody priced.
    """
    routed = RoutedPullSensor()
    routed.route("s0", ["a", "b"], "s1")
    assert routed.peer("s0", ["a", "b"]) == "s1"
    routed.folds[FetchAnswered](_answered(keys=("a", "b")))
    assert routed.peer("s0", ["a", "b"]) is None


def test_answering_a_fetch_nothing_priced_spends_nothing():
    """Which is what lets the plane dispatch it without knowing which link answered.

    A fetch the ranking answered names keys no memo matches, and the fold leaves every
    other requester's memo where it was.
    """
    routed = RoutedPullSensor()
    routed.route("s0", ["a"], "s1")
    routed.folds[FetchAnswered](_answered(requester="s2"))
    routed.folds[FetchAnswered](_answered(keys=("z",)))
    assert routed.peer("s0", ["a"]) == "s1"


def test_pulls_to_one_instance_are_answered_oldest_first():
    """Two requests in flight to one instance resolve in a fixed order.

    One rule for the read and the expiry, so the memo an answer came from is the memo
    that answer spends.
    """
    routed = RoutedPullSensor()
    routed.route("s0", ["a"], "s1")
    routed.route("s0", ["a"], "s2")
    assert routed.peer("s0", ["a"]) == "s1"
    routed.folds[FetchAnswered](_answered())
    assert routed.peer("s0", ["a"]) == "s2"


def test_a_memo_answers_exactly_what_was_planned():
    """A pull is all-or-nothing, so a fetch asks for precisely what it was told to.

    A smaller set is therefore not this pull with the evicted blocks removed -- it
    is a different pull, and answering it with this peer would charge it a
    locality tier chosen for another request.
    """
    routed = RoutedPullSensor()
    routed.route("s0", ["a", "b", "c"], "s1")
    assert routed.peer("s0", ["a", "b"]) is None
    assert routed.peer("s0", ["c", "b", "a"]) == "s1"  # order is not identity


def test_nobody_elses_pull_answers_a_fetch():
    """A peer priced for one requester says nothing about another's fetch."""
    routed = RoutedPullSensor()
    routed.route("s0", ["a"], "s1")
    assert routed.peer("s2", ["a"]) is None       # different requester
    assert routed.peer("s0", ["a", "z"]) is None  # more than was planned
    assert routed.peer("s0", ["a"]) == "s1"       # ...and it is still there


# --------------------------------------------------------------------------
# Both fold the action a decision dispatches, each into its own state.
# --------------------------------------------------------------------------


def _committed(*, source=None, pull=(), decode="d0", done=10.0, output_tokens=4):
    """One accepted decision, as the action a commit dispatches."""
    plan = Plan(
        match_blocks=len(pull), cached_tokens=0, uncached_tokens=0,
        reuse_source=source, transfer_bytes=0, queue_wait=0.0, ttft=1.0,
        done_time=done,
    )
    plan.pull_keys = list(pull)
    return Committed(
        Response(prefill="s0", decode=decode, plan=plan), output_tokens
    )


def test_a_reservation_is_written_by_the_action_and_nothing_else():
    """The fold is the whole of the write path, and it reads only the action.

    Which is what lets the scheduler dispatch one ``Committed`` instead of testing a
    flag: this sensor cannot see the scheduler's, and a run that does not predict
    composes no sensor for this fold to be registered on.
    """
    reserved = ReservationSensor()
    reserved.folds[Committed](_committed(decode="d1", done=10.0, output_tokens=7))
    assert list(reserved.pending(now=9.0)) == [
        Reservation(prefill_done=10.0, decode_id="d1", output_tokens=7)
    ]


def test_a_pull_is_remembered_only_when_the_plan_priced_one():
    """The condition is on the action's own payload, so the fold applies it.

    Most accepted plans recompute the gap rather than pull it, and a plan with no
    source (or no keys to fetch) leaves nothing for a later fetch to be answered from --
    so recording one would answer a fetch with a peer nothing was priced against.
    """
    routed = RoutedPullSensor()
    routed.folds[Committed](_committed(source=None, pull=()))
    routed.folds[Committed](_committed(source="s1", pull=()))     # priced no keys
    routed.folds[Committed](_committed(source=None, pull=["a"]))  # named no peer
    assert routed.peer("s0", ["a"]) is None, "nothing was priced, so nothing is owed"
    routed.folds[Committed](_committed(source="s1", pull=["a"]))
    assert routed.peer("s0", ["a"]) == "s1"


def test_load_counts_the_source_a_decision_priced_a_pull_against():
    """The load a ranking spreads over is written by the decision that names a source.

    Only a decision that priced a pull counts: one that recomputes the gap sends
    nothing to anybody, so loading a volume for it would rank down a source nobody is
    reading from.
    """
    load = SourceLoad()
    load.folds[Committed](_committed(source=None, pull=()))         # nothing named
    load.folds[Committed](_committed(source="s1", pull=["a"]))
    load.folds[Committed](_committed(source="s1", pull=["b"]))
    load.folds[Committed](_committed(source="s2", pull=["c"]))
    assert dict(load.named()) == {"s1": 2, "s2": 1}
