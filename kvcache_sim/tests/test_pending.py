"""What the coordinator decided and the cluster has not done yet.

Both sensors in :mod:`kvcache_sim.control._sensor._pending` are self-expiring, and
both expire on the *read* -- which is the whole reason they are objects rather than
two lists swept by whichever decision method happens to touch them. These assert the
expiry directly, because the scenarios that would exercise it do so rarely: a
measured run of the early-rejection comparison reads the reservations 800 times and
only 3 of those reads see an entry that has come true. The rule has to hold on all
800.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest kvcache_sim/tests/test_pending.py -q
"""

from __future__ import annotations

from kvcache_sim.control._sensor import ReservationSensor, RoutedPullSensor


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


def test_expiry_runs_on_the_read_not_on_the_write():
    """The property the extraction exists for.

    A routing decision reads this sensor before it writes to it, so expiry driven
    by the write is always one decision late -- every read would see entries whose
    prefill has since completed. Here the read is what cleans, so a read is never
    stale however long it has been since the last reservation.
    """
    reserved = ReservationSensor()
    reserved.reserve(prefill_done=1.0, decode_id="d0", output_tokens=4)
    reserved.reserve(prefill_done=2.0, decode_id="d1", output_tokens=4)
    # No further writes -- and the read is still correct at every instant.
    assert len(reserved.pending(now=0.0)) == 2
    assert [r.decode_id for r in reserved.pending(now=1.5)] == ["d1"]
    assert list(reserved.pending(now=99.0)) == []


# --------------------------------------------------------------------------
# RoutedPullSensor: a peer priced at routing, until the store asks about it.
# --------------------------------------------------------------------------


def test_a_routed_pull_is_answered_once():
    """Consumed on the match: a pull is fetched once.

    An entry left behind would be claimed by some later fetch from the same
    instance, which would be handed a peer chosen for a different request -- and
    charged a locality tier nobody priced.
    """
    routed = RoutedPullSensor()
    routed.route("s0", ["a", "b"], "s1")
    assert routed.claim("s0", ["a", "b"]) == "s1"
    assert routed.claim("s0", ["a", "b"]) is None


def test_pulls_to_one_instance_are_claimed_oldest_first():
    """Two requests in flight to one instance resolve in a fixed order."""
    routed = RoutedPullSensor()
    routed.route("s0", ["a"], "s1")
    routed.route("s0", ["a"], "s2")
    assert routed.claim("s0", ["a"]) == "s1"
    assert routed.claim("s0", ["a"]) == "s2"


def test_a_claim_is_for_exactly_what_was_planned():
    """A pull is all-or-nothing, so a fetch asks for precisely what it was told to.

    A smaller set is therefore not this pull with the evicted blocks removed -- it
    is a different pull, and answering it with this peer would charge it a
    locality tier chosen for another request.
    """
    routed = RoutedPullSensor()
    routed.route("s0", ["a", "b", "c"], "s1")
    assert routed.claim("s0", ["a", "b"]) is None
    assert routed.claim("s0", ["c", "b", "a"]) == "s1"  # order is not identity


def test_nobody_elses_pull_is_claimable():
    """A peer priced for one requester says nothing about another's fetch."""
    routed = RoutedPullSensor()
    routed.route("s0", ["a"], "s1")
    assert routed.claim("s2", ["a"]) is None       # different requester
    assert routed.claim("s0", ["a", "z"]) is None  # more than was planned
    assert routed.claim("s0", ["a"]) == "s1"       # ...and it is still there
