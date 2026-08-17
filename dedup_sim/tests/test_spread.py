"""What load spreading does to the dedup chain, on a key two trainers hold.

The chain's own outcome is one origin hop and a queue: reader ``i + 1`` waits on reader
``i``'s read-through. With ``n`` replicas of the key that is a choice rather than the
only answer, and these assert which one each configuration makes -- who served whom,
how much left any one trainer, and that the peer chain is still reached once every
replica is busy.

Outcome assertions, not timing: the edges the transports recorded and the routes the
plane took, both of which are exact.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests/test_spread.py -q
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from dedup_sim.workload.scenarios import WeightSync
from putget_sim.workload.put_get import DEFAULT_N
from realsim.run import Result

PAYLOAD_BYTES = DEFAULT_N * 4  # DEFAULT_N float32 elements

#: The runs :class:`~dedup_sim.workload.scenarios.WeightSync` declares, in order.
BASELINE, DEDUP, SPREAD = 0, 1, 2


def _run(trainers: int = 2, generators: int = 2) -> List[Result]:
    """Every run of the scenario, executed as the demo executes them."""
    return [run.execute() for run in WeightSync(trainers, generators).runs()]


def _served(result: Result) -> Dict[str, List[str]]:
    """``source -> the readers it served``, off the recorded transfer edges."""
    served: Dict[str, List[str]] = {}
    for src, dst, _label in result.ledger.edges:
        served.setdefault(src, []).append(dst)
    return served


def _edges(result: Result) -> List[Tuple[str, str]]:
    return [(src, dst) for src, dst, _label in result.ledger.edges]


def test_without_a_load_term_both_generators_read_the_same_trainer():
    """The premise: locality cannot separate two replicas, so the id does.

    Both the unrouted baseline and the plain chain send the first hop to ``t0`` and
    leave ``t1`` holding a copy nobody reads -- the baseline by directory order, the
    chain because that is the ranking's tie-break.
    """
    results = _run()
    assert _edges(results[BASELINE]) == [("t0", "g0"), ("t0", "g1")]
    assert _edges(results[DEDUP]) == [("t0", "g0"), ("g0", "g1")]
    assert "t1" not in _served(results[DEDUP])


def test_spread_gives_each_generator_a_trainer_of_its_own():
    """The outcome the load term buys: one hop per replica, in parallel.

    Two generators, two trainers, and each generator reads from a different one --
    which is the tie broken on something that moves: the combinator appends the queue
    the first decision left behind (:data:`proposed.selector.Balance`) and the plane's
    fold reads it.
    """
    spread = _run()[SPREAD]
    assert _edges(spread) == [("t0", "g0"), ("t1", "g1")]


def test_both_generators_still_read_through_under_spread():
    """Spreading changes the source, not the data plane: both publish what they read.

    The put is what makes a generator a source for the next reader, so this is the half
    of dedup that the spread run keeps -- the directory holds the key at every trainer
    *and* every generator when the run ends.
    """
    spread = _run()[SPREAD]
    holders = spread.sim.view.locate_live(["W"])["W"]
    assert set(holders) == {"t0", "t1", "g0", "g1"}


def test_spread_costs_one_origin_hop_per_replica_and_no_more():
    """The bound: a replica serves the key once, whatever the burst is.

    Four generators over two trainers is two origin hops, not four: once both replicas
    carry a reader, the load on them outweighs the wait behind a peer and the third
    generator is folded in behind one instead. The plain chain is 1x and the baseline
    ``m x``, so the three configurations bracket each other.
    """
    baseline, dedup, spread = _run(2, 4)
    assert baseline.ledger.origin_bytes == 4 * PAYLOAD_BYTES
    assert dedup.ledger.origin_bytes == 1 * PAYLOAD_BYTES
    assert spread.ledger.origin_bytes == 2 * PAYLOAD_BYTES


def test_a_reader_past_the_replicas_is_folded_in_behind_a_peer():
    """The fall-through, as a shape: two chains of two, not one chain of four.

    Every trainer is serving somebody by the time ``g2`` asks, so a peer outprices
    them -- which is what halves the depth of the chain the plain configuration
    builds.
    """
    spread = _run(2, 4)[SPREAD]
    assert _served(spread) == {"t0": ["g0"], "t1": ["g1"], "g0": ["g2"], "g1": ["g3"]}


def test_every_generator_receives_the_payload_under_spread():
    """The routing is a preference; nobody is left unserved by it."""
    for result in _run(2, 4):
        assert set(result.results) == {"g0", "g1", "g2", "g3"}
        for gid, payload in result.results.items():
            assert payload.numel() * payload.element_size() == PAYLOAD_BYTES, gid


def test_the_spread_trace_is_byte_identical_across_runs():
    """A load term is read state, so the tie it breaks has to be reproducible."""
    first, second = _run()[SPREAD], _run()[SPREAD]
    assert first.trace.render() == second.trace.render()
    assert first.trace.events == second.trace.events
