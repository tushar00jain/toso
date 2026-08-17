"""What dedup's one composed sensor does when it is not there, and when it is wrapped.

The fan-out tree is a sensor both links reach through the view they are attached to
(:mod:`dedup_sim.control._selector`), composed once in
:meth:`dedup_sim.control.routing.Dedup.attach`. Two failures have to be loud rather
than quiet: reading it when nobody composed it, and a link losing its own read when a
combinator narrows the view above it.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests/test_view.py -q
"""

from __future__ import annotations

import pytest

from dedup_sim.control._selector import Candidates
from dedup_sim.control._sensor import Asked, FanoutSensor
from dedup_sim.control._view import DedupView
from proposed import Endpoint
from proposed.selector import Balance, FirstMatch, Ordered
from proposed.view import View


class _Holds:
    """A directory in which ``volume`` holds every key asked about.

    Enough for a link to answer with a source it need not wait for -- which is the
    precondition the wrapping below is refused on.
    """

    def __init__(self, volume: str) -> None:
        self.volume = volume

    def locate_raw(self, keys, missing_ok: bool = False):
        return {key: {self.volume: None} for key in keys}


#: What a view is built over. The chain reads no tier off it -- a hop costs what the
#: cost model below says -- so nothing these stage turns on the endpoints.
_TOPOLOGY = {
    v: Endpoint(id=v, host=v, node=v) for v in ("origin", "r0", "r1")
}


def _hop(src_id: str, dst_id: str, nbytes: int) -> float:
    """The origin far and the peers near: the shape a chain forms in, staged.

    A ranking in seconds needs a price for a hop, and what makes a peer worth reading
    from is that its link is the cheaper one.
    """
    if src_id == dst_id:
        return 0.0
    return 10.0 if src_id == "origin" else 1.0


def _view(directory=None, **sensors):
    """A view over ``directory``, or over none where nothing reads one."""
    return View(directory, _TOPOLOGY, _hop).derived(DedupView, **sensors)


def test_a_fanout_nobody_composed_raises():
    """An empty tree would answer "no peer is planned" and route every reader to the
    origin -- the m x fabric this capability exists to avoid, with nothing failing."""
    with pytest.raises(RuntimeError, match="fan-out"):
        _view().fanout


def test_a_keyword_no_sensor_claims_fails_where_the_view_is_built():
    """Refused at the one place a view is assembled, not on a later read."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _view(fanut=FanoutSensor(fanout_cap=1))


def test_a_composed_fanout_answers_for_itself():
    composed = _view(fanout=FanoutSensor(fanout_cap=2))
    assert composed.fanout.planned("r0") is None      # nothing routed yet


def test_the_ranking_still_senses_the_fanout_under_a_re_ranking():
    """A combinator narrows the view, and the ranking under it must keep its own read.

    Load spreading is a :data:`~proposed.selector.Balance` over this ranking, so the
    ranking is attached to whatever that combinator declared. It declares the ranking's
    reads as well as its own (:func:`proposed.selector.declares`), or the ranking would
    be handed a view with no fan-out in it and raise on the first peer it prices.
    """
    fanout = FanoutSensor(fanout_cap=1)
    fanout.route("r0", "origin")                      # r0 is a peer, and owes W
    fanout.folds[Asked](Asked("r0", ("K",)))
    ranking = Candidates()
    chain = Ordered(FirstMatch([Balance(ranking)]))
    chain.attach(_view(_Holds("origin"), fanout=fanout, load=fanout))

    assert ranking.view.fanout is fanout
    # Ordered by the chain's own last link, as the plane declares it: the ranking prices,
    # the stage appends, and neither orders.
    assert chain.select(["K"], "r1").sources == ("r0", "origin")
