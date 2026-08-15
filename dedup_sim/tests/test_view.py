"""What dedup's one composed sensor does when it is not there, and when it is wrapped.

The fan-out tree is a sensor both links reach through the view they are attached to
(:mod:`dedup_sim.control._selector`), composed once in
:meth:`dedup_sim.control.routing.Dedup.attach`. Two failures have to be loud rather
than quiet: reading it when nobody composed it, and re-ranking a link whose answer
was already spent.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/tests/test_view.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from dedup_sim.control._selector import PlannedPeer
from dedup_sim.control._sensor import FanoutSensor
from dedup_sim.control._view import FanoutView
from proposed import Dispatcher
from proposed.selector import Discount, FirstMatch
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


def _view(directory=None, **sensors):
    """A view over ``directory``, or over none where nothing reads one."""
    return View(directory, {}).derived(FanoutView, **sensors)


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


def test_a_link_that_spends_a_slot_cannot_be_re_ranked():
    """The hazard the chain's shape exists to prevent, asserted mechanically.

    Answering consumes a fan-out slot, so a combinator that could reorder or drop that
    answer would spend the slot and hand the requester a source nothing planned. A
    reducer prices nothing, and a discount is arithmetic on a price, so the wrapping is
    refused the moment it answers rather than by a rule a reader has to remember.
    """
    fanout = FanoutSensor(fanout_cap=1)
    fanout.route("r0", "origin")                      # r0 joins, so it has a slot
    discounted = Discount(PlannedPeer(Dispatcher()))
    discounted.attach(_view(_Holds("r0"), fanout=fanout))

    with pytest.raises(ValueError, match="prices every source"):
        asyncio.run(discounted.select(["K"], "r1"))


def test_the_chain_holds_the_spending_link_at_its_head_unwrapped():
    """Head-of-chain is the position where spending and using coincide.

    A link behind one that can answer might never be asked; a link under a combinator
    might have its answer dropped. At the head of a FirstMatch, an answer wins.
    """
    peer = PlannedPeer(Dispatcher())
    chain = FirstMatch([peer])
    assert chain.selectors[0] is peer
    assert not isinstance(chain.selectors[0], Discount)
