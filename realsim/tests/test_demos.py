"""Every sim's demo is a real :class:`realsim.demo.Demo`, and it runs.

Two things no lint can check as well as construction and execution can:

* the :class:`~realsim.demo.Demo` ABC refuses to instantiate a subclass that has
  not implemented :meth:`~realsim.demo.Demo.scenarios`, and a demo without a
  ``name``/``description`` is caught here -- so "declared its parts" is enforced
  by the type, not by convention;
* until now **no test ran a demo**, so the CLI paths, the scenario narration and
  the reports were exercised only by a human typing ``python -m``. Every scenario
  of every sim runs below, which is also what makes the ABC's enforcement fire in
  CI rather than at someone's terminal.

Outcomes are asserted in each capability's own tests; this asserts the demo layer
holds together.
"""

from __future__ import annotations

import logging

import pytest

from dedup_sim.__main__ import DedupDemo
from kvcache_sim.__main__ import KVCacheDemo
from putget_sim.__main__ import PutGetDemo
import argparse

from realsim.demo import _add_run_flags, Console, Demo, Scenario
from realsim.run import Run

DEMOS = [PutGetDemo, DedupDemo, KVCacheDemo]
DEMO_IDS = [d.__name__ for d in DEMOS]


@pytest.fixture(autouse=True)
def _quiet_logging():
    """Keep demo output out of the test log; the demos log at INFO."""
    root = logging.getLogger()
    before = root.level, list(root.handlers)
    root.setLevel(logging.CRITICAL)
    yield
    root.setLevel(before[0])
    root.handlers[:] = before[1]


@pytest.mark.parametrize("demo_cls", DEMOS, ids=DEMO_IDS)
def test_every_demo_declares_its_parts(demo_cls):
    """Constructing it proves the ABC's requirements are met."""
    demo = demo_cls()
    assert demo.name and demo.name.endswith("_sim")
    assert demo.description
    scenarios = list(demo.scenarios())
    assert scenarios, "a demo with no scenarios can do nothing"
    assert all(isinstance(s, Scenario) for s in scenarios)
    # Names are the CLI's choices, so they must be distinct.
    names = [s.name for s in scenarios]
    assert len(set(names)) == len(names)
    # Declaring and narrating are separate: every scenario supplies both.
    assert all(callable(s.runs) and callable(s.show) for s in scenarios)


def test_an_incomplete_demo_cannot_be_constructed():
    """The ABC is the enforcement; this pins that it actually refuses."""

    class Incomplete(Demo):
        name = "nope_sim"
        description = "declares no scenarios"

    with pytest.raises(TypeError, match="scenarios"):
        Incomplete()


@pytest.mark.parametrize("demo_cls", DEMOS, ids=DEMO_IDS)
def test_every_demo_runs_end_to_end(demo_cls):
    """``python -m <sim>`` with no arguments: every scenario, then the takeaway."""
    demo_cls().main([])


@pytest.mark.parametrize("demo_cls", DEMOS, ids=DEMO_IDS)
def test_every_scenario_is_selectable(demo_cls):
    """Each named scenario runs on its own, as the CLI's positional arg."""
    demo = demo_cls()
    scenarios = list(demo.scenarios())
    if len(scenarios) < 2:
        pytest.skip(f"{demo.name} has a single scenario and no positional arg")
    for scenario in scenarios:
        demo_cls().main([scenario.name])


@pytest.mark.parametrize("demo_cls", DEMOS, ids=DEMO_IDS)
def test_declaring_a_scenario_executes_nothing(demo_cls):
    """``Scenario.runs`` returns configurations; only ``Demo.main`` runs them.

    A declared ``Run`` has no ``Simulation`` behind it yet, so nothing has
    touched a clock or a directory at declaration time.
    """
    demo = demo_cls()
    parser = argparse.ArgumentParser()
    demo.flags(parser)
    _add_run_flags(parser)
    if len(list(demo.scenarios())) > 1:
        parser.add_argument("scenario", nargs="?")
    args = parser.parse_args([])
    for scenario in demo.scenarios():
        runs = scenario.runs(args)
        assert runs, f"{demo.name}:{scenario.name} declared no runs"
        assert all(isinstance(r, Run) for r in runs)
        # Labels are how a report tells the configurations apart.
        labels = [r.label for r in runs]
        assert all(labels) and len(set(labels)) == len(labels)
        # Nothing has been assembled: a declared Run has no Simulation yet.
        assert all(r.plane is None or callable(r.plane) for r in runs)


def test_console_renders_a_report_once():
    """The one summary path every demo uses."""
    from realsim.run import Report

    class Fixed(Report):
        def render(self) -> str:
            return "rendered"

    logger = logging.getLogger("test_console")
    logged = []
    handler = logging.Handler()
    handler.emit = lambda record: logged.append(record.getMessage())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        Console(logger).summary(Fixed())
    finally:
        logger.removeHandler(handler)
    assert logged == ["(b) summary", "rendered"]
