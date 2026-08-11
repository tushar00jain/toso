"""Turning a finished run into text: :class:`Report`.

One method, so every capability's ``report/`` package exposes the same thing and
a demo can render any of them without knowing which capability it is holding.

A report owns no run state. It is constructed from the results it describes (and
whatever scenario facts it needs, which live on the workload those results carry)
and answers :meth:`render`. Keeping it a type rather than a bare function is what
lets :class:`~realsim.demo.Demo` treat all three sims identically.

Named ``reporting`` rather than ``report`` so it cannot be confused with
:mod:`sim_common.report`, which is the *measurement* side -- ``Ledger``, outcome
rows, the source->dest tree renderer -- that these reports read from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Report"]


class Report(ABC):
    """What a run produced, as text."""

    @abstractmethod
    def render(self) -> str:
        """The rendered summary, ready to log."""
