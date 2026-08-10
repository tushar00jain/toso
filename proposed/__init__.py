"""The surface proposed for torchstore itself.

Everything in this package is a component a real deployment would need but
torchstore does not have today. It is kept apart from ``realsim`` so the upstream
ask is legible at a glance: ``realsim`` is scaffolding that disappears outside the
simulator, ``proposed`` is the design being argued for.

* :mod:`proposed.policy` -- :class:`~proposed.policy.Policy`
  (``select`` / ``notice``) and :class:`~proposed.policy.Selection`. A controller
  consults a policy inside ``locate_volumes`` to decide *which* volume serves a
  requester, and may withhold the answer until that volume is usable. The default
  :class:`~proposed.policy.NaivePolicy` returns the directory's own answer, so an
  installed policy changes nothing until one is written.
* :mod:`proposed.view` -- :class:`~proposed.view.View`, the read-only observation
  a controller hands a policy: who holds a key, where volumes are, what time it is.

Import rule, enforced by ``realsim/tools/check_contract.py``: **this package may
not import** ``realsim`` **or the capability packages.** That is what keeps the
proposal honest -- if it needed the simulator, it could not be implemented inside
torchstore. It may use ``sim_common.topology`` for the locality types, which are
themselves part of the ask (gap 4).

The gaps each piece answers are listed in the design doc's "What torchstore is
missing" section.
"""
