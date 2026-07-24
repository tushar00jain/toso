"""Tests for the process-wide simulation config (sourcing + scoping).

Run from the worktree with the venv interpreter::

    PYTHONPATH=. /path/to/.venv/bin/python -m pytest sim_common/tests/test_config.py -q

Also runnable as a plain script if pytest is unavailable. Each test that mutates
the environment or the global config restores it in a ``finally`` so the suite
stays order-independent (there is no pytest fixture, to keep the script fallback).
"""

from __future__ import annotations

import os

from sim_common import config

_ENV = "TOSO_FINGERPRINT"


def _reset() -> None:
    """Clear the env override and reload the config to defaults."""
    os.environ.pop(_ENV, None)
    config.configure()


# --------------------------------------------------------------------------
# Defaults + scoped overrides.
# --------------------------------------------------------------------------


def test_default_is_off():
    _reset()
    try:
        assert config.current().fingerprint is False
    finally:
        _reset()


def test_overrides_scopes_and_restores():
    _reset()
    try:
        assert config.current().fingerprint is False
        with config.overrides(fingerprint=True):
            assert config.current().fingerprint is True
        assert config.current().fingerprint is False
    finally:
        _reset()


# --------------------------------------------------------------------------
# Sourcing precedence: CLI override > env > default.
# --------------------------------------------------------------------------


def test_configure_explicit_override():
    _reset()
    try:
        config.configure(fingerprint=True)
        assert config.current().fingerprint is True
        # None defers to env/default rather than forcing off.
        config.configure(fingerprint=None)
        assert config.current().fingerprint is False
    finally:
        _reset()


def test_env_enables_and_is_parsed():
    _reset()
    try:
        for raw in ("1", "true", "YES", "On"):
            os.environ[_ENV] = raw
            config.configure()
            assert config.current().fingerprint is True, raw
        for raw in ("0", "false", "no", "off", ""):
            os.environ[_ENV] = raw
            config.configure()
            assert config.current().fingerprint is False, raw
    finally:
        _reset()


def test_explicit_override_beats_env():
    _reset()
    try:
        os.environ[_ENV] = "0"
        # An explicit True wins over the env's false.
        config.configure(fingerprint=True)
        assert config.current().fingerprint is True
        # An unset (None) CLI flag defers to the env's true.
        os.environ[_ENV] = "1"
        config.configure(fingerprint=None)
        assert config.current().fingerprint is True
    finally:
        _reset()


# --------------------------------------------------------------------------
# Contention flag: default, env sourcing, override precedence.
# --------------------------------------------------------------------------

_CONTENTION_ENV = "TOSO_CONTENTION"


def _reset_contention() -> None:
    os.environ.pop(_CONTENTION_ENV, None)
    config.configure()


def test_contention_default_is_none():
    _reset_contention()
    try:
        assert config.current().contention == "none"
    finally:
        _reset_contention()


def test_contention_env_is_parsed_and_lowercased():
    _reset_contention()
    try:
        for raw, want in (("serialize", "serialize"), ("PROGRESSIVE", "progressive"),
                          (" none ", "none")):
            os.environ[_CONTENTION_ENV] = raw
            config.configure()
            assert config.current().contention == want, raw
    finally:
        _reset_contention()


def test_contention_explicit_override_beats_env():
    _reset_contention()
    try:
        os.environ[_CONTENTION_ENV] = "serialize"
        # An explicit CLI value wins over the env.
        config.configure(contention="progressive")
        assert config.current().contention == "progressive"
        # An unset (None) CLI flag defers to the env.
        config.configure(contention=None)
        assert config.current().contention == "serialize"
    finally:
        _reset_contention()


def test_contention_overrides_scopes_and_restores():
    _reset_contention()
    try:
        assert config.current().contention == "none"
        with config.overrides(contention="progressive"):
            assert config.current().contention == "progressive"
        assert config.current().contention == "none"
    finally:
        _reset_contention()


# --------------------------------------------------------------------------
# Collapse-charges flag: default, env sourcing, override precedence, scoping.
# --------------------------------------------------------------------------

_COLLAPSE_ENV = "TOSO_COLLAPSE_CHARGES"


def _reset_collapse() -> None:
    os.environ.pop(_COLLAPSE_ENV, None)
    config.configure()


def test_collapse_default_is_off():
    _reset_collapse()
    try:
        assert config.current().collapse_charges is False
    finally:
        _reset_collapse()


def test_collapse_env_is_parsed():
    _reset_collapse()
    try:
        for raw in ("1", "true", "YES", "On"):
            os.environ[_COLLAPSE_ENV] = raw
            config.configure()
            assert config.current().collapse_charges is True, raw
        for raw in ("0", "false", "no", "off", ""):
            os.environ[_COLLAPSE_ENV] = raw
            config.configure()
            assert config.current().collapse_charges is False, raw
    finally:
        _reset_collapse()


def test_collapse_explicit_override_beats_env():
    _reset_collapse()
    try:
        os.environ[_COLLAPSE_ENV] = "0"
        # An explicit True wins over the env's false.
        config.configure(collapse_charges=True)
        assert config.current().collapse_charges is True
        # An unset (None) CLI flag defers to the env's true.
        os.environ[_COLLAPSE_ENV] = "1"
        config.configure(collapse_charges=None)
        assert config.current().collapse_charges is True
    finally:
        _reset_collapse()


def test_collapse_overrides_scopes_and_restores():
    _reset_collapse()
    try:
        assert config.current().collapse_charges is False
        with config.overrides(collapse_charges=True):
            assert config.current().collapse_charges is True
        assert config.current().collapse_charges is False
    finally:
        _reset_collapse()


# --------------------------------------------------------------------------
# Script fallback (no pytest required).
# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
