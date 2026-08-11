"""A lightweight dict-backed stand-in for the controller's ``Trie`` directory.

The real ``Controller`` keeps its directory in a
``torchstore.storage_utils.trie.Trie`` (a ``pygtrie.StringTrie`` wrapper). That
trie is the one part of the off-actor directory path that costs real work per
key -- a node walk + allocation on every put/locate/delete -- yet the
``Controller``'s decision logic touches it only through the plain ``Mapping``
surface plus one extra affordance: ``keys().filter_by_prefix(...)``.

:class:`DictDirectory` supplies exactly that surface backed by a plain ``dict``,
so it drops into ``Controller.keys_to_storage_volumes`` unchanged and lets every
bit of the real ``Controller`` decision logic run over it (see
:class:`realsim.adapters.real_controller.ShimControllerAdapter`). The only
behaviour it must reproduce faithfully is the trie's prefix filter; iteration
*order* differs from the trie's DFS order, but the sims never consume directory
order in a metric (``locate_volumes`` iterates the caller's key list, not the
directory), so metrics stay byte-identical -- which the divergence-gate tests
assert live.

Prefix semantics (mirrored from ``Trie``): the trie splits keys into tokens on a
separator (``"."`` by default) and matches a prefix token-wise, so prefix
``"a.b"`` matches ``"a.b"`` and ``"a.b.c"`` but not ``"a.bc"``; a prefix that no
key extends yields ``[]`` (the real ``TrieKeysView.filter_by_prefix`` swallows
``pygtrie``'s ``KeyError`` and returns ``[]``). :meth:`_PrefixKeysView.filter_by_prefix`
reproduces that token-wise rule verbatim.
"""

from __future__ import annotations

from collections.abc import Iterator, KeysView, MutableMapping
from typing import Any

__all__ = ["DictDirectory"]


class _PrefixKeysView(KeysView):
    """A ``KeysView`` that adds ``Trie``-compatible prefix filtering.

    Mirrors :class:`torchstore.storage_utils.trie.TrieKeysView`: it iterates the
    directory's keys and additionally exposes :meth:`filter_by_prefix`.
    """

    def __init__(self, directory: "DictDirectory") -> None:
        super().__init__(directory)
        self._directory = directory

    def filter_by_prefix(self, prefix: str) -> list[str]:
        """Return keys whose token sequence starts with ``prefix``'s tokens.

        Token-wise match on the directory separator, reproducing
        ``pygtrie.StringTrie.iterkeys(prefix=...)`` (a prefix that no key extends
        yields ``[]``, matching ``TrieKeysView.filter_by_prefix``).
        """
        sep = self._directory.separator
        prefix_tokens = prefix.split(sep)
        n = len(prefix_tokens)
        return [
            key for key in self._directory if key.split(sep)[:n] == prefix_tokens
        ]


class DictDirectory(MutableMapping):
    """A plain-``dict`` directory with the ``Trie`` surface the ``Controller`` uses.

    Implements the ``MutableMapping`` protocol plus a :meth:`keys` view that
    supports ``filter_by_prefix``, so it is a drop-in for
    ``Controller.keys_to_storage_volumes`` (a
    :class:`torchstore.storage_utils.trie.Trie`) without the per-key trie tax.

    Args:
        separator: token separator for prefix matching (``"."``, matching the
            ``Trie`` default the real ``Controller`` constructs).
    """

    def __init__(self, *, separator: str = ".") -> None:
        self._data: dict[str, Any] = {}
        self.separator = separator

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        # Override the MutableMapping default (a try/except getitem) with the
        # dict's own membership test -- the hot check on the put/locate path.
        return key in self._data

    def keys(self) -> _PrefixKeysView:  # type: ignore[override]
        """Return a keys view that supports ``filter_by_prefix`` (like ``Trie``)."""
        return _PrefixKeysView(self)

    def __repr__(self) -> str:
        return f"DictDirectory({self._data!r})"
