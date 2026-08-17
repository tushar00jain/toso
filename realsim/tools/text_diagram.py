"""Render a strict XML-like DSL into monospaced Markdown diagrams.

Supported layout elements are ``stack``, ``row``, ``box``, ``text``, ``place-line``
and ``place-lines``. A row may put ``between`` elements between its blocks, with
``at`` children naming the rendered row where a connector belongs. Unknown elements
and attributes fail instead of being treated as HTML.

Run from the repository root::

    python -m realsim.tools.text_diagram path/to/diagrams.xml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple
from xml.etree import ElementTree

from realsim.tools.check_contract import REPO_ROOT

__all__ = []


@dataclass(frozen=True)
class _Block:
    """A rectangle of equally wide text lines."""

    lines: Tuple[str, ...]

    def __post_init__(self) -> None:
        widths = {len(line) for line in self.lines}
        if len(widths) > 1:
            raise ValueError(f"block lines have different widths: {sorted(widths)}")

    @property
    def width(self) -> int:
        return len(self.lines[0]) if self.lines else 0

    @property
    def height(self) -> int:
        return len(self.lines)

    def render(self) -> str:
        return "\n".join(line.rstrip() for line in self.lines).rstrip()


def _text(lines: Iterable[str], width: int | None = None) -> _Block:
    material = tuple(lines)
    wanted = max((len(line) for line in material), default=0) if width is None else width
    too_wide = [line for line in material if len(line) > wanted]
    if too_wide:
        raise ValueError(f"text exceeds width {wanted}: {too_wide[0]!r}")
    return _Block(tuple(line.ljust(wanted) for line in material))


def _title(title: str, width: int) -> str:
    label = f" {title} " if title else ""
    if len(label) > width:
        raise ValueError(f"box title exceeds width {width}: {title!r}")
    left = (width - len(label)) // 2
    return "─" * left + label + "─" * (width - left - len(label))


def _box(title: str, lines: Sequence[str], width: int | None = None) -> _Block:
    material = tuple(lines)
    minimum = max(
        len(f" {title} ") if title else 0,
        max((len(f" {line}") for line in material), default=0),
    )
    inner = minimum if width is None else width
    if inner < minimum:
        raise ValueError(f"box {title!r} needs width {minimum}, got {inner}")
    body = tuple("│" + f" {line}".ljust(inner) + "│" for line in material)
    return _Block(("┌" + _title(title, inner) + "┐", *body, "└" + "─" * inner + "┘"))


def _beside(
    blocks: Sequence[_Block],
    *,
    gap: int = 2,
    links: Mapping[Tuple[int, int], str] | None = None,
) -> _Block:
    if not blocks:
        return _Block(())
    links = {} if links is None else links
    height = max(block.height for block in blocks)
    gap_widths = [gap] * (len(blocks) - 1)
    for (gap_index, _row), label in links.items():
        gap_widths[gap_index] = max(gap_widths[gap_index], len(label))
    rows = []
    for row in range(height):
        parts = []
        for index, block in enumerate(blocks):
            parts.append(block.lines[row] if row < block.height else " " * block.width)
            if index < len(gap_widths):
                parts.append(links.get((index, row), "").center(gap_widths[index]))
        rows.append("".join(parts))
    return _Block(tuple(rows))


def _stack(blocks: Sequence[_Block], *, align: str = "left", gap: int = 0) -> _Block:
    if not blocks:
        return _Block(())
    width = max(block.width for block in blocks)
    rows = []
    for index, block in enumerate(blocks):
        for row in block.lines:
            if align == "left":
                rows.append(row.ljust(width))
            elif align == "center":
                rows.append(row.center(width))
            elif align == "right":
                rows.append(row.rjust(width))
            else:
                raise ValueError(f"unknown alignment: {align!r}")
        if index < len(blocks) - 1:
            rows.extend(" " * width for _ in range(gap))
    return _Block(tuple(rows))


def _place_line(width: int, placements: Mapping[int, str]) -> _Block:
    chars = [" "] * width
    for column, value in sorted(placements.items()):
        if column < 0 or column + len(value) > width:
            raise ValueError(f"{value!r} does not fit at column {column} in {width}")
        if any(char != " " for char in chars[column:column + len(value)]):
            raise ValueError(f"{value!r} overlaps another placement at column {column}")
        chars[column:column + len(value)] = value
    return _Block(("".join(chars),))


def _place_lines(width: int, placements: Mapping[int, Sequence[str]]) -> _Block:
    height = max((len(lines) for lines in placements.values()), default=0)
    rows = []
    for row in range(height):
        values = {
            column: lines[row]
            for column, lines in placements.items()
            if row < len(lines)
        }
        rows.extend(_place_line(width, values).lines)
    return _Block(tuple(rows))


def _attrs(element: ElementTree.Element, allowed: Sequence[str]) -> None:
    unknown = sorted(set(element.attrib) - set(allowed))
    if unknown:
        raise ValueError(f"<{element.tag}> has unknown attributes: {', '.join(unknown)}")


def _int(element: ElementTree.Element, name: str, default: int | None = None) -> int:
    raw = element.get(name)
    if raw is None:
        if default is None:
            raise ValueError(f"<{element.tag}> requires {name!r}")
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"<{element.tag}> {name!r} must be an integer: {raw!r}") from error


def _lines(element: ElementTree.Element) -> Tuple[str, ...]:
    lines = []
    for child in element:
        if child.tag != "line" or child.attrib or len(child):
            raise ValueError(f"<{element.tag}> accepts only plain <line> children")
        lines.append(child.text or "")
    return tuple(lines)


def _render_row(element: ElementTree.Element) -> _Block:
    _attrs(element, ("gap",))
    blocks = []
    links: Dict[Tuple[int, int], str] = {}
    pending: Dict[int, str] | None = None
    for child in element:
        if child.tag == "between":
            if not blocks or pending is not None:
                raise ValueError("<between> must occur once between two row blocks")
            _attrs(child, ())
            pending = {}
            for placement in child:
                if placement.tag != "at" or len(placement):
                    raise ValueError("<between> accepts only <at row=...> children")
                _attrs(placement, ("row",))
                row = _int(placement, "row")
                if row in pending:
                    raise ValueError(f"<between> has two connectors on row {row}")
                pending[row] = placement.text or ""
            continue
        if pending is not None:
            gap_index = len(blocks) - 1
            links.update({
                (gap_index, row): value for row, value in pending.items()
            })
            pending = None
        blocks.append(_render_element(child))
    if pending is not None:
        raise ValueError("<between> must be followed by a row block")
    return _beside(blocks, gap=_int(element, "gap", 2), links=links)


def _render_element(element: ElementTree.Element) -> _Block:
    if element.tag == "stack":
        _attrs(element, ("align", "gap"))
        return _stack(
            [_render_element(child) for child in element],
            align=element.get("align", "left"),
            gap=_int(element, "gap", 0),
        )
    if element.tag == "row":
        return _render_row(element)
    if element.tag == "box":
        _attrs(element, ("title", "width"))
        width = _int(element, "width") if element.get("width") is not None else None
        return _box(element.get("title", ""), _lines(element), width)
    if element.tag == "text":
        _attrs(element, ("width",))
        width = _int(element, "width") if element.get("width") is not None else None
        return _text(_lines(element), width)
    if element.tag == "place-line":
        _attrs(element, ("width",))
        placements = {}
        for child in element:
            if child.tag != "at" or len(child):
                raise ValueError("<place-line> accepts only <at column=...> children")
            _attrs(child, ("column",))
            column = _int(child, "column")
            if column in placements:
                raise ValueError(f"<place-line> has two values at column {column}")
            placements[column] = child.text or ""
        return _place_line(_int(element, "width"), placements)
    if element.tag == "place-lines":
        _attrs(element, ("width",))
        placements = {}
        for child in element:
            if child.tag != "at":
                raise ValueError("<place-lines> accepts only <at column=...> children")
            _attrs(child, ("column",))
            column = _int(child, "column")
            if column in placements:
                raise ValueError(f"<place-lines> has two groups at column {column}")
            placements[column] = _lines(child)
        return _place_lines(_int(element, "width"), placements)
    raise ValueError(f"unknown diagram element <{element.tag}>")


def _load(source: Path) -> Tuple[Path, Dict[str, _Block]]:
    root = ElementTree.parse(source).getroot()
    if root.tag != "diagrams":
        raise ValueError("diagram source root must be <diagrams>")
    _attrs(root, ("target",))
    target = root.get("target")
    if target is None:
        raise ValueError("<diagrams> requires 'target'")
    drawings = {}
    for element in root:
        if element.tag != "diagram":
            raise ValueError("<diagrams> accepts only <diagram> children")
        _attrs(element, ("id",))
        name = element.get("id")
        if not name:
            raise ValueError("<diagram> requires a non-empty 'id'")
        if name in drawings:
            raise ValueError(f"duplicate diagram id {name!r}")
        if len(element) != 1:
            raise ValueError(f"<diagram id={name!r}> must contain one layout element")
        drawings[name] = _render_element(element[0])
    return Path(target), drawings


def _replace(document: str, name: str, drawing: _Block) -> str:
    start = f"<!-- text-diagram:{name}:start -->"
    end = f"<!-- text-diagram:{name}:end -->"
    before, found, rest = document.partition(start)
    if not found:
        raise ValueError(f"missing marker {start}")
    _old, found, after = rest.partition(end)
    if not found:
        raise ValueError(f"missing marker {end}")
    generated = f"{start}\n```\n{drawing.render()}\n```\n{end}"
    return before + generated + after


def _render(document: str, drawings: Mapping[str, _Block]) -> str:
    for name, drawing in drawings.items():
        document = _replace(document, name, drawing)
    return document


def _main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m realsim.tools.text_diagram SOURCE.xml", file=sys.stderr)
        return 2
    source = Path(args[0]).resolve()
    target_rel, drawings = _load(source)
    target = (REPO_ROOT / target_rel).resolve()
    try:
        target.relative_to(REPO_ROOT)
    except ValueError:
        raise ValueError(f"diagram target leaves the repository: {target}")
    target.write_text(_render(target.read_text(), drawings))
    print(f"rendered {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
