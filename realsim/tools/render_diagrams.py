#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render diagram-like Markdown code blocks as PNG files.

Run:
    python render_diagrams.py README.md
    python render_diagrams.py README.md --output-dir rendered-diagrams --copy 1
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:
    raise SystemExit(
        "Pillow is required. Install it in the active virtual environment."
    ) from error


_DIAGRAM_MARKERS = frozenset("─│┌┐└┘├┤┬┴┼▶◀▲▼═║╔╗╚╝")
_UNICODE_RENDER_TRANSLATION = str.maketrans({"▶": ">", "◀": "<"})
_ASCII_TRANSLATION = str.maketrans(
    {
        "─": "-",
        "═": "=",
        "│": "|",
        "║": "|",
        "┌": "+",
        "┐": "+",
        "└": "+",
        "┘": "+",
        "├": "+",
        "┤": "+",
        "┬": "+",
        "┴": "+",
        "┼": "+",
        "╔": "+",
        "╗": "+",
        "╚": "+",
        "╝": "+",
        "▶": ">",
        "◀": "<",
        "▲": "^",
        "▼": "v",
        "→": ">",
        "←": "<",
        "–": "-",
        "—": "-",
    }
)


@dataclass(frozen=True)
class Diagram:
    heading: str
    text: str


def _looks_like_diagram(text: str) -> bool:
    return any(character in text for character in _DIAGRAM_MARKERS) or any(
        marker in text for marker in ("-->", "<--", "+--", "|  ")
    )


def _diagrams(markdown: str) -> list[Diagram]:
    """Return unlabeled or ``text`` code blocks that look like diagrams."""
    diagrams: list[Diagram] = []
    heading = "diagram"
    fence: str | None = None
    language = ""
    lines: list[str] = []

    for line in markdown.splitlines():
        if fence is None:
            heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if heading_match:
                heading = heading_match.group(1)

            fence_match = re.match(r"^(`{3,}|~{3,})(.*)$", line)
            if fence_match:
                fence = fence_match.group(1)
                language = fence_match.group(2).strip().lower()
                lines = []
            continue

        if line.strip() == fence:
            text = "\n".join(lines).strip("\n").expandtabs(4)
            if language in {"", "text"} and _looks_like_diagram(text):
                diagrams.append(Diagram(heading=heading, text=text))
            fence = None
            language = ""
            lines = []
        else:
            lines.append(line)

    if fence is not None:
        raise ValueError("unterminated Markdown code fence")
    return diagrams


def _slug(text: str) -> str:
    text = re.sub(r"[`*_]", "", text).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "diagram"


def _font(path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        path,
        Path("/usr/share/fonts/google-noto-vf/NotoSansMono-VF.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
        Path("/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf"),
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("C:/Windows/Fonts/consola.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("no monospace font found; pass one with --font")


def _render(
    diagram: Diagram,
    destination: Path,
    *,
    font: ImageFont.FreeTypeFont,
    padding: int,
    line_spacing: int,
    ascii_only: bool,
) -> None:
    translation = _ASCII_TRANSLATION if ascii_only else _UNICODE_RENDER_TRANSLATION
    text = diagram.text.translate(translation)
    probe = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=line_spacing
    )
    image = Image.new(
        "RGB",
        (right - left + 2 * padding, bottom - top + 2 * padding),
        "white",
    )
    ImageDraw.Draw(image).multiline_text(
        (padding - left, padding - top),
        text,
        fill="black",
        font=font,
        spacing=line_spacing,
    )
    image.save(destination, dpi=(192, 192))


def _copy_to_clipboard(path: Path) -> None:
    if command := shutil.which("wl-copy"):
        subprocess.run(
            [command, "--type", "image/png"], input=path.read_bytes(), check=True
        )
        return
    if command := shutil.which("xclip"):
        subprocess.run(
            [command, "-selection", "clipboard", "-t", "image/png", "-i", str(path)],
            check=True,
        )
        return
    raise RuntimeError("clipboard copy requires wl-copy or xclip")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("rendered-diagrams"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int, default=30)
    parser.add_argument("--padding", type=int, default=32)
    parser.add_argument("--line-spacing", type=int, default=0)
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="replace Unicode box-drawing characters before rendering",
    )
    parser.add_argument(
        "--copy",
        type=int,
        metavar="INDEX",
        help="copy the selected 1-based output image to the clipboard",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.font_size < 1 or args.padding < 0 or args.line_spacing < 0:
        raise SystemExit("font size must be positive; padding and spacing cannot be negative")

    diagrams = _diagrams(args.markdown.read_text(encoding="utf-8"))
    if not diagrams:
        raise SystemExit(f"no diagram-like code blocks found in {args.markdown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = _font(args.font, args.font_size)
    outputs: list[Path] = []
    for index, diagram in enumerate(diagrams, start=1):
        destination = args.output_dir / f"{index:02d}-{_slug(diagram.heading)}.png"
        _render(
            diagram,
            destination,
            font=font,
            padding=args.padding,
            line_spacing=args.line_spacing,
            ascii_only=args.ascii,
        )
        outputs.append(destination)
        print(destination)

    if args.copy is not None:
        if not 1 <= args.copy <= len(outputs):
            raise SystemExit(f"--copy must be between 1 and {len(outputs)}")
        _copy_to_clipboard(outputs[args.copy - 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
