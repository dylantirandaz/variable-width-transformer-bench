#!/usr/bin/env python3
"""Fetch and extract the White Nights story from Project Gutenberg HTML."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterable, Optional
from urllib.request import urlopen


DEFAULT_URL = "https://www.gutenberg.org/files/36034/36034-h/36034-h.htm"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "tiny_corpus.txt"
START_MARKER = '<a name="Pg1">WHITE NIGHTS</a>'
END_MARKER = "NOTES FROM UNDERGROUND"


def main() -> None:
    args = parse_args()
    html = read_html(args.url, args.input)
    text = extract_white_nights(html)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output} ({len(text.encode('utf-8')):,} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Project Gutenberg HTML URL.")
    parser.add_argument("--input", default=None, help="Use a local HTML file instead of fetching --url.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output UTF-8 corpus path.")
    return parser.parse_args()


def read_html(url: str, input_path: Optional[str]) -> str:
    if input_path:
        return Path(input_path).read_text(encoding="iso-8859-1")
    with urlopen(url, timeout=30) as response:
        return response.read().decode("iso-8859-1")


def extract_white_nights(html: str) -> str:
    start = html.find(START_MARKER)
    if start < 0:
        raise ValueError(f"could not find start marker {START_MARKER!r}")
    start = html.rfind("<p", 0, start)
    if start < 0:
        raise ValueError("could not find containing paragraph for story title")
    end = html.find(END_MARKER, start)
    if end < 0:
        raise ValueError(f"could not find end marker {END_MARKER!r}")

    fragment = html[start:end]
    parser = ParagraphTextParser()
    parser.feed(fragment)
    parser.close()
    paragraphs = [normalize_text(part) for part in parser.blocks]
    paragraphs = [part for part in paragraphs if part]
    return "\n\n".join(paragraphs).strip() + "\n"


def normalize_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n+", " ", value)
    return value.strip()


class ParagraphTextParser(HTMLParser):
    block_tags = {"p", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._tag_stack: list[str] = []
        self._current: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.block_tags:
            self._flush()
            self._tag_stack.append(tag)
        elif tag == "br" and self._tag_stack:
            self._current.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self.block_tags and tag in self._tag_stack:
            while self._tag_stack:
                popped = self._tag_stack.pop()
                if popped == tag:
                    break
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._tag_stack:
            return
        self._current.append(data)

    def _flush(self) -> None:
        text = "".join(self._current)
        self._current = []
        if text.strip():
            self.blocks.append(text)


if __name__ == "__main__":
    main()
