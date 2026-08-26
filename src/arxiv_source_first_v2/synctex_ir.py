"""Read-only SyncTeX source-to-page metadata for source-first page GT.

SyncTeX is emitted by the TeX engine while it lays out the *clean* document.
Unlike a colored locator shadow it does not insert visible material or TeX
groups into the author's source.  This module deliberately exposes only
source file/line and geometric records; it never reads PDF text.

The parser implements the small, stable subset of the SyncTeX v1 text format
needed by the experimental pipeline.  Unknown records are ignored.  Inputs
outside the caller supplied project root are retained in the parsed index but
can never be selected by :func:`alignment_for_probes`, whose probes already
come from the executed project-source allow-list.
"""

from __future__ import annotations

import bisect
import dataclasses
import gzip
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# TeX writes coordinates in scaled points.  PDF points use 72 rather than
# 72.27 points per inch, hence 65536 * 72 / 72.27.
SCALED_POINTS_PER_PDF_POINT = 65536.0 * 72.0 / 72.27

_INPUT_RE = re.compile(r"^Input:(\d+):(.*)$")
_SHEET_RE = re.compile(r"^\{(\d+)\s*$")
_RECORD_RE = re.compile(
    r"^(?P<kind>[\[\(vhxkg$])"
    r"(?P<tag>-?\d+),(?P<line>-?\d+):"
    r"(?P<x>-?\d+),(?P<y>-?\d+)"
    r"(?::(?P<tail>-?\d+(?:,-?\d+){0,2}))?\s*$"
)
_LEAF_KINDS = frozenset({"x", "k", "g"})


@dataclasses.dataclass(frozen=True)
class SyncTeXPoint:
    source_file: Path
    source_line: int
    page_number: int
    x: float
    y: float
    kind: str
    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0

    def as_json(self, root: Path | None = None) -> dict[str, Any]:
        source = self.source_file
        if root is not None:
            try:
                source_value = source.relative_to(root).as_posix()
            except ValueError:
                source_value = str(source)
        else:
            source_value = str(source)
        return {
            "source_file": source_value,
            "source_line": self.source_line,
            "page_number": self.page_number,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "kind": self.kind,
            "width": round(self.width, 6),
            "height": round(self.height, 6),
            "depth": round(self.depth, 6),
        }


@dataclasses.dataclass(frozen=True)
class SyncTeXIndex:
    source_path: Path
    inputs: Mapping[int, Path]
    by_line: Mapping[tuple[Path, int], tuple[SyncTeXPoint, ...]]
    pages: tuple[int, ...]
    records_seen: int
    records_indexed: int

    def points_for_line(
        self,
        source_file: str | os.PathLike[str],
        line: int,
    ) -> tuple[SyncTeXPoint, ...]:
        source = Path(source_file).expanduser().resolve(strict=False)
        return self.by_line.get((source, int(line)), ())

    def points_for_lines(
        self,
        source_file: str | os.PathLike[str],
        lines: Iterable[int],
    ) -> tuple[SyncTeXPoint, ...]:
        output: list[SyncTeXPoint] = []
        for line in sorted({int(value) for value in lines if int(value) > 0}):
            output.extend(self.points_for_line(source_file, line))
        return tuple(output)

    def as_json(self, project_root: Path | None = None) -> dict[str, Any]:
        project_inputs = 0
        if project_root is not None:
            root = project_root.resolve()
            for path in self.inputs.values():
                try:
                    path.relative_to(root)
                except ValueError:
                    continue
                project_inputs += 1
        return {
            "source": str(self.source_path),
            "format": "synctex_v1_source_line_geometry",
            "inputs": len(self.inputs),
            "project_inputs": project_inputs,
            "pages": list(self.pages),
            "records_seen": self.records_seen,
            "records_indexed": self.records_indexed,
            "source_lines_indexed": len(self.by_line),
            "pdf_text_used": False,
        }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _resolve_input(value: str, *, source_root: Path, synctex_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    # Engine-produced relative paths are normally relative to the source CWD.
    # Prefer that interpretation, then the metadata directory for unusual
    # wrappers that place their inputs beside the build output.
    source_candidate = (source_root / candidate).resolve(strict=False)
    if source_candidate.exists():
        return source_candidate
    return (synctex_path.parent / candidate).resolve(strict=False)


def parse_synctex(
    path: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str],
) -> SyncTeXIndex:
    """Parse source-line records from an uncompressed or gzip SyncTeX file."""

    synctex_path = Path(path).expanduser().resolve()
    root = Path(source_root).expanduser().resolve()
    if not synctex_path.is_file():
        raise FileNotFoundError(synctex_path)
    inputs: dict[int, Path] = {}
    raw_records: list[tuple[int, int, int, str, int, int, tuple[int, ...]]] = []
    current_page: int | None = None
    records_seen = 0
    with _open_text(synctex_path) as handle:
        for raw_line in handle:
            value = raw_line.rstrip("\r\n")
            input_match = _INPUT_RE.match(value)
            if input_match:
                tag = int(input_match.group(1))
                inputs[tag] = _resolve_input(
                    input_match.group(2),
                    source_root=root,
                    synctex_path=synctex_path,
                )
                continue
            sheet_match = _SHEET_RE.match(value)
            if sheet_match:
                current_page = int(sheet_match.group(1))
                continue
            if value == "}":
                current_page = None
                continue
            if current_page is None:
                continue
            record = _RECORD_RE.match(value)
            if record is None:
                continue
            records_seen += 1
            tail = tuple(
                int(item)
                for item in (record.group("tail") or "").split(",")
                if item
            )
            raw_records.append(
                (
                    int(record.group("tag")),
                    int(record.group("line")),
                    current_page,
                    record.group("kind"),
                    int(record.group("x")),
                    int(record.group("y")),
                    tail,
                )
            )

    grouped: dict[tuple[Path, int], list[SyncTeXPoint]] = defaultdict(list)
    records_indexed = 0
    for tag, source_line, page_number, kind, raw_x, raw_y, tail in raw_records:
        source_file = inputs.get(tag)
        if source_file is None or source_line <= 0 or page_number <= 0:
            continue
        width = tail[0] if len(tail) >= 1 else 0
        height = tail[1] if len(tail) >= 2 else 0
        depth = tail[2] if len(tail) >= 3 else 0
        grouped[(source_file, source_line)].append(
            SyncTeXPoint(
                source_file=source_file,
                source_line=source_line,
                page_number=page_number,
                x=raw_x / SCALED_POINTS_PER_PDF_POINT,
                y=raw_y / SCALED_POINTS_PER_PDF_POINT,
                kind=kind,
                width=width / SCALED_POINTS_PER_PDF_POINT,
                height=height / SCALED_POINTS_PER_PDF_POINT,
                depth=depth / SCALED_POINTS_PER_PDF_POINT,
            )
        )
        records_indexed += 1
    frozen = {
        key: tuple(values)
        for key, values in sorted(
            grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])
        )
    }
    pages = tuple(sorted({point.page_number for values in frozen.values() for point in values}))
    return SyncTeXIndex(
        source_path=synctex_path,
        inputs=dict(sorted(inputs.items())),
        by_line=frozen,
        pages=pages,
        records_seen=records_seen,
        records_indexed=records_indexed,
    )


def source_lines_for_span(
    source: str,
    start: int,
    end: int,
) -> tuple[int, ...]:
    """Return one-based source lines intersected by ``[start, end)``."""

    if start < 0 or end < start or end > len(source):
        raise ValueError(f"invalid source span [{start}, {end}) for length {len(source)}")
    starts = [0]
    starts.extend(index + 1 for index, character in enumerate(source) if character == "\n")
    if start == end:
        return (bisect.bisect_right(starts, start),)
    first = bisect.bisect_right(starts, start)
    last = bisect.bisect_right(starts, max(start, end - 1))
    return tuple(range(first, last + 1))


def _components(points: Sequence[SyncTeXPoint]) -> list[list[float]]:
    """Build conservative baseline/lane components from metadata points."""

    useful = [point for point in points if point.kind in _LEAF_KINDS]
    if not useful:
        useful = list(points)
    if not useful:
        return []
    by_baseline: list[list[SyncTeXPoint]] = []
    for point in sorted(useful, key=lambda item: (item.y, item.x, item.kind)):
        if not by_baseline or abs(point.y - by_baseline[-1][0].y) > 2.0:
            by_baseline.append([point])
        else:
            by_baseline[-1].append(point)
    output: list[list[float]] = []
    for baseline in by_baseline:
        runs: list[list[SyncTeXPoint]] = []
        for point in sorted(baseline, key=lambda item: item.x):
            if not runs or point.x - runs[-1][-1].x > 36.0:
                runs.append([point])
            else:
                runs[-1].append(point)
        for run in runs:
            left = min(point.x for point in run)
            right = max(point.x + max(0.0, point.width) for point in run)
            baseline_y = sum(point.y for point in run) / len(run)
            above = max((max(0.0, point.height) for point in run), default=0.0)
            below = max((max(0.0, point.depth) for point in run), default=0.0)
            # Leaf records have no height.  A small line box is sufficient for
            # ordering and is intentionally not presented as glyph geometry.
            top = baseline_y - max(above, 4.0)
            bottom = baseline_y + max(below, 2.0)
            if right <= left:
                right = left + 1.0
            output.append([left, top, right, bottom])
    return output


def _union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def alignment_for_probes(
    index: SyncTeXIndex,
    probes: Sequence[Any],
    atom_locators: Mapping[str, Any],
    *,
    line_overrides: Mapping[str, tuple[Path, Sequence[int]]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Translate source probes into the color-alignment row contract.

    ``probes`` and ``atom_locators`` are intentionally duck typed so this
    read-only module does not depend on either the stable pipeline or its
    experimental script module.
    """

    source_cache: dict[Path, str] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    mapped = multi_page = components_total = 0
    overrides = line_overrides or {}
    for probe in probes:
        locator = atom_locators.get(probe.probe_id)
        override = overrides.get(probe.probe_id)
        if override is not None:
            source_file = Path(override[0]).resolve(strict=False)
            lines = tuple(int(value) for value in override[1])
        else:
            source_file = Path(
                locator.source_file if locator is not None else probe.source_file
            ).resolve(strict=False)
        if override is None and locator is not None:
            source = source_cache.get(source_file)
            if source is None:
                source = source_file.read_text(encoding="utf-8", errors="replace")
                source_cache[source_file] = source
            lines = source_lines_for_span(
                source,
                int(locator.source_start),
                int(locator.source_end),
            )
        elif override is None:
            lines = tuple(int(value) for value in probe.source_lines)
        points = index.points_for_lines(source_file, lines)
        by_page: dict[int, list[SyncTeXPoint]] = defaultdict(list)
        for point in points:
            by_page[point.page_number].append(point)
        pages: list[dict[str, Any]] = []
        for page_number, page_points in sorted(by_page.items()):
            components = _components(page_points)
            if not components:
                continue
            components_total += len(components)
            pages.append(
                {
                    "page_number": page_number,
                    "bbox_points": _union(components),
                    "components": components,
                    "characters": len(page_points),
                    "source_lines": list(lines),
                    "locator": "synctex_clean_source_line",
                }
            )
        rows[probe.probe_id] = pages
        mapped += bool(pages)
        multi_page += len(pages) > 1
    total = len(probes)
    return rows, {
        "locator": "synctex_clean_source_line",
        "probes_total": total,
        "probes_mapped": mapped,
        "probes_unmapped": total - mapped,
        "probes_spanning_multiple_pages": multi_page,
        "glyph_components": components_total,
        "coverage": round(mapped / max(1, total), 8),
        "pdf_text_used": False,
    }


__all__ = [
    "SCALED_POINTS_PER_PDF_POINT",
    "SyncTeXIndex",
    "SyncTeXPoint",
    "alignment_for_probes",
    "parse_synctex",
    "source_lines_for_span",
]
