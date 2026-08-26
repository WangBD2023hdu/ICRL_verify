"""Geometry-only reading order for source-localized page fragments.

This module is deliberately independent from the stable source-first pipeline.
It consumes source ordinals and geometry recovered from compiler instrumentation;
it never consumes PDF text.  The central abstraction is a *banded layout graph*:
full-width blocks split a page into vertical bands, and each column band is read
left-to-right (all of the left lane, then all of the right lane).  That makes the
following common layouts explicit instead of relying on a page-wide y-sort::

    full title -> two columns
    two columns -> full spanning block -> two columns
    two columns -> full footer

The result includes immutable diagnostics.  Callers that use ``strict=True``
get a :class:`LayoutConflictError` for an ambiguous or contradictory layout;
callers doing a filtering pass can use the default ``strict=False`` and inspect
``result.diagnostics``.

``SourceFragment`` and ``GlyphComponent`` contain no rendered text.  A caller
may keep Markdown/source text alongside these records, keyed by ``fragment_id``;
the layout graph only decides order and page membership.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


_EPSILON = 1e-7


@dataclass(frozen=True, slots=True)
class Rect:
    """An immutable PDF-coordinate rectangle (x0, y0, x1, y1).

    Coordinates are intentionally named after the geometry API rather than
    after PDF words or glyph text.  The module treats y as increasing downward,
    which is the convention used by the existing page geometry helpers.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("rectangle coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("rectangle must have positive width and height")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def vertical_overlap(self, other: "Rect") -> float:
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass(frozen=True, slots=True)
class SourceFragment:
    """A source-localized, text-free fragment on one rendered page.

    ``source_ordinal`` is assigned by the source instrumentation and is the
    only source-side ordering signal used by this module.  It is a tiebreaker
    for geometry and a consistency check, never a replacement for geometry.
    """

    fragment_id: str
    source_ordinal: int
    bbox: Rect | Sequence[float]
    page_number: int = 1
    kind: str = "text"
    source_group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.fragment_id:
            raise ValueError("fragment_id must not be empty")
        if int(self.source_ordinal) != self.source_ordinal:
            raise ValueError("source_ordinal must be an integer")
        if int(self.page_number) != self.page_number or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.bbox, Rect):
            try:
                values = tuple(float(value) for value in self.bbox)
            except (TypeError, ValueError) as exc:
                raise ValueError("bbox must be Rect or a four-value sequence") from exc
            if len(values) != 4:
                raise ValueError("bbox must have four values")
            object.__setattr__(self, "bbox", Rect(*values))


@dataclass(frozen=True, slots=True)
class GlyphComponent:
    """A source-ordered geometric component used to build fragment boxes.

    This is useful when a compiler probe yields several disconnected glyph
    islands for one source unit.  No character or PDF text is stored here.
    """

    component_id: str
    fragment_id: str
    source_ordinal: int
    bbox: Rect | Sequence[float]
    page_number: int = 1
    component_ordinal: int = 0

    def __post_init__(self) -> None:
        if not self.component_id or not self.fragment_id:
            raise ValueError("component_id and fragment_id must not be empty")
        if int(self.source_ordinal) != self.source_ordinal:
            raise ValueError("source_ordinal must be an integer")
        if int(self.component_ordinal) != self.component_ordinal:
            raise ValueError("component_ordinal must be an integer")
        if int(self.page_number) != self.page_number or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.bbox, Rect):
            values = tuple(float(value) for value in self.bbox)
            if len(values) != 4:
                raise ValueError("bbox must have four values")
            object.__setattr__(self, "bbox", Rect(*values))


@dataclass(frozen=True, slots=True)
class LayoutBand:
    """One vertical band in a page layout graph.

    ``lane`` is ``full``, ``left``, ``right``, or ``columns``.  A ``columns``
    band stores fragments in its deterministic left-then-right order.
    """

    band_id: str
    page_number: int
    top: float
    bottom: float
    lane: str
    fragment_ids: tuple[str, ...]
    full_span: bool = False

    def __post_init__(self) -> None:
        if self.bottom < self.top:
            raise ValueError("band bottom must not precede top")
        if self.lane not in {"full", "left", "right", "columns"}:
            raise ValueError(f"unknown layout lane: {self.lane}")


@dataclass(frozen=True, slots=True)
class LayoutEdge:
    """A precedence edge in the geometry/source layout graph."""

    before: str
    after: str
    relation: str


@dataclass(frozen=True, slots=True)
class LayoutDiagnostic:
    """A stable, serializable layout diagnostic."""

    code: str
    message: str
    fragment_ids: tuple[str, ...] = ()
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class LayoutGraphResult:
    """Result of :func:`build_layout_graph`.

    ``ordered_fragment_ids`` is empty when an error prevents a safe order.  A
    non-strict call may still return a partial order for diagnostics, but its
    ``accepted`` property remains false whenever an error is present.
    """

    page_number: int | None
    layout_kind: str
    column_split: float | None
    ordered_fragment_ids: tuple[str, ...]
    bands: tuple[LayoutBand, ...]
    edges: tuple[LayoutEdge, ...]
    diagnostics: tuple[LayoutDiagnostic, ...] = ()

    @property
    def errors(self) -> tuple[LayoutDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def accepted(self) -> bool:
        return not self.errors and bool(self.ordered_fragment_ids)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostic view without source/PDF text."""

        return {
            "page_number": self.page_number,
            "layout_kind": self.layout_kind,
            "column_split": self.column_split,
            "ordered_fragment_ids": list(self.ordered_fragment_ids),
            "bands": [
                {
                    "band_id": band.band_id,
                    "page_number": band.page_number,
                    "top": band.top,
                    "bottom": band.bottom,
                    "lane": band.lane,
                    "fragment_ids": list(band.fragment_ids),
                    "full_span": band.full_span,
                }
                for band in self.bands
            ],
            "edges": [
                {
                    "before": edge.before,
                    "after": edge.after,
                    "relation": edge.relation,
                }
                for edge in self.edges
            ],
            "diagnostics": [
                {
                    "code": item.code,
                    "message": item.message,
                    "fragment_ids": list(item.fragment_ids),
                    "severity": item.severity,
                }
                for item in self.diagnostics
            ],
        }


class LayoutConflictError(ValueError):
    """Raised by strict mode when geometry/source constraints conflict."""

    def __init__(self, result: LayoutGraphResult):
        self.result = result
        details = "; ".join(
            f"{item.code}: {item.message}" for item in result.errors
        )
        super().__init__(details or "layout constraints conflict")


def fragments_from_glyph_components(
    components: Iterable[GlyphComponent],
    *,
    kind: str = "text",
) -> tuple[SourceFragment, ...]:
    """Aggregate source-ordered glyph geometry into immutable fragments.

    Components for a fragment must belong to one page.  Their union is the
    geometry used by the layout graph; component order is retained only through
    the minimum source ordinal.  This helper is geometry-only and does not read
    or reconstruct PDF text.
    """

    grouped: dict[str, list[GlyphComponent]] = defaultdict(list)
    for component in components:
        grouped[component.fragment_id].append(component)
    output: list[SourceFragment] = []
    grouped_values = sorted(
        grouped.items(),
        key=lambda item: (
            min(component.source_ordinal for component in item[1]),
            item[0],
        ),
    )
    for fragment_id, values in grouped_values:
        pages = {item.page_number for item in values}
        if len(pages) != 1:
            raise ValueError(
                f"fragment {fragment_id} has glyph components on multiple pages"
            )
        rectangle = values[0].bbox
        for item in values[1:]:
            rectangle = rectangle.union(item.bbox)  # type: ignore[union-attr]
        output.append(
            SourceFragment(
                fragment_id=fragment_id,
                source_ordinal=min(item.source_ordinal for item in values),
                bbox=rectangle,
                page_number=values[0].page_number,
                kind=kind,
            )
        )
    return tuple(output)


def _source_key(fragment: SourceFragment) -> tuple[int, str]:
    return (int(fragment.source_ordinal), fragment.fragment_id)


def _geometry_key(
    fragment: SourceFragment,
    *,
    y_tolerance: float,
) -> tuple[int, float, int, str]:
    # Quantizing only the primary y key makes nearly co-baseline probes stable;
    # the raw y and source ordinal remain deterministic tie-breakers.
    bucket = int(round(fragment.bbox.y0 / max(y_tolerance, _EPSILON)))
    # Within one baseline bucket, source order is the deterministic tie-breaker
    # (the compiler can report tiny y jitter for glyph probes from one line).
    return (bucket, int(fragment.source_ordinal), fragment.bbox.y0, fragment.fragment_id)


def _unique_diagnostics(
    diagnostics: Iterable[LayoutDiagnostic],
) -> tuple[LayoutDiagnostic, ...]:
    output: list[LayoutDiagnostic] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for item in diagnostics:
        key = (item.code, item.fragment_ids, item.message)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return tuple(output)


def _infer_two_columns(
    fragments: Sequence[SourceFragment],
    page_width: float,
    *,
    full_span_ratio: float,
    min_column_gap_ratio: float,
) -> tuple[float | None, tuple[LayoutDiagnostic, ...]]:
    """Infer a gutter from center gaps and interval crossings.

    The split is not assumed to be page center.  We test every meaningful gap
    between source fragment centers, minimizing boxes that cross the candidate
    gutter and preferring a wide, balanced gap.  Wide blocks are excluded from
    the crossing score because they are the full-span separators we are trying
    to identify.
    """

    narrow = [
        fragment
        for fragment in fragments
        if fragment.bbox.width < page_width * full_span_ratio
    ]
    if len(narrow) < 4:
        return None, ()
    centers = sorted(
        {
            round(fragment.bbox.center_x, 7)
            for fragment in narrow
        }
    )
    # Two stable lane centers are sufficient (for example a title plus one
    # line in each column).  Requiring four distinct centers incorrectly
    # rejects the common case where several source probes share a baseline.
    if len(centers) < 2:
        return None, ()
    min_gap = page_width * min_column_gap_ratio
    candidates: list[tuple[tuple[float, float, float, float], float]] = []
    for left_center, right_center in zip(centers, centers[1:]):
        gap = right_center - left_center
        if gap + _EPSILON < min_gap:
            continue
        split = (left_center + right_center) / 2.0
        left_count = sum(item.bbox.center_x < split for item in narrow)
        right_count = len(narrow) - left_count
        if left_count < 2 or right_count < 2:
            continue
        crossing = sum(
            item.bbox.x0 < split - _EPSILON and item.bbox.x1 > split + _EPSILON
            for item in narrow
        )
        imbalance = abs(left_count - right_count) / max(1, len(narrow))
        # Crossing is the hard constraint.  Then prefer the widest actual
        # gutter, balanced support, and finally a split closer to the center.
        score = (
            float(crossing),
            -gap / page_width,
            imbalance,
            abs(split - page_width / 2.0) / page_width,
        )
        candidates.append((score, split))
    if not candidates:
        return None, ()
    _, split = min(candidates, key=lambda item: item[0])
    left = [item for item in narrow if item.bbox.center_x < split]
    right = [item for item in narrow if item.bbox.center_x >= split]
    left_edge = max((item.bbox.x1 for item in left), default=split)
    right_edge = min((item.bbox.x0 for item in right), default=split)
    if right_edge < left_edge - page_width * 0.02:
        # Text lines can be long enough to overlap the inferred split without
        # being genuinely full-span.  Keep the split, but expose the ambiguity
        # so a strict caller can filter it instead of silently guessing.
        ids = tuple(item.fragment_id for item in (*left, *right))
        return split, (
            LayoutDiagnostic(
                code="overlapping_column_extents",
                message=(
                    "inferred column extents overlap by more than 2% of page width"
                ),
                fragment_ids=tuple(sorted(ids)),
            ),
        )
    return split, ()


def _sort_lane(
    values: Sequence[SourceFragment],
    *,
    y_tolerance: float,
) -> list[SourceFragment]:
    return sorted(
        values,
        key=lambda item: _geometry_key(item, y_tolerance=y_tolerance),
    )


def _check_source_geometry_consistency(
    values: Sequence[SourceFragment],
    *,
    y_tolerance: float,
) -> list[LayoutDiagnostic]:
    diagnostics: list[LayoutDiagnostic] = []
    geometry = _sort_lane(values, y_tolerance=y_tolerance)
    for previous, current in zip(geometry, geometry[1:]):
        if (
            previous.bbox.y1 - current.bbox.y0 > y_tolerance
            or previous.source_ordinal <= current.source_ordinal
        ):
            continue
        diagnostics.append(
            LayoutDiagnostic(
                code="source_geometry_conflict",
                message=(
                    "source ordinal decreases while geometry advances within one lane"
                ),
                fragment_ids=(previous.fragment_id, current.fragment_id),
            )
        )
    return diagnostics


def _chain_edges(ordered: Sequence[str], relation: str = "reading_order") -> tuple[LayoutEdge, ...]:
    return tuple(
        LayoutEdge(before=before, after=after, relation=relation)
        for before, after in zip(ordered, ordered[1:])
    )


def _empty_result(
    page_number: int | None,
    diagnostics: Sequence[LayoutDiagnostic],
    *,
    layout_kind: str = "invalid",
) -> LayoutGraphResult:
    return LayoutGraphResult(
        page_number=page_number,
        layout_kind=layout_kind,
        column_split=None,
        ordered_fragment_ids=(),
        bands=(),
        edges=(),
        diagnostics=_unique_diagnostics(diagnostics),
    )


def build_layout_graph(
    fragments: Sequence[SourceFragment],
    *,
    page_width: float,
    page_height: float | None = None,
    strict: bool = False,
    full_span_ratio: float = 0.60,
    min_column_gap_ratio: float = 0.08,
    y_tolerance: float = 2.0,
    gutter_tolerance_ratio: float = 0.015,
) -> LayoutGraphResult:
    """Build a deterministic, geometry/source-order page graph.

    Parameters are in the same coordinate units as ``bbox``.  ``page_width``
    is required because unequal column widths cannot be recovered reliably from
    fragment boxes alone without a page boundary.  ``page_height`` is optional;
    it is used only to label an explicit bottom/footer band.

    ``strict=False`` returns diagnostics for filtering.  ``strict=True`` raises
    :class:`LayoutConflictError` whenever a diagnostic has error severity.
    No text extraction or PDF text comparison is performed.
    """

    diagnostics: list[LayoutDiagnostic] = []
    if not math.isfinite(float(page_width)) or page_width <= 0:
        diagnostics.append(
            LayoutDiagnostic("invalid_page_width", "page_width must be finite and positive")
        )
        result = _empty_result(None, diagnostics)
        if strict:
            raise LayoutConflictError(result)
        return result
    if page_height is not None and (
        not math.isfinite(float(page_height)) or page_height <= 0
    ):
        diagnostics.append(
            LayoutDiagnostic("invalid_page_height", "page_height must be finite and positive")
        )
        result = _empty_result(None, diagnostics)
        if strict:
            raise LayoutConflictError(result)
        return result
    if not fragments:
        diagnostics.append(LayoutDiagnostic("empty_page", "no source fragments supplied"))
        result = _empty_result(None, diagnostics)
        if strict:
            raise LayoutConflictError(result)
        return result
    pages = {int(fragment.page_number) for fragment in fragments}
    page_number = min(pages) if len(pages) == 1 else None
    if len(pages) != 1:
        diagnostics.append(
            LayoutDiagnostic(
                "mixed_page_input",
                "all fragments must belong to the same page",
                tuple(sorted(fragment.fragment_id for fragment in fragments)),
            )
        )
        result = _empty_result(page_number, diagnostics)
        if strict:
            raise LayoutConflictError(result)
        return result
    ids = [fragment.fragment_id for fragment in fragments]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        diagnostics.append(
            LayoutDiagnostic(
                "duplicate_fragment_id",
                "fragment identifiers must be unique",
                tuple(duplicates),
            )
        )
    if full_span_ratio <= 0.0 or full_span_ratio >= 1.0:
        diagnostics.append(
            LayoutDiagnostic(
                "invalid_full_span_ratio",
                "full_span_ratio must lie strictly between zero and one",
            )
        )
    if y_tolerance < 0.0:
        diagnostics.append(
            LayoutDiagnostic("invalid_y_tolerance", "y_tolerance must not be negative")
        )
    if diagnostics:
        result = _empty_result(page_number, diagnostics)
        if strict:
            raise LayoutConflictError(result)
        return result

    ordered_input = tuple(sorted(fragments, key=_source_key))
    split, split_diagnostics = _infer_two_columns(
        ordered_input,
        float(page_width),
        full_span_ratio=full_span_ratio,
        min_column_gap_ratio=min_column_gap_ratio,
    )
    diagnostics.extend(split_diagnostics)
    two_columns = split is not None
    layout_kind = "two_column" if two_columns else "single_column"
    labels: dict[str, str] = {}
    fragment_by_id = {fragment.fragment_id: fragment for fragment in fragments}
    ambiguous_ids: list[str] = []
    gutter_tolerance = page_width * max(0.0, gutter_tolerance_ratio)
    for fragment in fragments:
        if not two_columns:
            labels[fragment.fragment_id] = "full"
            continue
        assert split is not None
        crosses_split = (
            fragment.bbox.x0 < split - gutter_tolerance
            and fragment.bbox.x1 > split + gutter_tolerance
        )
        if (
            fragment.bbox.width >= page_width * full_span_ratio
            or crosses_split
            or (
                fragment.bbox.x0 <= page_width * 0.04
                and fragment.bbox.x1 >= page_width * 0.96
            )
        ):
            labels[fragment.fragment_id] = "full"
        elif fragment.bbox.x1 <= split + gutter_tolerance:
            labels[fragment.fragment_id] = "left"
        elif fragment.bbox.x0 >= split - gutter_tolerance:
            labels[fragment.fragment_id] = "right"
        else:
            ambiguous_ids.append(fragment.fragment_id)
    if ambiguous_ids:
        diagnostics.append(
            LayoutDiagnostic(
                "ambiguous_lane",
                "fragment intersects the inferred gutter without a safe lane",
                tuple(sorted(ambiguous_ids)),
            )
        )

    full = [item for item in fragments if labels.get(item.fragment_id) == "full"]
    left = [item for item in fragments if labels.get(item.fragment_id) == "left"]
    right = [item for item in fragments if labels.get(item.fragment_id) == "right"]
    for values in (left, right, full):
        diagnostics.extend(
            _check_source_geometry_consistency(values, y_tolerance=y_tolerance)
        )

    full_sorted = _sort_lane(full, y_tolerance=y_tolerance)
    # In a single-column page every fragment is labelled ``full`` by design;
    # near-baseline run-in headings or source probes can legitimately overlap
    # in y.  The overlap constraint is meaningful only for true spanning
    # separators inside a multi-column page.
    if two_columns:
        for previous, current in zip(full_sorted, full_sorted[1:]):
            if previous.bbox.y1 > current.bbox.y0 + y_tolerance:
                diagnostics.append(
                    LayoutDiagnostic(
                        "overlapping_full_blocks",
                        "full-span blocks overlap vertically",
                        (previous.fragment_id, current.fragment_id),
                    )
                )
    # A column fragment physically occupying the vertical interval of a full
    # spanning block is an unresolvable geometry conflict, not a reason to
    # silently choose a side.
    if two_columns:
        for spanning in full_sorted:
            for column in (*left, *right):
                if spanning.bbox.vertical_overlap(column.bbox) > y_tolerance:
                    diagnostics.append(
                        LayoutDiagnostic(
                            "column_overlaps_full_block",
                            "column fragment overlaps a full-span block vertically",
                            (spanning.fragment_id, column.fragment_id),
                        )
                    )

    bands: list[LayoutBand] = []
    ordered: list[SourceFragment] = []
    if not two_columns:
        values = _sort_lane(fragments, y_tolerance=y_tolerance)
        if values:
            bands.append(
                LayoutBand(
                    band_id=f"page-{page_number:04d}-full-000",
                    page_number=page_number,
                    top=min(item.bbox.y0 for item in values),
                    bottom=max(item.bbox.y1 for item in values),
                    lane="full",
                    fragment_ids=tuple(item.fragment_id for item in values),
                    full_span=True,
                )
            )
            ordered.extend(values)
    else:
        cursor = -math.inf
        band_index = 0
        for spanning in full_sorted:
            before_left = [item for item in left if item.bbox.y0 >= cursor and item.bbox.y0 < spanning.bbox.y0]
            before_right = [item for item in right if item.bbox.y0 >= cursor and item.bbox.y0 < spanning.bbox.y0]
            band_values = _sort_lane(before_left, y_tolerance=y_tolerance) + _sort_lane(
                before_right, y_tolerance=y_tolerance
            )
            if band_values:
                bands.append(
                    LayoutBand(
                        band_id=f"page-{page_number:04d}-columns-{band_index:03d}",
                        page_number=page_number,
                        top=min(item.bbox.y0 for item in band_values),
                        bottom=max(item.bbox.y1 for item in band_values),
                        lane="columns",
                        fragment_ids=tuple(item.fragment_id for item in band_values),
                    )
                )
                ordered.extend(band_values)
                band_index += 1
            bands.append(
                LayoutBand(
                    band_id=f"page-{page_number:04d}-full-{band_index:03d}",
                    page_number=page_number,
                    top=spanning.bbox.y0,
                    bottom=spanning.bbox.y1,
                    lane="full",
                    fragment_ids=(spanning.fragment_id,),
                    full_span=True,
                )
            )
            ordered.append(spanning)
            cursor = max(cursor, spanning.bbox.y1)
            band_index += 1
        trailing_left = [item for item in left if item.bbox.y0 >= cursor]
        trailing_right = [item for item in right if item.bbox.y0 >= cursor]
        trailing = _sort_lane(trailing_left, y_tolerance=y_tolerance) + _sort_lane(
            trailing_right, y_tolerance=y_tolerance
        )
        if trailing:
            bands.append(
                LayoutBand(
                    band_id=f"page-{page_number:04d}-columns-{band_index:03d}",
                    page_number=page_number,
                    top=min(item.bbox.y0 for item in trailing),
                    bottom=max(item.bbox.y1 for item in trailing),
                    lane="columns",
                    fragment_ids=tuple(item.fragment_id for item in trailing),
                )
            )
            ordered.extend(trailing)
    seen_order: set[str] = set()
    for item in ordered:
        if item.fragment_id in seen_order:
            diagnostics.append(
                LayoutDiagnostic(
                    "fragment_assigned_twice",
                    "fragment was assigned to more than one layout band",
                    (item.fragment_id,),
                )
            )
        seen_order.add(item.fragment_id)
    missing = sorted(set(fragment_by_id) - seen_order)
    if missing:
        diagnostics.append(
            LayoutDiagnostic(
                "fragment_unassigned",
                "fragment could not be assigned to a deterministic band",
                tuple(missing),
            )
        )
    ordered_ids = tuple(item.fragment_id for item in ordered)
    result = LayoutGraphResult(
        page_number=page_number,
        layout_kind=layout_kind,
        column_split=split,
        ordered_fragment_ids=ordered_ids,
        bands=tuple(bands),
        edges=_chain_edges(ordered_ids),
        diagnostics=_unique_diagnostics(diagnostics),
    )
    if strict and result.errors:
        raise LayoutConflictError(result)
    return result


def order_page_fragments(
    fragments: Sequence[SourceFragment],
    *,
    page_width: float,
    page_height: float | None = None,
    strict: bool = False,
    **kwargs: object,
) -> LayoutGraphResult:
    """Compatibility alias for callers that think in terms of page ordering."""

    return build_layout_graph(
        fragments,
        page_width=page_width,
        page_height=page_height,
        strict=strict,
        **kwargs,
    )


__all__ = [
    "GlyphComponent",
    "LayoutBand",
    "LayoutConflictError",
    "LayoutDiagnostic",
    "LayoutEdge",
    "LayoutGraphResult",
    "Rect",
    "SourceFragment",
    "build_layout_graph",
    "fragments_from_glyph_components",
    "order_page_fragments",
]
