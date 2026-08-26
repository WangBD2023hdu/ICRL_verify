#!/usr/bin/env python3
"""Build experimental source-first page GT with source spans and layout bands.

This is an isolated v2 experiment.  It deliberately does not modify or call
the stable runner entry point.  Markdown is created from LaTeX source; clean
PDF text is used only by the final exact verifier.  Every safe locator shadow
is compiled and the best invariant shadow is selected independently per page.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
STABLE_SCRIPTS = REPO_ROOT / "scripts"
for value in (str(SOURCE_ROOT), str(STABLE_SCRIPTS)):
    if value not in sys.path:
        sys.path.insert(0, value)

import pdfplumber  # noqa: E402

from arxiv_source_first_v2.contracts import (  # noqa: E402
    EXPERIMENTAL_CONTRACT,
    EXPERIMENTAL_SCHEMA_VERSION,
    assert_stable_files,
)
from arxiv_source_first_v2.layout_graph import (  # noqa: E402
    SourceFragment as LayoutFragment,
    build_layout_graph,
)
from arxiv_source_first_v2.list_ir import (  # noqa: E402
    ListIRSafetyError,
    serialize_source_list,
)
from arxiv_source_first_v2.execution_ir import (  # noqa: E402
    ExecutionIR,
    build_execution_ir,
)
from arxiv_source_first_v2.external_verbatim import (  # noqa: E402
    ExternalVerbatimBlock,
    ExternalVerbatimIR,
    build_external_verbatim_ir,
    render_fenced_code,
)
from arxiv_source_first_v2.source_ir import (  # noqa: E402
    SourceAtom,
    atoms_to_markdown,
    build_source_atoms,
    reconstruct_page_markdown,
)
from arxiv_source_first_v2.safe_macros import (  # noqa: E402
    MacroExpansionError,
    SafeMacroRegistry,
    collect_safe_macros,
    expand_safe_macros,
)
from arxiv_source_first_v2.synctex_ir import (  # noqa: E402
    alignment_for_probes as synctex_alignment_for_probes,
    parse_synctex,
)
from arxiv_source_first_v2.structural_ir import (  # noqa: E402
    EquationTailResolution,
    TheoremStructuralIR,
    build_theorem_ir_from_sources,
    resolve_display_equation_tail,
)
from arxiv_source_first_v2.verifier_projection import (  # noqa: E402
    MATH_VISIBLE_FLOW_PROJECTION_VERSION,
    PROJECTION_VERSION as FOLIO_PROJECTION_VERSION,
    project_bottom_margin_folio,
    project_source_verifier_visible_flow,
)
import build_arxiv_page_markdown_gt as page_gt  # noqa: E402
import build_latex_color_alignment_pilot as color_pilot  # noqa: E402
import build_source_first_color_page_gt as stable  # noqa: E402


PROBE_POLICY_VERSION = (
    "clean_synctex_then_external_verbatim_color_multi_shadow_source_atom_structural_v7"
)
LAYOUT_POLICY_VERSION = "banded_source_geometry_dag_v1"
VERIFIER_CONTRACT_VERSION = stable.VERIFIER_CONTRACT_VERSION
HEARTBEAT_SECONDS = 30.0
MIN_ELIGIBLE_VISIBLE_CHARACTERS = 80
STRUCTURAL_KINDS = frozenset({"display_math", "table"})
ATOM_LOCALIZATION_MODES = frozenset({"source_atom", "source_wrapper_atom"})
EXTERNAL_VERBATIM_LOCALIZATION_MODE = "external_verbatim_line"
HEADING_METADATA_FILENAME_SUFFIX = ".sfv2headings"
WRAPPER_ARGUMENT_SENTINEL = "SFVTWOARGUMENTSENTINEL"
LAYOUT_ONLY_WRAPPER_ENVIRONMENTS = frozenset({"tcolorbox"})
STATIC_LAYOUT_KEYVALS = re.compile(r"[A-Za-z0-9\s,=.!:+*/-]*\Z")
TCOLORBOX_LAYOUT_ONLY_KEYS = frozenset(
    {
        "arc",
        "bottom",
        "boxrule",
        "boxsep",
        "colback",
        "colframe",
        "enlarge bottom by",
        "enlarge left by",
        "enlarge right by",
        "enlarge top by",
        "left",
        "opacityback",
        "opacityframe",
        "outer arc",
        "right",
        "sharp corners",
        "top",
        "width",
    }
)
MAX_BOUNDARY_CARRIERS = 1
MAX_TOKEN_FRONTIER_CUTS = 64
MAX_WHOLE_FRONTIER_CUTS = 128
MAX_PAGE_SOURCE_CANDIDATES = 4096
FRONTIER_POLICY_VERSION = "strict_source_leading_frontier_v1"
VERIFIER_PROJECTION_VERSION = "strict_source_visible_flow_math_braces_fenced_info_v3"


@dataclasses.dataclass(frozen=True)
class AtomLocator:
    probe_id: str
    source_file: Path
    source_start: int
    source_end: int
    atom_ordinal: int


@dataclasses.dataclass(frozen=True)
class VisibleWrapperMacro:
    """One deterministic one-argument visible source wrapper.

    ``body`` is the source body without its outer definition braces and
    ``argument_marker_start`` addresses its unique ``#1`` marker.  Definitions
    with unknown rendering commands, repeated parameters, optional arguments,
    or competing definitions never become instances of this class.
    """

    name: str
    source_file: Path
    definition_start: int
    definition_end: int
    body: str
    argument_marker_start: int


@dataclasses.dataclass(frozen=True)
class VisibleWrapperInvocation:
    """A safe whole-paragraph invocation and its source-only expansion."""

    macro: VisibleWrapperMacro
    call_start: int
    call_end: int
    argument_start: int
    argument_end: int
    argument_source: str
    expanded_source: str
    expanded_argument_start: int
    expanded_argument_end: int


@dataclasses.dataclass(frozen=True)
class LocatedFragment:
    fragment_id: str
    unit_id: str
    paragraph_id: str
    kind: str
    markdown: str
    probe_ids: tuple[str, ...]
    source_file: Path
    source_start_line: int
    source_ordinal: int
    page_number: int
    bbox: tuple[float, float, float, float]
    components: tuple[tuple[float, float, float, float], ...]
    # A finite, immutable source/AUX-derived lattice for structural typography
    # whose class rendering cannot be known from source alone.  PDF text may
    # exact-select one value later, but never creates or edits a candidate.
    structural_markdown_candidates: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass
class ShadowCandidate:
    shadow_id: str
    probes: list[stable.SourceProbe]
    atom_locators: dict[str, AtomLocator]
    unit_atoms: dict[str, tuple[SourceAtom, ...]]
    modes: dict[str, str]
    colored_pdf: Path
    geometry: dict[str, Any]
    logical_invariance: dict[str, Any]
    color_rows: dict[str, list[dict[str, Any]]]
    color_summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--main-tex", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--drop-references", action="store_true")
    parser.add_argument("--drop-figures", action="store_true")
    parser.add_argument("--max-pages", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument(
        "--engine",
        choices=("pdflatex", "xelatex", "latex_dvips_ps2pdf"),
        default="pdflatex",
    )
    parser.add_argument("--latexmk", type=Path, default=color_pilot.LATEXMK)
    parser.add_argument("--pdftoppm", type=Path, default=page_gt.PDFTOPPM)
    parser.add_argument(
        "--min-eligible-visible-characters",
        type=int,
        default=MIN_ELIGIBLE_VISIBLE_CHARACTERS,
    )
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    values = list(rows)
    atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in values
        ),
    )
    return len(values)


def run_synctex_compile(
    *,
    source_root: Path,
    main_tex: Path,
    build_dir: Path,
    log_path: Path,
    timeout_seconds: int,
    engine: str,
) -> tuple[Path, Path]:
    """Compile the untouched experimental clean tree with SyncTeX enabled."""

    flags = {
        "pdflatex": "-pdf",
        "xelatex": "-xelatex",
        "latex_dvips_ps2pdf": "-pdfps",
    }
    if engine not in flags:
        raise ValueError(f"unsupported LaTeX engine: {engine}")
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(color_pilot.LATEXMK),
        "-norc",
        "-g",
        flags[engine],
        "-synctex=1",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        f"-outdir={build_dir}",
        str(main_tex),
    ]
    started = time.monotonic()
    print(
        f"[synctex_compile_start] engine={engine} source={source_root} "
        f"main={main_tex} timeout={timeout_seconds}s",
        flush=True,
    )
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        last_heartbeat = started
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                process.kill()
                process.wait()
                raise TimeoutError(
                    f"source-first-v2-synctex compile timed out after {elapsed:.1f}s"
                )
            if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                print(
                    f"[synctex_compile_progress] elapsed={color_pilot.elapsed_text(elapsed)} "
                    f"log_bytes={log_path.stat().st_size if log_path.exists() else 0}",
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            time.sleep(0.25)
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(
            f"source-first-v2-synctex compile failed rc={process.returncode}\n{tail}"
        )
    pdf_path = build_dir / main_tex.with_suffix(".pdf").name
    candidates = (
        build_dir / (main_tex.stem + ".synctex.gz"),
        build_dir / (main_tex.stem + ".synctex"),
    )
    synctex_path = next((path for path in candidates if path.is_file()), None)
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise FileNotFoundError(f"SyncTeX compile produced no PDF: {pdf_path}")
    if synctex_path is None or synctex_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"SyncTeX compile produced no metadata: {candidates[0]}"
        )
    print(
        f"[synctex_compile_done] pdf_bytes={pdf_path.stat().st_size} "
        f"synctex_bytes={synctex_path.stat().st_size} "
        f"elapsed={color_pilot.elapsed_text(elapsed)}",
        flush=True,
    )
    return pdf_path, synctex_path


def union_bbox(
    boxes: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    if not boxes:
        raise ValueError("cannot union an empty bbox list")
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def order_units_by_execution(
    units: Sequence[stable.SourceUnit],
    execution_ir: ExecutionIR,
) -> tuple[list[stable.SourceUnit], list[dict[str, Any]], dict[str, Any]]:
    """Merge paragraphs, headings, and tables in actual TeX execution order."""

    accepted: list[tuple[tuple[int, int], int, stable.SourceUnit]] = []
    rejected: list[dict[str, Any]] = []
    for input_index, unit in enumerate(units):
        resolution = execution_ir.resolve(unit.source_file, line=unit.start_line)
        if not resolution.is_unique:
            rejected.append(
                {
                    "source_unit_id": unit.unit_id,
                    "reason": f"execution_ir_{resolution.status}",
                    "source_file": str(unit.source_file),
                    "source_line": unit.start_line,
                    "detail": resolution.message,
                }
            )
            continue
        assert resolution.ordinal is not None
        accepted.append((resolution.ordinal.key, input_index, unit))
    accepted.sort(key=lambda value: (value[0], value[1]))
    diagnostics = []
    for value in execution_ir.diagnostics:
        rendered = dataclasses.asdict(value)
        for key in ("source_file", "target_file"):
            if rendered.get(key) is not None:
                rendered[key] = str(rendered[key])
        diagnostics.append(rendered)
    return (
        [value[2] for value in accepted],
        rejected,
        {
            "status": "passed" if accepted else "failed",
            "units_input": len(units),
            "units_ordered": len(accepted),
            "units_rejected": len(rejected),
            "executed_sources": [str(path) for path in execution_ir.executed_sources],
            "diagnostics": diagnostics,
        },
    )


def heading_record_id(block: page_gt.SourceBlock, source_root: Path) -> str:
    relative = block.source_file.resolve().relative_to(source_root.resolve()).as_posix()
    payload = json.dumps(
        {
            "source": relative,
            "line": block.start_line,
            "command": block.heading_command,
            "title": block.heading_source_title,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def heading_metadata_support() -> str:
    """TeX hook that records the class-expanded visible heading label."""

    return r"""
\makeatletter
\newwrite\sfvTwoHeadingOut
\immediate\openout\sfvTwoHeadingOut=\jobname.sfv2headings
\newcommand{\sfvTwoRecordHeading}[2]{%
  \begingroup
  \protected@edef\sfvTwoExpandedLabel{\@seccntformat{#2}}%
  \immediate\write\sfvTwoHeadingOut{SFV2|#1|\meaning\sfvTwoExpandedLabel}%
  \endgroup
}
\AtEndDocument{\immediate\closeout\sfvTwoHeadingOut}
\makeatother
""".strip()


def instrument_heading_metadata_tree(
    source_root: Path,
    blocks: Sequence[page_gt.SourceBlock],
    main_tex: Path,
) -> dict[str, tuple[Path, int, str]]:
    """Insert zero-dimensional compiler-label writes into a disposable tree."""

    root = source_root.resolve()
    records: dict[str, tuple[Path, int, str]] = {}
    by_file: dict[Path, list[page_gt.SourceBlock]] = collections.defaultdict(list)
    for block in blocks:
        if (
            block.kind == "heading"
            and block.heading_command
            and not block.heading_starred
        ):
            by_file[block.source_file.resolve()].append(block)
    for source_file, file_blocks in by_file.items():
        relative = source_file.relative_to(root)
        target = root / relative
        source = target.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines(keepends=True)
        starts: list[int] = []
        cursor = 0
        for line in lines:
            starts.append(cursor)
            cursor += len(line)
        edits: list[tuple[int, str]] = []
        for block in file_blocks:
            start = starts[block.start_line - 1]
            end = starts[block.end_line] if block.end_line < len(starts) else len(source)
            window = source[start:end]
            command = str(block.heading_command)
            match = re.search(
                r"\\" + re.escape(command) + r"\*?\s*(?:\[[^\]]*\]\s*)?\{",
                window,
            )
            if match is None:
                continue
            argument = page_gt.extract_balanced(window, match.end() - 1)
            if argument is None:
                continue
            record_id = heading_record_id(block, root)
            edits.append(
                (
                    start + argument[1],
                    f"\\sfvTwoRecordHeading{{{record_id}}}{{{command}}}",
                )
            )
            records[record_id] = (relative, block.start_line, command)
        for offset, insertion in sorted(edits, reverse=True):
            source = source[:offset] + insertion + source[offset:]
        atomic_write_text(target, source)
    main_path = root / main_tex
    source = main_path.read_text(encoding="utf-8", errors="replace")
    begin = re.search(r"\\begin\s*\{document\}", source)
    if begin is None:
        raise ValueError("main source has no begin{document} for heading metadata")
    source = source[: begin.start()] + heading_metadata_support() + "\n" + source[begin.start() :]
    atomic_write_text(main_path, source)
    return records


def heading_meaning_visible_label(value: str) -> str | None:
    r"""Reduce a compiler-expanded ``\@seccntformat`` meaning to visible text."""

    if "macro:->" in value:
        value = value.split("macro:->", 1)[1]
    value = re.sub(r"\\(?:hskip|kern)\s*[-+]?\d*\.?\d+\s*(?:pt|em|ex|mu)", " ", value)
    value = re.sub(r"\\(?:quad|qquad|enspace|thinspace|space|relax|protect)\b", " ", value)
    value = re.sub(r"\\(?:mbox|hbox)\s*\{([^{}]*)\}", r"\1", value)
    value = page_gt.latex_to_plain(value)
    value = page_gt.normalize_space(value)
    if not value or not re.search(r"[A-Za-z0-9]", value):
        return None
    return value


def compiler_heading_labels(
    *,
    clean_root: Path,
    clean_pdf: Path,
    main_tex: Path,
    blocks: Sequence[page_gt.SourceBlock],
    output_dir: Path,
    timeout_seconds: int,
    engine: str,
) -> tuple[dict[tuple[Path, int, str], str], dict[str, Any]]:
    metadata_root = output_dir / "heading_metadata" / "source"
    metadata_build = output_dir / "heading_metadata" / "build"
    shutil.copytree(clean_root, metadata_root)
    shadow_blocks = [
        dataclasses.replace(
            block,
            source_file=(metadata_root / block.source_file.resolve().relative_to(clean_root.resolve())).resolve(),
        )
        for block in blocks
    ]
    records = instrument_heading_metadata_tree(
        metadata_root, shadow_blocks, main_tex
    )
    metadata_pdf = color_pilot.run_compile(
        source_root=metadata_root,
        main_tex=main_tex,
        build_dir=metadata_build,
        log_path=output_dir / "logs" / "heading_metadata.log",
        label="source-first-v2-heading-metadata",
        timeout_seconds=timeout_seconds,
        engine=engine,
    )
    invariant = compare_pdf_logical_invariance(clean_pdf, metadata_pdf)
    if not invariant["all_pages_equal"]:
        raise RuntimeError("heading metadata hook changed logical page content/order")
    metadata_path = metadata_build / (main_tex.stem + HEADING_METADATA_FILENAME_SUFFIX)
    if not metadata_path.is_file():
        candidate = metadata_root / (main_tex.stem + HEADING_METADATA_FILENAME_SUFFIX)
        metadata_path = candidate if candidate.is_file() else metadata_path
    if not metadata_path.is_file():
        raise FileNotFoundError("compiler produced no heading metadata stream")
    labels: dict[tuple[Path, int, str], str] = {}
    rejected: list[dict[str, str]] = []
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("SFV2|"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3 or parts[1] not in records:
            continue
        label = heading_meaning_visible_label(parts[2])
        relative, source_line, command = records[parts[1]]
        if label is None:
            rejected.append({"record_id": parts[1], "meaning": parts[2]})
            continue
        labels[((clean_root / relative).resolve(), source_line, command)] = label
    return labels, {
        "status": "passed" if labels else "failed",
        "records_requested": len(records),
        "labels_resolved": len(labels),
        "labels_rejected": rejected,
        "logical_invariance": invariant,
        "metadata_file": str(metadata_path),
    }


def _source_fragment_base_offset(
    source_file: Path,
    source_lines: Sequence[int],
    fragment: str,
) -> int | None:
    """Return an exact source offset, never a guessed provenance position."""

    if not fragment or not source_file.is_file() or not source_lines:
        return None
    source = source_file.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines(keepends=True)
    first = min(source_lines)
    last = max(source_lines)
    if first < 1 or last > len(lines):
        return None
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    left = starts[first - 1]
    right = starts[last] if last < len(starts) else len(source)
    window = source[left:right]
    relative = window.find(fragment)
    if relative < 0 or window.find(fragment, relative + 1) >= 0:
        return None
    return left + relative


@dataclasses.dataclass(frozen=True)
class ControlArgumentSpan:
    """A source-proven control-sequence argument, including its delimiters."""

    command: str
    argument_ordinal: int
    delimiter: str
    source_start: int
    source_end: int

    def as_json(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "argument_ordinal": self.argument_ordinal,
            "delimiter": self.delimiter,
            "source_character_span": [self.source_start, self.source_end],
        }


@dataclasses.dataclass(frozen=True)
class SourceSyntacticIndex:
    source_file: Path
    source: str
    active_source: str
    argument_spans: tuple[ControlArgumentSpan, ...]


_CONTROL_SEQUENCE = re.compile(r"\\(?:[A-Za-z@]+|.)")


def _skip_control_argument_space(value: str, cursor: int) -> int | None:
    """Skip TeX space, but never infer an argument across a paragraph break."""

    start = cursor
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    if value[start:cursor].count("\n") >= 2:
        return None
    return cursor


def _control_argument_spans(source: str) -> tuple[ControlArgumentSpan, ...]:
    """Index explicit ``\\command{...}``/``\\command[...]`` source groups.

    This is intentionally a syntactic safety index, not a TeX interpreter.  A
    group is recorded only when its delimiter follows a literal control
    sequence and any preceding argument groups without a paragraph break.
    False positives merely make a color locator metadata-only; they can never
    contribute text to GT.
    """

    active = tex_comment_mask(source)
    spans: list[ControlArgumentSpan] = []
    for match in _CONTROL_SEQUENCE.finditer(active):
        token = match.group(0)
        command = token[1:]
        cursor = match.end()
        if cursor < len(active) and active[cursor] == "*":
            cursor += 1
        for ordinal in range(1, 9):
            next_cursor = _skip_control_argument_space(active, cursor)
            if next_cursor is None or next_cursor >= len(active):
                break
            cursor = next_cursor
            opening = active[cursor]
            if opening not in "[{":
                break
            closing = "]" if opening == "[" else "}"
            end = balanced_delimiter_end(active, cursor, opening, closing)
            if end is None:
                break
            spans.append(
                ControlArgumentSpan(
                    command=command,
                    argument_ordinal=ordinal,
                    delimiter=opening + closing,
                    source_start=cursor,
                    source_end=end,
                )
            )
            cursor = end
    return tuple(spans)


def build_source_syntactic_index(source_file: Path) -> SourceSyntacticIndex | None:
    if not source_file.is_file():
        return None
    source = source_file.read_text(encoding="utf-8", errors="replace")
    return SourceSyntacticIndex(
        source_file=source_file.resolve(),
        source=source,
        active_source=tex_comment_mask(source),
        argument_spans=_control_argument_spans(source),
    )


def source_fragment_syntactic_gate(
    source_file: Path,
    source_lines: Sequence[int],
    fragment: str,
    *,
    require_payload_boundary: bool,
    index: SourceSyntacticIndex | None = None,
) -> dict[str, Any]:
    """Prove that a parsed fragment is an executable source payload boundary.

    The result is derived exclusively from the original project source.  It
    rejects fragments cut from command/environment arguments and, for
    executable color shadows, fragments that do not own both physical line
    boundaries used by the legacy instrumenter.
    """

    source_index = index or build_source_syntactic_index(source_file)
    base = _source_fragment_base_offset(source_file, source_lines, fragment)
    provenance: dict[str, Any] = {
        "source_file": str(source_file),
        "source_lines": [min(source_lines), max(source_lines)] if source_lines else [],
        "raw_source_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
        "pdf_text_used": False,
    }
    if source_index is None or base is None:
        return {
            **provenance,
            "status": "rejected",
            "reason": "exact_source_span_unavailable_or_ambiguous",
        }
    end = base + len(fragment)
    provenance["source_character_span"] = [base, end]
    containing = [
        span
        for span in source_index.argument_spans
        if span.source_start <= base < span.source_end
    ]
    if containing:
        span = min(containing, key=lambda value: value.source_end - value.source_start)
        if span.command == "begin" and span.delimiter == "[]":
            reason = "environment_optional_argument_fragment"
        else:
            reason = "control_sequence_argument_fragment"
        return {
            **provenance,
            "status": "rejected",
            "reason": reason,
            "argument": span.as_json(),
            "fragment_end_within_argument": end <= span.source_end,
        }
    if not require_payload_boundary:
        return {**provenance, "status": "passed", "reason": None}

    active = source_index.active_source
    line_start = active.rfind("\n", 0, base) + 1
    line_end = active.find("\n", end)
    if line_end < 0:
        line_end = len(active)
    prefix = active[line_start:base]
    suffix = active[end:line_end]
    prefix_is_payload_boundary = re.fullmatch(
        r"\s*(?:\\(?:noindent|indent)\b\s*)?", prefix
    ) is not None
    suffix_is_payload_boundary = re.fullmatch(
        r"\s*(?:\\par\b\s*)?", suffix
    ) is not None
    if not prefix_is_payload_boundary or not suffix_is_payload_boundary:
        return {
            **provenance,
            "status": "rejected",
            "reason": "source_fragment_not_at_physical_payload_boundary",
            "line_prefix_nonspace_characters": len(re.sub(r"\s", "", prefix)),
            "line_suffix_nonspace_characters": len(re.sub(r"\s", "", suffix)),
            "line_prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "line_suffix_sha256": hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        }
    return {**provenance, "status": "passed", "reason": None}


def _registered_macro_calls(
    source: str,
    names: Iterable[str],
) -> list[str]:
    active = tex_comment_mask(source)
    return sorted(
        name
        for name in set(names)
        if re.search(r"\\" + re.escape(name) + r"\b", active)
    )


def _normalize_safe_macro_visible_styles(value: str) -> str:
    """Erase source-only font wrappers unsupported by the stable serializer.

    ``safe_macros`` has already proved these commands to be one-argument
    visible wrappers.  We erase them only outside inline math so formula LaTeX
    remains source-faithful.  Bold/italic/code wrappers supported by the
    Markdown serializer are intentionally retained.
    """

    output: list[str] = []
    text_start = 0
    cursor = 0
    in_dollar_math = False
    in_paren_math = False

    def normalized_text(fragment: str) -> str:
        result = fragment
        for command in ("textmd", "textsf", "textsl", "textup"):
            result = stable.replace_balanced_command(
                result,
                command,
                argument_count=1,
                visible_argument=0,
            )
        return result

    while cursor < len(value):
        if not in_dollar_math and not in_paren_math and value.startswith(r"\(", cursor):
            output.append(normalized_text(value[text_start:cursor]))
            math_end = value.find(r"\)", cursor + 2)
            if math_end < 0:
                # Leave the malformed remainder to the strict serializer.
                output.append(value[cursor:])
                return "".join(output)
            output.append(value[cursor : math_end + 2])
            cursor = math_end + 2
            text_start = cursor
            continue
        if value[cursor] == "$" and (cursor == 0 or value[cursor - 1] != "\\"):
            if not in_dollar_math:
                output.append(normalized_text(value[text_start:cursor]))
                text_start = cursor
                in_dollar_math = True
            else:
                output.append(value[text_start : cursor + 1])
                text_start = cursor + 1
                in_dollar_math = False
            cursor += 1
            continue
        cursor += 1
    if in_dollar_math:
        output.append(value[text_start:])
    else:
        output.append(normalized_text(value[text_start:]))
    return "".join(output)


def _serialize_heading_title_from_source(
    block: page_gt.SourceBlock,
    expanded_title: str,
    references: dict[str, stable.AuxReference] | None,
) -> str:
    """Use the ordinary strict source serializer for a heading title.

    In particular this reuses deterministic command handling (including
    ``\\href{target}{visible}``), AUX reference resolution, inline math, and
    opaque-command rejection.  PDF text is not an input.
    """

    paragraph = page_gt.SourceParagraph(
        paragraph_id=f"heading-title:{block.block_id}",
        kind="paragraph",
        source_file=block.source_file,
        source_lines=list(range(block.start_line, block.end_line + 1)),
        raw_latex=_normalize_safe_macro_visible_styles(expanded_title),
    )
    return stable.source_paragraph_to_markdown(paragraph, references)


def _exact_heading_source_title(block: page_gt.SourceBlock) -> str | None:
    """Recover the exact title argument retained by the source block parser.

    Real parsed blocks keep the untouched title argument in ``raw_latex`` and
    a legacy, lossy plain-text approximation in ``heading_source_title``.  A
    few direct unit fixtures use a complete ``\\section{...}`` command, so the
    latter shape is unwrapped without consulting PDF text.
    """

    raw = str(block.raw_latex or "").strip()
    command = str(block.heading_command or "")
    if not raw:
        return block.heading_source_title
    match = re.match(
        r"^\\" + re.escape(command) + r"\*?\s*(?:\[[^\]]*\]\s*)?",
        raw,
    )
    if match is None:
        return raw
    cursor = match.end()
    argument = page_gt.extract_balanced(raw, cursor)
    if argument is None or raw[argument[1] :].strip():
        return None
    return argument[0]


def prepare_heading_blocks_for_safe_admission(
    blocks: Sequence[page_gt.SourceBlock],
    safe_macros: SafeMacroRegistry,
    *,
    references: dict[str, stable.AuxReference] | None,
) -> tuple[
    list[page_gt.SourceBlock],
    dict[str, str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Expand and strictly serialize every source heading before admission."""

    admitted: list[page_gt.SourceBlock] = []
    serialized_titles: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    headings_total = 0
    headings_changed = 0
    headings_successful = 0
    for block in blocks:
        if block.kind != "heading":
            admitted.append(block)
            continue
        headings_total += 1
        original_title = _exact_heading_source_title(block)
        if not block.heading_command or not original_title:
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "source_file": str(block.source_file),
                    "source_lines": [block.start_line, block.end_line],
                    "reason": "heading source title unavailable",
                    "admission_stage": "safe_macro_pre_admission",
                }
            )
            continue
        invoked_accepted = _registered_macro_calls(
            original_title, safe_macros.accepted_names
        )
        base_offset = _source_fragment_base_offset(
            block.source_file,
            range(block.start_line, block.end_line + 1),
            original_title,
        )
        if invoked_accepted and base_offset is None:
            expanded = original_title
            events = []
            expansion_rejections = [
                {
                    "reason": "source fragment offset is ambiguous",
                    "macros": invoked_accepted,
                }
            ]
        else:
            expanded, events, expansion_rejections = expand_registered_macro_calls(
                original_title,
                safe_macros,
                source_file=block.source_file,
                source_base_offset=base_offset or 0,
            )
        unresolved_accepted = _registered_macro_calls(
            expanded, safe_macros.accepted_names
        )
        invoked_rejected = _registered_macro_calls(
            original_title, safe_macros.rejected_names
        )
        failure_reasons: list[str] = []
        if expansion_rejections:
            failure_reasons.append("safe macro expansion rejected")
        if unresolved_accepted:
            failure_reasons.append("safe macro expansion incomplete")
        if invoked_rejected:
            failure_reasons.append("rejected project macro invoked")
        title_markdown: str | None = None
        if not failure_reasons:
            try:
                title_markdown = _serialize_heading_title_from_source(
                    block, expanded, references
                )
            except Exception as error:  # noqa: BLE001 - admission is fail closed
                failure_reasons.append(
                    f"{type(error).__name__}: {error}"
                )
        if failure_reasons or title_markdown is None:
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "source_file": str(block.source_file),
                    "source_lines": [block.start_line, block.end_line],
                    "reason": "; ".join(failure_reasons),
                    "admission_stage": "safe_macro_pre_admission",
                    "unresolved_accepted_macros": unresolved_accepted,
                    "invoked_rejected_macros": invoked_rejected,
                    "macro_expansion_rejections": expansion_rejections,
                    "macro_provenance": events,
                }
            )
            continue
        changed = expanded != original_title
        headings_changed += int(changed)
        headings_successful += 1
        admitted.append(dataclasses.replace(block, heading_source_title=expanded))
        serialized_titles[block.block_id] = title_markdown
        if changed:
            provenance.append(
                {
                    "source_block_id": block.block_id,
                    "source_file": str(block.source_file),
                    "source_lines": [block.start_line, block.end_line],
                    "original_title_sha256": hashlib.sha256(
                        original_title.encode("utf-8")
                    ).hexdigest(),
                    "expanded_title_sha256": hashlib.sha256(
                        expanded.encode("utf-8")
                    ).hexdigest(),
                    "macros": sorted(
                        {
                            str(event.get("macro_name"))
                            for event in events
                            if event.get("macro_name")
                        }
                    ),
                    "macro_provenance": events,
                    "original_raw_latex_preserved": True,
                    "original_source_lines_preserved": True,
                }
            )
    report = {
        "policy": "safe_macro_pre_admission_v1",
        "total": headings_total,
        "changed": headings_changed,
        "successful": headings_successful,
        "rejected": len(rejected),
        "provenance": provenance,
        "rejections": rejected,
        "original_source_provenance_preserved": True,
        "pdf_text_used": False,
    }
    return admitted, serialized_titles, rejected, report


def apply_compiler_heading_labels(
    units: Sequence[stable.SourceUnit],
    blocks: Sequence[page_gt.SourceBlock],
    labels: Mapping[tuple[Path, int, str], str],
    serialized_titles: Mapping[str, str],
) -> tuple[list[stable.SourceUnit], dict[str, Any]]:
    block_by_key = {
        (block.source_file.resolve(), block.start_line, str(block.heading_command)): block
        for block in blocks
        if block.kind == "heading" and block.heading_command
    }
    output: list[stable.SourceUnit] = []
    rejected: list[dict[str, Any]] = []
    changed = 0
    for unit in units:
        if unit.kind != "heading" or not unit.source_command:
            output.append(unit)
            continue
        key = (unit.source_file.resolve(), unit.start_line, unit.source_command)
        label = labels.get(key)
        block = block_by_key.get(key)
        if block is None or block.block_id not in serialized_titles:
            rejected.append(
                {
                    "source_block_id": unit.paragraph_id,
                    "source_file": str(unit.source_file),
                    "source_lines": [unit.start_line, unit.end_line],
                    "reason": "strict source heading title unavailable",
                    "admission_stage": "heading_markdown_serialization",
                }
            )
            continue
        title = serialized_titles[block.block_id]
        level = int(block.heading_level or 2)
        if (
            label is not None
            and unit.source_command not in {"paragraph", "subparagraph"}
            and not block.heading_starred
        ):
            prefix = label + " "
        elif block.heading_starred:
            prefix = ""
        else:
            legacy_title_visible = page_gt.latex_to_plain(
                stable.tex_text_punctuation_to_unicode(
                    str(block.heading_source_title or "")
                )
            )
            legacy_title = stable.inline_markup.escape_markdown_text(
                legacy_title_visible
            )
            marker = "#" * level + " "
            if not unit.markdown.startswith(marker):
                rejected.append(
                    {
                        "source_block_id": block.block_id,
                        "reason": "existing heading level is inconsistent",
                        "admission_stage": "heading_markdown_serialization",
                    }
                )
                continue
            body = unit.markdown[len(marker) :]
            if not legacy_title or not body.endswith(legacy_title):
                rejected.append(
                    {
                        "source_block_id": block.block_id,
                        "reason": "existing heading prefix is ambiguous",
                        "admission_stage": "heading_markdown_serialization",
                    }
                )
                continue
            prefix = body[: -len(legacy_title)]
        markdown = "#" * level + " " + prefix + title
        changed += int(markdown != unit.markdown)
        output.append(
            dataclasses.replace(
                unit,
                markdown=markdown,
            )
        )
    return output, {
        "total": sum(unit.kind == "heading" for unit in units),
        "changed": changed,
        "successful": len(output),
        "rejected": len(rejected),
        "rejections": rejected,
        "pdf_text_used": False,
    }


def recover_compiler_labeled_headings(
    units: Sequence[stable.SourceUnit],
    blocks: Sequence[page_gt.SourceBlock],
    labels: Mapping[tuple[Path, int, str], str],
    serialized_titles: Mapping[str, str],
) -> tuple[list[stable.SourceUnit], dict[str, Any]]:
    """Construct headings lost by AUX title matching from compiler labels."""

    output = list(units)
    existing = {
        (unit.source_file.resolve(), unit.start_line, str(unit.source_command))
        for unit in units
        if unit.kind == "heading" and unit.source_command
    }
    recovered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for block in blocks:
        if (
            block.kind != "heading"
            or not block.heading_command
            or not block.heading_source_title
            or block.heading_starred
            or block.heading_command in {"paragraph", "subparagraph"}
        ):
            continue
        key = (
            block.source_file.resolve(),
            block.start_line,
            str(block.heading_command),
        )
        if key in existing:
            continue
        label = labels.get(key)
        if not label:
            continue
        title = serialized_titles.get(block.block_id)
        if title is None:
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "reason": "strict source heading title unavailable",
                }
            )
            continue
        level = int(block.heading_level or 2)
        index = len(output)
        unit = stable.SourceUnit(
            unit_id=f"src-heading-recovered-{index + 1:07d}",
            kind="heading",
            paragraph_id=block.block_id,
            source_file=block.source_file,
            source_lines=tuple(range(block.start_line, block.end_line + 1)),
            raw_latex=block.raw_latex,
            markdown="#" * level + " " + label + " " + title,
            rgb=color_pilot.deterministic_rgb(4_000_000 + index),
            source_command=block.heading_command,
        )
        output.append(unit)
        existing.add(key)
        recovered.append(
            {
                "source_block_id": block.block_id,
                "label": label,
                "title": title,
            }
        )
    return output, {
        "recovered": len(recovered),
        "recovered_headings": recovered,
        "rejected": rejected,
        "pdf_text_used": False,
    }


_VISIBLE_WRAPPER_DECLARATION = re.compile(
    r"\\(?P<declaration>newcommand|renewcommand|providecommand|DeclareRobustCommand)"
    r"\*?\s*(?:\{\s*\\(?P<braced>[A-Za-z@]+)\s*\}|"
    r"\\(?P<direct>[A-Za-z@]+))"
)
_PRIMITIVE_WRAPPER_DECLARATION = re.compile(
    r"\\(?:def|gdef|edef|xdef)\s*\\(?P<name>[A-Za-z@]+)"
)


def tex_comment_mask(value: str) -> str:
    """Blank comments without changing source character offsets."""

    output = list(value)
    cursor = 0
    while cursor < len(value):
        if value[cursor] != "%":
            cursor += 1
            continue
        slashes = 0
        left = cursor - 1
        while left >= 0 and value[left] == "\\":
            slashes += 1
            left -= 1
        if slashes % 2:
            cursor += 1
            continue
        end = value.find("\n", cursor)
        end = len(value) if end < 0 else end
        for index in range(cursor, end):
            output[index] = " "
        cursor = end
    return "".join(output)


def remove_tex_comments_for_wrapper(value: str) -> str:
    """Apply TeX's comment line-continuation rule to a macro body."""

    output: list[str] = []
    cursor = 0
    while cursor < len(value):
        if value[cursor] != "%":
            output.append(value[cursor])
            cursor += 1
            continue
        slashes = 0
        left = cursor - 1
        while left >= 0 and value[left] == "\\":
            slashes += 1
            left -= 1
        if slashes % 2:
            output.append(value[cursor])
            cursor += 1
            continue
        newline = value.find("\n", cursor)
        if newline < 0:
            break
        cursor = newline + 1
        while cursor < len(value) and value[cursor] in " \t":
            cursor += 1
    return "".join(output)


def unwrap_layout_only_visible_wrapper(body: str) -> tuple[str | None, str | None]:
    """Remove one allow-listed environment which contributes no visible text."""

    stripped = body.strip()
    begin = re.match(r"\\begin\s*\{(?P<environment>[A-Za-z*]+)\}", stripped)
    if begin is None:
        return body, None
    environment = str(begin.group("environment"))
    if environment not in LAYOUT_ONLY_WRAPPER_ENVIRONMENTS:
        return None, f"visible_or_unknown_wrapper_environment={environment}"
    cursor = begin.end()
    while cursor < len(stripped) and stripped[cursor].isspace():
        cursor += 1
    if cursor < len(stripped) and stripped[cursor] == "[":
        keyvals_end = balanced_delimiter_end(stripped, cursor, "[", "]")
        if keyvals_end is None:
            return None, "layout_wrapper_keyvals_unbalanced"
        keyvals = stripped[cursor + 1 : keyvals_end - 1]
        if STATIC_LAYOUT_KEYVALS.fullmatch(keyvals) is None:
            return None, "layout_wrapper_keyvals_are_dynamic"
        keys = []
        for item in keyvals.split(","):
            item = item.strip()
            if not item:
                continue
            key = item.split("=", 1)[0].strip().casefold()
            keys.append(key)
        if environment == "tcolorbox" and any(
            key not in TCOLORBOX_LAYOUT_ONLY_KEYS for key in keys
        ):
            return None, "tcolorbox_key_may_contribute_visible_content"
        cursor = keyvals_end
    while cursor < len(stripped) and stripped[cursor].isspace():
        cursor += 1
    ending = re.search(
        r"\\end\s*\{" + re.escape(environment) + r"\}\s*\Z",
        stripped,
    )
    if ending is None or ending.start() < cursor:
        return None, "layout_wrapper_end_is_missing_or_not_outermost"
    content = stripped[cursor : ending.start()]
    if re.search(r"\\(?:begin|end)\s*\{", content):
        return None, "nested_wrapper_environment_is_unsupported"
    return content, None


def unique_wrapper_parameter(body: str) -> tuple[int | None, str | None]:
    """Return the unique literal ``#1`` position or a fail-closed reason."""

    positions: list[int] = []
    cursor = 0
    while cursor < len(body):
        if body[cursor] == "\\" and cursor + 1 < len(body):
            cursor += 2
            continue
        if body[cursor] != "#":
            cursor += 1
            continue
        if body.startswith("#1", cursor):
            positions.append(cursor)
            cursor += 2
            continue
        return None, "unsupported_parameter_token"
    if len(positions) != 1:
        return None, f"parameter_occurrences={len(positions)}"
    return positions[0], None


def validate_visible_wrapper_body(
    body: str,
    argument_marker_start: int,
) -> str | None:
    """Validate rendering and source-span isolation of the argument marker."""

    expanded = (
        body[:argument_marker_start]
        + WRAPPER_ARGUMENT_SENTINEL
        + body[argument_marker_start + 2 :]
    )
    marker_start = argument_marker_start
    marker_end = marker_start + len(WRAPPER_ARGUMENT_SENTINEL)
    atoms = build_source_atoms(expanded)
    if any(atom.kind == "opaque" for atom in atoms):
        return "body_contains_unknown_or_compiler_dependent_macro"
    touching = [
        atom
        for atom in atoms
        if atom.source_start < marker_end and atom.source_end > marker_start
    ]
    if not touching:
        return "argument_is_not_visible"
    if any(
        atom.source_start < marker_start or atom.source_end > marker_end
        for atom in touching
    ):
        return "argument_not_isolated_from_static_wrapper_text"
    if any(atom.kind not in {"text", "whitespace"} for atom in touching):
        return "argument_is_not_visible_prose"
    rendered = atoms_to_markdown(atoms)
    if rendered.count(WRAPPER_ARGUMENT_SENTINEL) != 1:
        return "argument_rendering_is_not_unique"
    return None


def collect_safe_visible_wrapper_macros(
    source_files: Iterable[Path],
) -> tuple[dict[str, VisibleWrapperMacro], dict[str, Any]]:
    """Collect deterministic one-argument wrappers from executed sources.

    FLS/execution filtering is performed by the caller through ``source_files``.
    Any competing or non-``newcommand`` declaration for a name blocks that
    name, even if one otherwise-safe declaration was also observed.
    """

    occurrences: dict[str, list[tuple[VisibleWrapperMacro | None, dict[str, Any]]]] = (
        collections.defaultdict(list)
    )
    files_scanned = 0
    for source_file in sorted({Path(path).resolve() for path in source_files}):
        if not source_file.is_file():
            continue
        try:
            source = source_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files_scanned += 1
        active = tex_comment_mask(source)
        for match in _VISIBLE_WRAPPER_DECLARATION.finditer(active):
            name = str(match.group("braced") or match.group("direct"))
            detail: dict[str, Any] = {
                "name": name,
                "source_file": str(source_file),
                "definition_start": match.start(),
                "declaration": match.group("declaration"),
            }
            declaration = str(match.group("declaration"))
            cursor = match.end()
            while cursor < len(active) and active[cursor].isspace():
                cursor += 1
            if declaration != "newcommand":
                detail["reason"] = f"unsupported_declaration={declaration}"
                occurrences[name].append((None, detail))
                continue
            if cursor >= len(active) or active[cursor] != "[":
                detail["reason"] = "arity_is_not_explicitly_one"
                occurrences[name].append((None, detail))
                continue
            arity_end = balanced_delimiter_end(active, cursor, "[", "]")
            if arity_end is None or active[cursor + 1 : arity_end - 1].strip() != "1":
                detail["reason"] = "arity_is_not_one"
                occurrences[name].append((None, detail))
                continue
            cursor = arity_end
            while cursor < len(active) and active[cursor].isspace():
                cursor += 1
            if cursor < len(active) and active[cursor] == "[":
                detail["reason"] = "optional_argument_default_is_unsupported"
                occurrences[name].append((None, detail))
                continue
            if cursor >= len(active) or active[cursor] != "{":
                detail["reason"] = "definition_body_is_missing"
                occurrences[name].append((None, detail))
                continue
            body_end = balanced_delimiter_end(active, cursor, "{", "}")
            if body_end is None:
                detail["reason"] = "definition_body_is_unbalanced"
                occurrences[name].append((None, detail))
                continue
            source_body = remove_tex_comments_for_wrapper(
                source[cursor + 1 : body_end - 1]
            )
            body, reason = unwrap_layout_only_visible_wrapper(source_body)
            if reason is not None:
                detail["reason"] = reason
                occurrences[name].append((None, detail))
                continue
            assert body is not None
            marker_start, reason = unique_wrapper_parameter(body)
            if reason is None:
                assert marker_start is not None
                reason = validate_visible_wrapper_body(body, marker_start)
            if reason is not None:
                detail["reason"] = reason
                occurrences[name].append((None, detail))
                continue
            assert marker_start is not None
            macro = VisibleWrapperMacro(
                name=name,
                source_file=source_file,
                definition_start=match.start(),
                definition_end=body_end,
                body=body,
                argument_marker_start=marker_start,
            )
            detail["reason"] = None
            occurrences[name].append((macro, detail))
        for match in _PRIMITIVE_WRAPPER_DECLARATION.finditer(active):
            name = str(match.group("name"))
            occurrences[name].append(
                (
                    None,
                    {
                        "name": name,
                        "source_file": str(source_file),
                        "definition_start": match.start(),
                        "declaration": "primitive_def",
                        "reason": "primitive_definition_is_unsupported",
                    },
                )
            )

    safe: dict[str, VisibleWrapperMacro] = {}
    rejected: list[dict[str, Any]] = []
    for name, values in sorted(occurrences.items()):
        candidates = [value for value, _detail in values if value is not None]
        if len(values) == 1 and len(candidates) == 1:
            safe[name] = candidates[0]
            continue
        if len(values) > 1:
            rejected.append(
                {
                    "name": name,
                    "reason": "multiple_or_competing_definitions",
                    "definitions": [detail for _value, detail in values],
                }
            )
        else:
            rejected.append(values[0][1])
    return safe, {
        "files_scanned": files_scanned,
        "definitions_seen": sum(len(values) for values in occurrences.values()),
        "safe_macros": sorted(safe),
        "safe_macro_count": len(safe),
        "blocked_macros": sorted(set(occurrences) - set(safe)),
        "rejections": rejected,
    }


def parse_visible_wrapper_invocation(
    raw_latex: str,
    macros: Mapping[str, VisibleWrapperMacro],
) -> tuple[VisibleWrapperInvocation | None, str | None]:
    """Parse one whole-paragraph safe wrapper invocation."""

    if not macros:
        return None, "no_safe_visible_wrapper_definition"
    active = tex_comment_mask(raw_latex)
    pattern = re.compile(
        r"\\(?P<name>" + "|".join(re.escape(name) for name in sorted(macros)) + r")\b"
    )
    matches = list(pattern.finditer(active))
    if len(matches) != 1:
        return None, f"safe_wrapper_invocations={len(matches)}"
    match = matches[0]
    before = active[: match.start()]
    if re.fullmatch(r"\s*(?:\\(?:noindent|leavevmode)\b\s*)*", before) is None:
        return None, "visible_content_precedes_wrapper_call"
    cursor = match.end()
    while cursor < len(active) and active[cursor].isspace():
        cursor += 1
    if cursor >= len(active) or active[cursor] != "{":
        return None, "wrapper_argument_is_missing"
    argument_group_end = balanced_delimiter_end(active, cursor, "{", "}")
    if argument_group_end is None:
        return None, "wrapper_argument_is_unbalanced"
    after = active[argument_group_end:]
    if re.fullmatch(r"\s*(?:\\par\b\s*)?", after) is None:
        return None, "visible_content_follows_wrapper_call"
    argument_start = cursor + 1
    argument_end = argument_group_end - 1
    argument = raw_latex[argument_start:argument_end]
    macro = macros[str(match.group("name"))]
    marker = macro.argument_marker_start
    expanded = macro.body[:marker] + argument + macro.body[marker + 2 :]
    expanded_argument_start = marker
    expanded_argument_end = marker + len(argument)
    atoms = build_source_atoms(expanded)
    if any(atom.kind == "opaque" for atom in atoms):
        return None, "expanded_wrapper_contains_unknown_or_compiler_dependent_macro"
    argument_atoms = [
        atom
        for atom in atoms
        if atom.source_start < expanded_argument_end
        and atom.source_end > expanded_argument_start
    ]
    if not argument_atoms or not any(
        atom.kind not in {"whitespace", "opaque"} and atom.visible_text.strip()
        for atom in argument_atoms
    ):
        return None, "wrapper_argument_has_no_visible_atoms"
    if any(
        atom.source_start < expanded_argument_start
        or atom.source_end > expanded_argument_end
        for atom in argument_atoms
    ):
        return None, "wrapper_argument_atoms_overlap_static_text"
    return (
        VisibleWrapperInvocation(
            macro=macro,
            call_start=match.start(),
            call_end=argument_group_end,
            argument_start=argument_start,
            argument_end=argument_end,
            argument_source=argument,
            expanded_source=expanded,
            expanded_argument_start=expanded_argument_start,
            expanded_argument_end=expanded_argument_end,
        ),
        None,
    )


def expand_registered_macro_calls(
    source: str,
    registry: SafeMacroRegistry,
    *,
    source_file: Path,
    source_base_offset: int = 0,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand admitted project calls while leaving unrelated TeX to its parser.

    Each known invocation is independently checked by ``safe_macros``.  This
    permits ordinary inline math/control words elsewhere in a paragraph to be
    handled by the established source serializer without weakening the macro
    registry's definition policy.
    """

    definitions = registry.by_name
    if not definitions:
        return source, [], []
    active = tex_comment_mask(source)
    output: list[str] = []
    events: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    cursor = 0
    search = 0
    pattern = re.compile(r"\\([A-Za-z@]+)")
    while match := pattern.search(active, search):
        name = match.group(1)
        definition = definitions.get(name)
        if definition is None:
            search = match.end()
            continue
        command_end = match.end()
        invocation_end = command_end
        if definition.arity == 1:
            argument_cursor = command_end
            while argument_cursor < len(source) and source[argument_cursor].isspace():
                argument_cursor += 1
            argument = page_gt.extract_balanced(source, argument_cursor)
            if argument is None:
                rejections.append(
                    {
                        "macro_name": name,
                        "source_span": [
                            source_base_offset + match.start(),
                            source_base_offset + command_end,
                        ],
                        "reason": "mandatory_braced_argument_missing",
                    }
                )
                search = command_end
                continue
            invocation_end = argument[1]
        # Include enough right context for exact xspace/control-word delimiter
        # behavior, and replace that context together with the invocation.
        context_end = invocation_end
        if definition.arity == 0:
            while context_end < len(source) and source[context_end].isspace():
                context_end += 1
        if context_end < len(source) and source[context_end] != "\\":
            context_end += 1
        if context_end == invocation_end and context_end < len(source):
            context_end += 1
        window = source[match.start() : context_end]
        try:
            result = expand_safe_macros(
                window,
                registry,
                source_file=source_file,
                source_base_offset=source_base_offset + match.start(),
            )
        except MacroExpansionError as error:
            rejections.append(
                {
                    "macro_name": name,
                    "source_span": [
                        source_base_offset + match.start(),
                        source_base_offset + invocation_end,
                    ],
                    "reason": str(error),
                }
            )
            search = invocation_end
            continue
        output.append(source[cursor : match.start()])
        output.append(result.text)
        events.extend(item.as_dict() for item in result.provenance)
        cursor = context_end
        search = context_end
    output.append(source[cursor:])
    return "".join(output), events, rejections


def _render_list_inline_source(
    source: str,
    references: Mapping[str, stable.AuxReference] | None,
) -> str:
    """Render one list body through the existing strict source serializer."""

    value = stable.normalize_source_deterministic_commands(source)
    value = stable.resolve_source_references(value, references or {})
    value = stable.tex_text_punctuation_to_unicode(value)
    plan = stable.inline_markup.parse_inline_plan(value.strip())
    if int(plan.feature_counts.get("opaque", 0)):
        raise ValueError("list body contains compiler-dependent or unknown macros")
    rendered = stable.inline_markup.render_inline_source(plan).strip()
    if not rendered:
        raise ValueError("list body rendered empty")
    return rendered


def _source_range_for_list_item(
    source_paragraphs: Sequence[page_gt.SourceParagraph],
    *,
    all_list_paragraphs: Sequence[page_gt.SourceParagraph],
) -> tuple[tuple[int, ...], str] | None:
    """Return original source lines/text when an item has no line overlap.

    A parent item with a continuation after a nested list would span the child
    item's lines.  One SourceUnit cannot safely represent that overlapping
    source range, so this helper rejects it; the child remains independently
    serializable and no unit is duplicated.
    """

    if not source_paragraphs:
        return None
    source_file = source_paragraphs[0].source_file.resolve()
    if any(item.source_file.resolve() != source_file for item in source_paragraphs):
        return None
    selected_ids = {item.paragraph_id for item in source_paragraphs}
    start = min(line for item in source_paragraphs for line in item.source_lines)
    end = max(line for item in source_paragraphs for line in item.source_lines)
    for item in all_list_paragraphs:
        if item.paragraph_id in selected_ids or item.source_file.resolve() != source_file:
            continue
        if any(start <= line <= end for line in item.source_lines):
            return None
    try:
        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines(
            keepends=True
        )
    except OSError:
        return None
    if start < 1 or end > len(lines):
        return None
    raw = "".join(lines[start - 1 : end]).rstrip("\r\n")
    if not raw.strip():
        return None
    return tuple(sorted({line for item in source_paragraphs for line in item.source_lines})), raw


def _partition_source_list_instances(
    paragraphs: Sequence[page_gt.SourceParagraph],
) -> list[list[page_gt.SourceParagraph]]:
    """Split execution-ordered records at provable top-level list resets.

    The source paragraph parser fixes ordinals independently for each literal
    list environment.  Therefore a literal top-level ``\\item`` whose ordinal
    resets (normally to one) is a hard instance boundary.  File transitions
    are also boundaries: joining a list across files without explicit parser
    ownership would be a guess.  Nested lists remain with their parent group.
    """

    groups: list[list[page_gt.SourceParagraph]] = []
    current: list[page_gt.SourceParagraph] = []
    root_environment: str | None = None
    last_top_ordinal: int | None = None
    current_file: Path | None = None
    for paragraph in paragraphs:
        depth = int(paragraph.item_depth or 0)
        ordinal = int(paragraph.item_ordinal or 0)
        literal_item = re.match(r"^\s*\\item(?:\s|\[|$)", paragraph.raw_latex) is not None
        source_file = paragraph.source_file.resolve()
        new_instance = bool(current) and source_file != current_file
        if (
            current
            and depth == 1
            and literal_item
            and (
                paragraph.list_environment != root_environment
                or last_top_ordinal is None
                or ordinal <= last_top_ordinal
            )
        ):
            new_instance = True
        if new_instance:
            groups.append(current)
            current = []
            root_environment = None
            last_top_ordinal = None
        if not current:
            current_file = source_file
            if depth == 1:
                root_environment = paragraph.list_environment
        current.append(paragraph)
        if depth == 1 and literal_item:
            last_top_ordinal = ordinal
            root_environment = root_environment or paragraph.list_environment
    if current:
        groups.append(current)
    return groups


def _list_record_for_instance(
    paragraph: page_gt.SourceParagraph,
    *,
    instance_id: str,
) -> dict[str, Any]:
    """Expose a unique explicit ID only for the top-level list lane."""

    return {
        "paragraph_id": paragraph.paragraph_id,
        "kind": paragraph.kind,
        "source_file": paragraph.source_file,
        "source_lines": tuple(paragraph.source_lines),
        "raw_latex": paragraph.raw_latex,
        "list_environment": paragraph.list_environment,
        "item_depth": paragraph.item_depth,
        "item_ordinal": paragraph.item_ordinal,
        "list_id": instance_id if paragraph.item_depth == 1 else None,
    }


def build_source_list_units(
    paragraphs: Sequence[page_gt.SourceParagraph],
    *,
    references: Mapping[str, stable.AuxReference] | None,
    color_index_offset: int = 0,
    execution_ir: ExecutionIR | None = None,
) -> tuple[list[stable.SourceUnit], list[dict[str, Any]], dict[str, Any]]:
    """Build one non-overlapping SourceUnit per source list item.

    The Markdown item/continuation merge is delegated to ``list_ir``.  The
    serializer consumes only expanded LaTeX source and the resulting unit
    keeps the original source range/raw text for later localization.
    """

    candidates = [
        paragraph
        for paragraph in paragraphs
        if paragraph.list_environment in {"itemize", "enumerate", "description"}
    ]
    if not candidates:
        return [], [], {"status": "skipped", "paragraphs": 0, "items": 0}
    ordered: list[page_gt.SourceParagraph] = []
    rejected: list[dict[str, Any]] = []
    if execution_ir is None:
        ordered = list(candidates)
    else:
        located: list[tuple[tuple[int, int], int, page_gt.SourceParagraph]] = []
        for index, paragraph in enumerate(candidates):
            resolution = execution_ir.resolve(
                paragraph.source_file,
                line=paragraph.start_line,
            )
            if not resolution.is_unique:
                rejected.append(
                    {
                        "source_paragraph_id": paragraph.paragraph_id,
                        "source_file": str(paragraph.source_file),
                        "source_lines": [paragraph.start_line, paragraph.end_line],
                        "reason": f"list_execution_ir_{resolution.status}",
                        "admission_stage": "source_list_ir",
                    }
                )
                continue
            assert resolution.ordinal is not None
            located.append((resolution.ordinal.key, index, paragraph))
        located.sort(key=lambda value: (value[0], value[1]))
        ordered = [value[2] for value in located]
    instance_groups = _partition_source_list_instances(ordered)
    successful_results: list[tuple[str, Any, list[page_gt.SourceParagraph]]] = []
    rejected_instances: list[dict[str, Any]] = []
    for group_index, group in enumerate(instance_groups, start=1):
        first = group[0]
        identity = (
            f"{first.source_file.resolve()}|{first.start_line}|"
            f"{first.list_environment}|{first.paragraph_id}"
        )
        instance_id = "sfv2-list-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]
        records = [
            _list_record_for_instance(paragraph, instance_id=instance_id)
            for paragraph in group
        ]
        try:
            result = serialize_source_list(
                records,
                render_inline=lambda value: _render_list_inline_source(
                    value, references
                ),
            )
        except (
            ListIRSafetyError,
            ValueError,
            stable.inline_markup.InlineParseError,
        ) as error:
            reason = (
                f"source_list_serialization_failed:{type(error).__name__}:{error}"
            )
            rejected.extend(
                {
                    "source_paragraph_id": paragraph.paragraph_id,
                    "source_file": str(paragraph.source_file),
                    "source_lines": [paragraph.start_line, paragraph.end_line],
                    "list_instance_id": instance_id,
                    "reason": reason,
                    "admission_stage": "source_list_ir",
                }
                for paragraph in group
            )
            rejected_instances.append(
                {
                    "instance_id": instance_id,
                    "instance_ordinal": group_index,
                    "paragraph_ids": [item.paragraph_id for item in group],
                    "reason": reason,
                }
            )
            continue
        successful_results.append((instance_id, result, group))

    original_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    list_units: list[stable.SourceUnit] = []
    item_rejections: list[dict[str, Any]] = []
    all_serialized_items = [
        item
        for _instance_id, result, _group in successful_results
        for item in result.items
    ]
    for item in all_serialized_items:
        original_fragments = [
            original_by_id.get(paragraph_id) for paragraph_id in item.paragraph_ids
        ]
        if any(fragment is None for fragment in original_fragments):
            item_rejections.append(
                {
                    "item_key": item.item_key,
                    "paragraph_ids": list(item.paragraph_ids),
                    "reason": "source_list_original_provenance_missing",
                    "admission_stage": "source_list_ir",
                }
            )
            continue
        typed_fragments = [fragment for fragment in original_fragments if fragment is not None]
        source_range = _source_range_for_list_item(
            typed_fragments,
            all_list_paragraphs=paragraphs,
        )
        if source_range is None:
            item_rejections.append(
                {
                    "item_key": item.item_key,
                    "paragraph_ids": list(item.paragraph_ids),
                    "reason": "source_list_item_source_range_overlaps_or_unavailable",
                    "admission_stage": "source_list_ir",
                }
            )
            continue
        source_lines, original_raw = source_range
        first = typed_fragments[0]
        index = color_index_offset + len(list_units)
        list_units.append(
            stable.SourceUnit(
                unit_id=f"src-{index + 1:07d}",
                kind=f"{item.environment}_item",
                paragraph_id=first.paragraph_id,
                source_file=first.source_file,
                source_lines=source_lines,
                raw_latex=original_raw,
                markdown=item.markdown,
                rgb=color_pilot.deterministic_rgb(index),
                source_command="list_ir",
            )
        )
    rejected.extend(item_rejections)
    report = {
        "status": (
            "passed"
            if list_units and not rejected_instances and not rejected
            else "partial"
            if list_units
            else "failed"
        ),
        "policy": "provable_top_level_list_instance_isolation_v1",
        "paragraphs": len(candidates),
        "paragraphs_execution_order_rejected": sum(
            str(item.get("reason", "")).startswith("list_execution_ir_")
            for item in rejected
        ),
        "instances_total": len(instance_groups),
        "instances_accepted": len(successful_results),
        "instances_rejected": len(rejected_instances),
        "rejected_instances": rejected_instances,
        "items": len(all_serialized_items),
        "units": len(list_units),
        "continuations": sum(
            item.continuation_count for item in all_serialized_items
        ),
        "rejected_items": len(item_rejections),
        "item_provenance": [item.as_json() for item in all_serialized_items],
        "generation_source": "latex_source",
        "pdf_text_used": False,
    }
    return list_units, rejected, report


def build_source_units_with_visible_wrappers(
    paragraphs: Sequence[page_gt.SourceParagraph],
    *,
    references: dict[str, stable.AuxReference] | None,
    macros: Mapping[str, VisibleWrapperMacro],
    safe_macros: SafeMacroRegistry | None = None,
    execution_ir: ExecutionIR | None = None,
) -> tuple[
    list[stable.SourceUnit],
    list[dict[str, Any]],
    dict[str, VisibleWrapperInvocation],
    dict[str, Any],
]:
    """Admit every ordinary paragraph through source-only macro expansion.

    The expanded text is only a serialization view.  Every returned unit keeps
    the original ``raw_latex`` and ``source_lines`` so coloring/localization
    provenance continues to point at the compiled project source.
    """

    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    admitted_paragraphs: list[page_gt.SourceParagraph] = []
    pre_admission_rejections: list[dict[str, Any]] = []
    structural_argument_rejections: list[dict[str, Any]] = []
    expansion_by_paragraph: dict[str, dict[str, Any]] = {}
    changed_paragraph_ids: set[str] = set()
    syntactic_indexes: dict[Path, SourceSyntacticIndex | None] = {}
    for paragraph in paragraphs:
        resolved_source = paragraph.source_file.resolve()
        if resolved_source not in syntactic_indexes:
            syntactic_indexes[resolved_source] = build_source_syntactic_index(
                paragraph.source_file
            )
        syntactic_gate = source_fragment_syntactic_gate(
            paragraph.source_file,
            paragraph.source_lines,
            paragraph.raw_latex,
            require_payload_boundary=False,
            index=syntactic_indexes[resolved_source],
        )
        if syntactic_gate.get("reason") in {
            "environment_optional_argument_fragment",
            "control_sequence_argument_fragment",
        }:
            rejection = {
                "source_paragraph_id": paragraph.paragraph_id,
                "source_file": str(paragraph.source_file),
                "source_lines": [paragraph.start_line, paragraph.end_line],
                "reason": f"structural_argument_fragment:{syntactic_gate['reason']}",
                "admission_stage": "source_boundary_pre_admission",
                "syntactic_provenance": syntactic_gate,
            }
            pre_admission_rejections.append(rejection)
            structural_argument_rejections.append(rejection)
            continue
        expanded = paragraph.raw_latex
        expansion_events: list[dict[str, Any]] = []
        expansion_rejections: list[dict[str, Any]] = []
        unresolved_accepted: list[str] = []
        invoked_rejected: list[str] = []
        if safe_macros is not None:
            invoked_accepted = _registered_macro_calls(
                paragraph.raw_latex, safe_macros.accepted_names
            )
            base_offset = _source_fragment_base_offset(
                paragraph.source_file,
                paragraph.source_lines,
                paragraph.raw_latex,
            )
            if invoked_accepted and base_offset is None:
                expansion_rejections = [
                    {
                        "reason": "source fragment offset is ambiguous",
                        "macros": invoked_accepted,
                    }
                ]
            else:
                expanded, expansion_events, expansion_rejections = (
                    expand_registered_macro_calls(
                        paragraph.raw_latex,
                        safe_macros,
                        source_file=paragraph.source_file,
                        source_base_offset=base_offset or 0,
                    )
                )
            unresolved_accepted = _registered_macro_calls(
                expanded, safe_macros.accepted_names
            )
            invoked_rejected = _registered_macro_calls(
                paragraph.raw_latex, safe_macros.rejected_names
            )
        failure_reasons: list[str] = []
        if expansion_rejections:
            failure_reasons.append("safe macro expansion rejected")
        if unresolved_accepted:
            failure_reasons.append("safe macro expansion incomplete")
        if invoked_rejected:
            failure_reasons.append("rejected project macro invoked")
        if failure_reasons:
            pre_admission_rejections.append(
                {
                    "source_paragraph_id": paragraph.paragraph_id,
                    "source_file": str(paragraph.source_file),
                    "source_lines": [paragraph.start_line, paragraph.end_line],
                    "reason": "; ".join(failure_reasons),
                    "admission_stage": "safe_macro_pre_admission",
                    "unresolved_accepted_macros": unresolved_accepted,
                    "invoked_rejected_macros": invoked_rejected,
                    "macro_expansion_rejections": expansion_rejections,
                    "macro_provenance": expansion_events,
                }
            )
            continue
        changed = expanded != paragraph.raw_latex
        if changed:
            changed_paragraph_ids.add(paragraph.paragraph_id)
            expansion_by_paragraph[paragraph.paragraph_id] = {
                "source_paragraph_id": paragraph.paragraph_id,
                "source_file": str(paragraph.source_file),
                "source_lines": [paragraph.start_line, paragraph.end_line],
                "original_source_sha256": hashlib.sha256(
                    paragraph.raw_latex.encode("utf-8")
                ).hexdigest(),
                "expanded_source_sha256": hashlib.sha256(
                    expanded.encode("utf-8")
                ).hexdigest(),
                "macros": sorted(
                    {
                        str(event.get("macro_name"))
                        for event in expansion_events
                        if event.get("macro_name")
                    }
                ),
                "provenance": expansion_events,
                "original_raw_latex_preserved": True,
                "original_source_lines_preserved": True,
            }
        admitted_paragraphs.append(
            dataclasses.replace(
                paragraph,
                raw_latex=_normalize_safe_macro_visible_styles(expanded),
            )
        )

    ordinary_admitted_paragraphs = [
        paragraph
        for paragraph in admitted_paragraphs
        if paragraph.list_environment not in {"itemize", "enumerate", "description"}
    ]
    base_units, base_rejections = stable.build_source_units(
        ordinary_admitted_paragraphs, references=references
    )
    # The stable serializer has consumed the expanded view.  Restore exact
    # source provenance before any locator is built.
    base_units = [
        dataclasses.replace(
            unit,
            raw_latex=paragraph_by_id[unit.paragraph_id].raw_latex,
            source_lines=tuple(paragraph_by_id[unit.paragraph_id].source_lines),
        )
        for unit in base_units
    ]
    retained_rejections: list[dict[str, Any]] = list(pre_admission_rejections)
    wrapper_units: dict[str, VisibleWrapperInvocation] = {}
    recovered: list[dict[str, Any]] = []
    safe_macro_expansion_rejections: list[dict[str, Any]] = [
        rejection
        for rejection in pre_admission_rejections
        if rejection.get("admission_stage") == "safe_macro_pre_admission"
    ]
    for rejection in base_rejections:
        paragraph_id = str(rejection.get("source_paragraph_id") or "")
        paragraph = paragraph_by_id.get(paragraph_id)
        if paragraph is None or paragraph.kind != "paragraph":
            retained_rejections.append(rejection)
            continue
        if paragraph_id in changed_paragraph_ids:
            value = {
                **rejection,
                "admission_stage": "strict_source_markdown_serialization",
                "macro_provenance": expansion_by_paragraph[paragraph_id][
                    "provenance"
                ],
            }
            retained_rejections.append(value)
            safe_macro_expansion_rejections.append(value)
            # A second, wrapper-specific interpretation after expansion would
            # no longer have exact invocation offsets in the original source.
            # Fail closed rather than fabricate locator provenance.
            continue
        invocation, reason = parse_visible_wrapper_invocation(
            paragraph.raw_latex, macros
        )
        if invocation is None:
            retained_rejections.append(
                {**rejection, "visible_wrapper_recovery_reason": reason}
            )
            continue
        expanded_paragraph = dataclasses.replace(
            paragraph, raw_latex=invocation.expanded_source
        )
        try:
            markdown = stable.source_paragraph_to_markdown(
                expanded_paragraph, references
            )
        except (ValueError, stable.inline_markup.InlineParseError) as error:
            retained_rejections.append(
                {
                    **rejection,
                    "visible_wrapper_recovery_reason": (
                        f"expanded_source_rejected:{type(error).__name__}:{error}"
                    ),
                }
            )
            continue
        atoms = build_source_atoms(invocation.expanded_source)
        if (
            any(atom.kind == "opaque" for atom in atoms)
            or atoms_to_markdown(atoms).strip() != markdown.strip()
        ):
            retained_rejections.append(
                {
                    **rejection,
                    "visible_wrapper_recovery_reason": (
                        "expanded_source_atom_markdown_mismatch"
                    ),
                }
            )
            continue
        index = len(base_units)
        unit = stable.SourceUnit(
            unit_id=f"src-{index + 1:07d}",
            kind=paragraph.kind,
            paragraph_id=paragraph.paragraph_id,
            source_file=paragraph.source_file,
            source_lines=tuple(paragraph.source_lines),
            raw_latex=paragraph.raw_latex,
            markdown=markdown,
            rgb=color_pilot.deterministic_rgb(index),
            source_command=invocation.macro.name,
        )
        base_units.append(unit)
        wrapper_units[unit.unit_id] = invocation
        recovered.append(
            {
                "source_unit_id": unit.unit_id,
                "source_paragraph_id": paragraph.paragraph_id,
                "macro": invocation.macro.name,
                "source_file": str(paragraph.source_file),
                "source_lines": [paragraph.start_line, paragraph.end_line],
            }
        )
    list_units, list_rejections, list_report = build_source_list_units(
        admitted_paragraphs,
        references=references,
        color_index_offset=len(base_units),
        execution_ir=execution_ir,
    )
    base_units.extend(list_units)
    retained_rejections.extend(list_rejections)
    successful_ids = {unit.paragraph_id for unit in base_units}
    safe_macro_recovered = [
        expansion_by_paragraph[paragraph_id]
        for paragraph_id in sorted(changed_paragraph_ids)
        if paragraph_id in successful_ids
    ]
    admission_report = {
        "policy": "safe_macro_pre_admission_v1",
        "total": len(paragraphs),
        "changed": len(changed_paragraph_ids),
        "successful": len(successful_ids),
        "rejected": len(retained_rejections),
        "provenance": safe_macro_recovered,
        "rejections": retained_rejections,
        "structural_argument_fragments_rejected": len(
            structural_argument_rejections
        ),
        "structural_argument_rejections": structural_argument_rejections,
        "original_source_provenance_preserved": True,
        "pdf_text_used": False,
    }
    return base_units, retained_rejections, wrapper_units, {
        "recovered_units": len(recovered),
        "recovered": recovered,
        "safe_macro_recovered_units": len(safe_macro_recovered),
        "safe_macro_recovered": safe_macro_recovered,
        "safe_macro_expansion_rejections": safe_macro_expansion_rejections,
        "structural_argument_fragments_rejected": len(
            structural_argument_rejections
        ),
        "structural_argument_rejections": structural_argument_rejections,
        "remaining_rejections": len(retained_rejections),
        "safe_macro_admission": admission_report,
        "list_units": len(list_units),
        "list_admission": list_report,
        "list_rejections": list_rejections,
    }


def scoped_color_prefix(rgb: tuple[int, int, int], engine: str) -> str:
    # Do not wrap inline text in PDF ``q``/``Q`` graphics-state pairs.  In
    # several classes Q restores a stale text matrix and causes later words to
    # overprint at an earlier position.  The stable primitive is deliberately
    # reused by value (the stable file itself remains untouched).
    return stable.pdf_literal_color(rgb, engine)


def scoped_color_suffix(engine: str) -> str:
    return stable.pdf_literal_restore(engine)


def source_offsets_for_unit(unit: stable.SourceUnit) -> tuple[str, int] | None:
    source = unit.source_file.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines(keepends=True)
    if unit.start_line < 1 or unit.end_line > len(lines):
        return None
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    left = starts[unit.start_line - 1]
    right = starts[unit.end_line] if unit.end_line < len(starts) else len(source)
    window = source[left:right]
    relative = window.find(unit.raw_latex)
    if relative < 0 or window.find(unit.raw_latex, relative + 1) >= 0:
        return None
    return source, left + relative


def build_macro_expansion_instrumentation_mismatches(
    units: Sequence[stable.SourceUnit],
    provenance: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find expanded GT units whose original source atoms reconstruct differently.

    Macro expansion is permitted for source-derived Markdown serialization, but
    atom/whole color locators must never pretend that expanded Markdown has
    executable offsets in the untouched macro call.  The returned mapping is
    source-only and carries the exact expansion records that triggered it.
    """

    normalized: list[tuple[str | None, Path | None, int, int, Mapping[str, Any]]] = []
    for item in provenance:
        identifier = item.get("source_paragraph_id") or item.get("source_block_id")
        source_value = item.get("source_file")
        source_path = Path(str(source_value)).resolve() if source_value else None
        lines = item.get("source_lines") or []
        if not isinstance(lines, Sequence) or len(lines) < 2:
            continue
        normalized.append(
            (
                str(identifier) if identifier is not None else None,
                source_path,
                int(lines[0]),
                int(lines[-1]),
                item,
            )
        )

    mismatches: dict[str, dict[str, Any]] = {}
    for unit in units:
        matching = [
            item
            for identifier, source_path, start, end, item in normalized
            if identifier == unit.paragraph_id
            or (
                source_path == unit.source_file.resolve()
                and not (unit.end_line < start or unit.start_line > end)
            )
        ]
        if not matching:
            continue
        atom_error: str | None = None
        atoms: tuple[SourceAtom, ...] = ()
        try:
            atoms = build_source_atoms(unit.raw_latex)
            reconstructed = atoms_to_markdown(atoms).strip()
        except Exception as error:  # noqa: BLE001 - locator gate is fail closed
            reconstructed = ""
            atom_error = f"{type(error).__name__}: {error}"
        if atom_error is None and reconstructed == unit.markdown.strip():
            continue
        mismatches[unit.unit_id] = {
            "reason": "macro_expanded_markdown_original_atom_mismatch",
            "source_file": str(unit.source_file),
            "source_lines": [unit.start_line, unit.end_line],
            "original_raw_source_sha256": hashlib.sha256(
                unit.raw_latex.encode("utf-8")
            ).hexdigest(),
            "original_atom_markdown_sha256": hashlib.sha256(
                reconstructed.encode("utf-8")
            ).hexdigest(),
            "expanded_markdown_sha256": hashlib.sha256(
                unit.markdown.strip().encode("utf-8")
            ).hexdigest(),
            "original_atoms_contain_opaque": any(
                atom.kind == "opaque" for atom in atoms
            ),
            "original_atom_reconstruction_error": atom_error,
            "macro_provenance": matching,
            "source_provenance_preserved": True,
            "pdf_text_used": False,
        }
    return mismatches


def _unit_instrumentation_safety(
    unit: stable.SourceUnit,
    *,
    source_index: SourceSyntacticIndex | None,
    macro_expansion_mismatches: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    provenance: list[dict[str, Any]] = []
    if unit.kind in {
        "paragraph",
        "itemize_item",
        "enumerate_item",
        "description_item",
    }:
        syntactic = source_fragment_syntactic_gate(
            unit.source_file,
            unit.source_lines,
            unit.raw_latex,
            require_payload_boundary=True,
            index=source_index,
        )
        if syntactic["status"] != "passed":
            reasons.append(str(syntactic["reason"]))
            provenance.append(syntactic)
    macro_mismatch = macro_expansion_mismatches.get(unit.unit_id)
    if macro_mismatch is not None:
        reasons.append("macro_expanded_markdown_original_atom_mismatch")
        provenance.append(dict(macro_mismatch))
    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "source_file": str(unit.source_file),
        "source_lines": [unit.start_line, unit.end_line],
        "status": "metadata_only" if reasons else "executable",
        "reasons": sorted(set(reasons)),
        "provenance": provenance,
        "localization_fallback": "synctex_clean_only" if reasons else None,
        "source_provenance_preserved": True,
        "pdf_text_used": False,
    }


def instrument_synctex_line_identity_tree(
    clean_root: Path,
    locator_root: Path,
    probes: Sequence[stable.SourceProbe],
    atom_locators: Mapping[str, AtomLocator],
) -> tuple[dict[str, tuple[Path, tuple[int, ...]]], dict[str, Any]]:
    """Give safe source atoms unique SyncTeX line identities.

    A comment plus newline is inserted immediately *before* each already
    parsed visible atom.  TeX's comment consumes the inserted newline, so the
    input token stream is unchanged.  The resulting PDF must nevertheless
    pass an exact glyph-geometry gate before any metadata is trusted.
    """

    clean_root = clean_root.resolve()
    locator_root = locator_root.resolve(strict=False)
    shutil.copytree(clean_root, locator_root)
    probes_by_file: dict[Path, list[stable.SourceProbe]] = collections.defaultdict(list)
    edits_by_file: dict[Path, dict[int, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for probe in probes:
        probes_by_file[probe.source_file.resolve()].append(probe)
        locator = atom_locators.get(probe.probe_id)
        if locator is not None:
            edits_by_file[locator.source_file.resolve()][int(locator.source_start)].append(
                probe.probe_id
            )

    overrides: dict[str, tuple[Path, tuple[int, ...]]] = {}
    inserted = 0
    files_instrumented = 0
    duplicate_positions = 0
    for source_file, file_probes in sorted(
        probes_by_file.items(), key=lambda item: str(item[0])
    ):
        try:
            relative = source_file.relative_to(clean_root)
        except ValueError as error:
            raise ValueError(
                f"SyncTeX identity source escapes clean root: {source_file}"
            ) from error
        target = locator_root / relative
        source = source_file.read_text(encoding="utf-8", errors="replace")
        positions = edits_by_file.get(source_file, {})
        expanded_positions: list[int] = []
        chunks: list[str] = []
        cursor = 0
        for position, probe_ids in sorted(positions.items()):
            if position < cursor or position > len(source):
                raise ValueError(
                    f"invalid SyncTeX identity insertion {source_file}:{position}"
                )
            chunks.append(source[cursor:position])
            safe_ids = sorted(set(probe_ids))
            duplicate_positions += max(0, len(safe_ids) - 1)
            for probe_id in safe_ids:
                if re.fullmatch(r"[A-Za-z0-9_-]+", probe_id) is None:
                    raise ValueError(f"unsafe SyncTeX marker id: {probe_id!r}")
                chunks.append(f"%SFV2SYNC:{probe_id}\n")
                expanded_positions.append(position)
                inserted += 1
            cursor = position
        chunks.append(source[cursor:])
        if positions:
            atomic_write_text(target, "".join(chunks))
            files_instrumented += 1

        line_starts = [0]
        line_starts.extend(
            index + 1 for index, character in enumerate(source) if character == "\n"
        )
        line_ends = [*line_starts[1:], len(source)]
        positions_sorted = sorted(expanded_positions)
        line_map: dict[int, tuple[int, ...]] = {}
        for line_number, (left, right) in enumerate(
            zip(line_starts, line_ends), start=1
        ):
            before = bisect.bisect_left(positions_sorted, left)
            through = bisect.bisect_left(positions_sorted, right)
            first_generated = line_number + before
            line_map[line_number] = tuple(
                range(first_generated, first_generated + (through - before) + 1)
            )
        for probe in file_probes:
            locator = atom_locators.get(probe.probe_id)
            if locator is not None:
                original_line = bisect.bisect_right(
                    line_starts, int(locator.source_start)
                )
                generated_line = original_line + bisect.bisect_right(
                    positions_sorted, int(locator.source_start)
                )
                lines = (generated_line,)
            else:
                lines = tuple(
                    line
                    for original in probe.source_lines
                    for line in line_map.get(int(original), ())
                )
            overrides[probe.probe_id] = (target.resolve(strict=False), lines)
    return overrides, {
        "policy": "comment_newline_before_safe_visible_atom_v1",
        "files_instrumented": files_instrumented,
        "markers_inserted": inserted,
        "duplicate_source_positions": duplicate_positions,
        "probe_line_overrides": len(overrides),
        "visible_token_stream_intended_unchanged": True,
        "requires_exact_pdf_geometry_gate": True,
    }


def instrument_external_verbatim_color_tree(
    clean_root: Path,
    shadow_root: Path,
    blocks_by_unit: Mapping[str, ExternalVerbatimBlock],
    probes: Sequence[stable.SourceProbe],
    engine: str,
) -> dict[str, Any]:
    """Color literal ``verbatiminput`` lines through its source line hook."""

    clean_root = clean_root.resolve()
    shadow_root = shadow_root.resolve(strict=False)
    shutil.copytree(clean_root, shadow_root)
    probe_by_id = {probe.probe_id: probe for probe in probes}
    edits: dict[Path, list[tuple[int, int, str, str]]] = collections.defaultdict(list)
    blocks_instrumented = lines_instrumented = unsupported = 0
    for unit_id, block in blocks_by_unit.items():
        if block.command_name != "verbatiminput":
            unsupported += 1
            continue
        colors_by_line = {
            record.external_source_line: probe_by_id[record.record_id].rgb
            for record in block.records
            if record.record_id in probe_by_id
        }
        if not colors_by_line:
            continue
        maximum_line = max(colors_by_line)
        branches: list[str] = [""]
        black = stable.pdf_literal_restore(engine)
        for line_number in range(1, maximum_line + 1):
            rgb = colors_by_line.get(line_number)
            branches.append(
                stable.pdf_literal_color(rgb, engine) if rgb is not None else black
            )
        selector = "\\ifcase\\sfvTwoVerbatimLine" + "\\or".join(branches)
        selector += "\\else" + black + "\\fi"
        prefix = (
            "\\begingroup\\makeatletter"
            "\\newcount\\sfvTwoVerbatimLine\\sfvTwoVerbatimLine=0\\relax"
            "\\def\\verbatim@processline{"
            "\\advance\\sfvTwoVerbatimLine by 1\\relax"
            "\\leavevmode"
            + selector
            + "\\the\\verbatim@line"
            + black
            + "\\par}\\makeatother "
        )
        edits[block.execution_source.resolve()].append(
            (
                block.execution_source_byte_start,
                block.execution_source_byte_end,
                prefix,
                "\\endgroup",
            )
        )
        blocks_instrumented += 1
        lines_instrumented += len(colors_by_line)

    for source_file, file_edits in edits.items():
        relative = source_file.relative_to(clean_root)
        target = shadow_root / relative
        source_bytes = source_file.read_bytes()
        source = source_bytes.decode("utf-8")
        converted: list[tuple[int, int, str, str]] = []
        for byte_start, byte_end, prefix, suffix in file_edits:
            try:
                start = len(source_bytes[:byte_start].decode("utf-8"))
                end = len(source_bytes[:byte_end].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"external verbatim invocation byte boundary is not UTF-8: {source_file}"
                ) from error
            converted.append((start, end, prefix, suffix))
        for start, end, prefix, suffix in sorted(converted, reverse=True):
            source = source[:start] + prefix + source[start:end] + suffix + source[end:]
        atomic_write_text(target, source)
    return {
        "policy": "verbatim_processline_leavevmode_pdf_color_v2",
        "blocks_instrumented": blocks_instrumented,
        "lines_instrumented": lines_instrumented,
        "unsupported_blocks": unsupported,
        "source_content_from_external_files": True,
        "pdf_text_used": False,
        "requires_exact_pdf_geometry_gate": True,
    }


def allocate_color(used: set[tuple[int, int, int]], cursor: list[int]) -> tuple[int, int, int]:
    while True:
        rgb = color_pilot.deterministic_rgb(cursor[0])
        cursor[0] += 1
        if rgb not in used:
            used.add(rgb)
            return rgb


def build_atom_probes(
    units: Sequence[stable.SourceUnit],
    base_probes: Sequence[stable.SourceProbe],
    wrapper_invocations: Mapping[str, VisibleWrapperInvocation] | None = None,
    *,
    include_plain_source_atoms: bool = True,
) -> tuple[
    list[stable.SourceProbe],
    dict[str, AtomLocator],
    dict[str, tuple[SourceAtom, ...]],
    dict[str, str],
]:
    """Replace safe paragraph probes with markup-aware source atoms."""

    probes_by_unit: dict[str, list[stable.SourceProbe]] = collections.defaultdict(list)
    for probe in base_probes:
        probes_by_unit[probe.unit_id].append(probe)
    used = {probe.rgb for probe in base_probes} | {unit.rgb for unit in units}
    color_cursor = [2_000_000]
    output: list[stable.SourceProbe] = []
    locators: dict[str, AtomLocator] = {}
    unit_atoms: dict[str, tuple[SourceAtom, ...]] = {}
    modes: dict[str, str] = {}
    wrappers = wrapper_invocations or {}
    for unit in units:
        replacement: list[stable.SourceProbe] = []
        location = source_offsets_for_unit(unit) if unit.kind == "paragraph" else None
        if location is not None:
            _source, base_offset = location
            wrapper = wrappers.get(unit.unit_id)
            if wrapper is None and not include_plain_source_atoms:
                atoms = ()
                visible_atoms = []
                localization_mode = "source_atom"
            elif wrapper is None:
                atoms = build_source_atoms(
                    unit.raw_latex, source_base_offset=base_offset
                )
                visible_atoms = [
                    atom
                    for atom in atoms
                    if atom.kind not in {"whitespace", "opaque"}
                    and atom.visible_text.strip()
                ]
                localization_mode = "source_atom"
            else:
                atoms = build_source_atoms(wrapper.expanded_source)
                visible_atoms = [
                    atom
                    for atom in atoms
                    if atom.source_start >= wrapper.expanded_argument_start
                    and atom.source_end <= wrapper.expanded_argument_end
                    and atom.kind not in {"whitespace", "opaque"}
                    and atom.visible_text.strip()
                ]
                localization_mode = "source_wrapper_atom"
            exact_source_markdown = atoms_to_markdown(atoms).strip() == unit.markdown.strip()
            if visible_atoms and exact_source_markdown and not any(
                atom.kind == "opaque" for atom in atoms
            ):
                total = len(visible_atoms)
                for ordinal, atom in enumerate(visible_atoms, start=1):
                    probe_id = f"{unit.unit_id}-atom-{ordinal:05d}"
                    replacement.append(
                        stable.SourceProbe(
                            probe_id=probe_id,
                            unit_id=unit.unit_id,
                            paragraph_id=unit.paragraph_id,
                            kind=unit.kind,
                            source_file=unit.source_file,
                            source_lines=unit.source_lines,
                            markdown_fragment=atom.markdown_fragment,
                            rgb=allocate_color(used, color_cursor),
                            ordinal=ordinal,
                            total=total,
                            localization_mode=localization_mode,
                            token_span=None,
                        )
                    )
                    if wrapper is None:
                        source_start = atom.source_start
                        source_end = atom.source_end
                    else:
                        source_start = (
                            base_offset
                            + wrapper.argument_start
                            + atom.source_start
                            - wrapper.expanded_argument_start
                        )
                        source_end = (
                            base_offset
                            + wrapper.argument_start
                            + atom.source_end
                            - wrapper.expanded_argument_start
                        )
                    locators[probe_id] = AtomLocator(
                        probe_id=probe_id,
                        source_file=unit.source_file,
                        source_start=source_start,
                        source_end=source_end,
                        atom_ordinal=atom.ordinal,
                    )
                unit_atoms[unit.unit_id] = atoms
        if replacement:
            output.extend(replacement)
            modes[unit.unit_id] = replacement[0].localization_mode
        else:
            output.extend(probes_by_unit[unit.unit_id])
            modes[unit.unit_id] = probes_by_unit[unit.unit_id][0].localization_mode
    return output, locators, unit_atoms, modes


def external_verbatim_units(
    ir: ExternalVerbatimIR,
    *,
    color_index_offset: int,
) -> tuple[list[stable.SourceUnit], dict[str, ExternalVerbatimBlock]]:
    """Convert safe external-source blocks to execution-order source units."""

    units: list[stable.SourceUnit] = []
    blocks_by_unit: dict[str, ExternalVerbatimBlock] = {}
    for block in ir.blocks:
        if not block.records:
            continue
        index = color_index_offset + len(units)
        unit_id = f"extverb-{index + 1:07d}"
        unit = stable.SourceUnit(
            unit_id=unit_id,
            kind="external_verbatim",
            paragraph_id=block.call_id,
            source_file=block.execution_source,
            source_lines=(block.execution_source_line,),
            raw_latex=block.visible_text,
            markdown=block.fenced_markdown,
            rgb=color_pilot.deterministic_rgb(index),
            source_command=block.command_name,
        )
        units.append(unit)
        blocks_by_unit[unit_id] = block
    return units, blocks_by_unit


def replace_external_verbatim_line_probes(
    probes: Sequence[stable.SourceProbe],
    modes: Mapping[str, str],
    blocks_by_unit: Mapping[str, ExternalVerbatimBlock],
) -> tuple[list[stable.SourceProbe], dict[str, str]]:
    """Use one source-file line probe per visible external verbatim line."""

    if not blocks_by_unit:
        return list(probes), dict(modes)
    output = [probe for probe in probes if probe.unit_id not in blocks_by_unit]
    updated_modes = dict(modes)
    used = {probe.rgb for probe in probes}
    color_cursor = [3_000_000]
    for unit_id, block in blocks_by_unit.items():
        visible = [
            record for record in block.records if record.visible_text.strip()
        ]
        if not visible:
            updated_modes.pop(unit_id, None)
            continue
        record_ordinal = {
            record.record_id: index
            for index, record in enumerate(block.records, start=1)
        }
        for record in visible:
            output.append(
                stable.SourceProbe(
                    probe_id=record.record_id,
                    unit_id=unit_id,
                    paragraph_id=block.call_id,
                    kind="external_verbatim",
                    source_file=record.external_source,
                    source_lines=(record.external_source_line,),
                    markdown_fragment=record.visible_text,
                    rgb=allocate_color(used, color_cursor),
                    ordinal=record_ordinal[record.record_id],
                    total=len(block.records),
                    localization_mode=EXTERNAL_VERBATIM_LOCALIZATION_MODE,
                    token_span=None,
                )
            )
        updated_modes[unit_id] = EXTERNAL_VERBATIM_LOCALIZATION_MODE
    return output, updated_modes


def structural_units(
    blocks: Sequence[page_gt.SourceBlock],
    *,
    color_index_offset: int,
    table_numbers: Mapping[str, Sequence[str]] | None = None,
    references: Mapping[str, stable.AuxReference] | None = None,
) -> tuple[list[stable.SourceUnit], list[dict[str, Any]]]:
    units: list[stable.SourceUnit] = []
    rejected: list[dict[str, Any]] = []
    number_queues = {
        key: list(values) for key, values in (table_numbers or {}).items()
    }
    for block in blocks:
        if block.kind not in STRUCTURAL_KINDS:
            continue
        markdown = (block.markdown or "").strip()
        if not markdown:
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "kind": block.kind,
                    "reason": "empty structural Markdown",
                }
            )
            continue
        if block.kind == "table" and block.table_parse_status != "parsed":
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "kind": block.kind,
                    "reason": f"table parse status={block.table_parse_status}",
                }
            )
            continue
        if block.kind == "table" and not (block.caption_markdown or "").strip():
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "kind": block.kind,
                    "reason": "table has no safe caption locator",
                }
            )
            continue
        if block.kind == "table" and block.caption_markdown:
            compiler_number: str | None = None
            literal_labels = re.findall(
                r"\\label\s*\{([^{}]+)\}", block.raw_latex
            )
            table_refs = [
                (references or {}).get(label)
                for label in literal_labels
                if (references or {}).get(label) is not None
                and (references or {})[label].kind in {"table", "subtable"}
            ]
            unique_label_numbers = {
                reference.number.strip()
                for reference in table_refs
                if reference is not None and reference.number.strip()
            }
            if len(unique_label_numbers) == 1:
                compiler_number = next(iter(unique_label_numbers))
            if compiler_number is None:
                key = page_gt.normalize_space(
                    page_gt.latex_to_plain(block.caption_markdown)
                ).casefold()
                queue = number_queues.get(key, [])
                if queue:
                    compiler_number = queue.pop(0)
            if compiler_number:
                markdown = f"Table {compiler_number}\n\n{markdown}"
        index = color_index_offset + len(units)
        units.append(
            stable.SourceUnit(
                unit_id=f"struct-{index + 1:07d}",
                kind=block.kind,
                paragraph_id=f"source-block-{block.block_id}",
                source_file=block.source_file,
                source_lines=tuple(range(block.start_line, block.end_line + 1)),
                raw_latex=block.raw_latex,
                markdown=markdown,
                rgb=color_pilot.deterministic_rgb(index),
                source_command=None,
            )
        )
    return units, rejected


def parse_aux_label_number_candidates(
    aux_paths: Iterable[Path],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Collect unique compiler numbers per literal AUX label.

    Unlike the stable convenience mapping, this experimental reader preserves
    conflicting numbers.  ``structural_ir`` therefore receives a sequence and
    rejects a theorem/equation unless the label resolves to exactly one unique
    compiler number.  Repeated identical ``\\newlabel`` records are harmless.
    """

    observed: dict[str, list[str]] = collections.defaultdict(list)
    records = rejected = 0
    files_read = 0
    for aux_path in aux_paths:
        if not aux_path.is_file():
            continue
        files_read += 1
        for line in aux_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            marker = re.search(r"\\newlabel\s*", line)
            if marker is None:
                continue
            cursor = marker.end()
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            label_group = page_gt.extract_balanced(line, cursor)
            if label_group is None:
                rejected += 1
                continue
            label, cursor = label_group
            if not label.strip() or label.endswith("@cref"):
                continue
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            payload_group = page_gt.extract_balanced(line, cursor)
            if payload_group is None:
                rejected += 1
                continue
            payload = payload_group[0]
            field_cursor = 0
            while field_cursor < len(payload) and payload[field_cursor].isspace():
                field_cursor += 1
            number_group = page_gt.extract_balanced(payload, field_cursor)
            if number_group is None:
                rejected += 1
                continue
            number = number_group[0].strip()
            if re.fullmatch(
                r"[A-Za-z0-9]+(?:[.\-:][A-Za-z0-9]+)*", number
            ) is None:
                rejected += 1
                continue
            observed[label.strip()].append(number)
            records += 1
    resolved: dict[str, tuple[str, ...]] = {}
    ambiguous = 0
    duplicate_records = 0
    for label, values in observed.items():
        unique = tuple(dict.fromkeys(values))
        resolved[label] = unique
        ambiguous += len(unique) > 1
        duplicate_records += len(values) - len(unique)
    return dict(sorted(resolved.items())), {
        "policy": "all_aux_unique_label_number_candidates_v1",
        "generation_source": "compiler_aux",
        "pdf_text_used": False,
        "files_read": files_read,
        "records_parsed": records,
        "records_rejected": rejected,
        "labels": len(resolved),
        "ambiguous_labels": ambiguous,
        "duplicate_identical_records": duplicate_records,
    }


def _source_lines_for_character_span(
    source: str, span: tuple[int, int]
) -> tuple[int, ...]:
    start, end = span
    if start < 0 or end <= start or end > len(source):
        return ()
    first = source.count("\n", 0, start) + 1
    last = source.count("\n", 0, end - 1) + 1
    return tuple(range(first, last + 1))


def build_theorem_heading_units(
    theorem_ir: TheoremStructuralIR,
    *,
    color_index_offset: int,
) -> tuple[
    list[stable.SourceUnit],
    dict[str, tuple[tuple[str, str], ...]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Admit source/AUX theorem headings as SyncTeX-only units.

    The unit owns only the literal ``\\begin{...}[optional title]`` span, not
    the theorem body.  Every heading spelling is frozen before PDF extraction.
    Color instrumentation is prohibited separately for ``theorem_heading``.
    """

    units: list[stable.SourceUnit] = []
    variants: dict[str, tuple[tuple[str, str], ...]] = {}
    rejected: list[dict[str, Any]] = []
    accepted_blocks: list[dict[str, Any]] = []
    source_cache: dict[Path, str] = {}
    for block in theorem_ir.blocks:
        source_file = block.source_file.resolve()
        try:
            if source_file not in source_cache:
                source_cache[source_file] = source_file.read_text(
                    encoding="utf-8", errors="strict"
                )
            source = source_cache[source_file]
        except (OSError, UnicodeError) as error:
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "kind": "theorem_heading",
                    "reason": f"exact UTF-8 theorem source unavailable:{error}",
                    "pdf_text_used": False,
                }
            )
            continue
        heading_end = (
            block.optional_title_span[1] + 1
            if block.optional_title_span is not None
            else block.begin_span[1]
        )
        heading_span = (block.begin_span[0], heading_end)
        source_lines = _source_lines_for_character_span(source, heading_span)
        candidate_values = tuple(
            dict.fromkeys(
                (candidate.policy, candidate.markdown)
                for candidate in block.heading_candidates
                if candidate.markdown.strip()
            )
        )
        if not source_lines or not candidate_values:
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "kind": "theorem_heading",
                    "reason": "theorem heading span or finite candidate set unavailable",
                    "source_character_span": list(heading_span),
                    "pdf_text_used": False,
                }
            )
            continue
        raw_latex = source[slice(*heading_span)]
        if not raw_latex.startswith(r"\begin"):
            rejected.append(
                {
                    "source_block_id": block.block_id,
                    "kind": "theorem_heading",
                    "reason": "theorem heading exact source span is inconsistent",
                    "source_character_span": list(heading_span),
                    "pdf_text_used": False,
                }
            )
            continue
        index = color_index_offset + len(units)
        unit_id = f"theorem-heading-{index + 1:07d}"
        unit = stable.SourceUnit(
            unit_id=unit_id,
            kind="theorem_heading",
            paragraph_id=block.block_id,
            source_file=source_file,
            source_lines=source_lines,
            raw_latex=raw_latex,
            markdown=candidate_values[0][1],
            rgb=color_pilot.deterministic_rgb(index),
            source_command=None,
        )
        units.append(unit)
        variants[unit_id] = candidate_values
        accepted_blocks.append(
            {
                "source_block_id": block.block_id,
                "source_file": str(source_file),
                "source_lines": [source_lines[0], source_lines[-1]],
                "source_character_span": list(heading_span),
                "environment": block.environment,
                "label": block.label,
                "aux_number": block.aux_number,
                "candidate_policies": [value[0] for value in candidate_values],
                "generation_sources": ["latex_source", "compiler_aux"],
                "pdf_text_used": False,
            }
        )
    report = {
        "policy": "strict_theorem_begin_span_aux_heading_lattice_v1",
        "status": "passed" if units else "no_units",
        "blocks_in_ir": len(theorem_ir.blocks),
        "units": len(units),
        "candidate_count": sum(len(value) for value in variants.values()),
        "accepted": accepted_blocks,
        "rejected": rejected,
        "source_span_scope": "begin_command_and_optional_title_only",
        "localization": "synctex_metadata_only",
        "generation_sources": ["latex_source", "compiler_aux"],
        "pdf_text_used": False,
    }
    return units, variants, rejected, report


def apply_display_equation_tail_ir(
    units: Sequence[stable.SourceUnit],
    blocks: Sequence[page_gt.SourceBlock],
    aux_label_numbers: Mapping[str, Any],
) -> tuple[
    list[stable.SourceUnit],
    dict[str, tuple[tuple[str, str], ...]],
    dict[str, Any],
]:
    """Attach a bounded source/AUX equation-number lattice to safe displays.

    Only the single-number ``equation`` environment is admitted.  Multi-row
    environments and explicit ``\\tag`` remain unchanged because their number
    position is not a deterministic tail of the complete formula Markdown.
    """

    block_by_id = {
        f"source-block-{block.block_id}": block
        for block in blocks
        if block.kind == "display_math"
    }
    output: list[stable.SourceUnit] = []
    variants: dict[str, tuple[tuple[str, str], ...]] = {}
    resolutions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for unit in units:
        block = block_by_id.get(unit.paragraph_id)
        if unit.kind != "display_math" or block is None:
            output.append(unit)
            continue
        environment_match = re.match(
            r"\s*\\begin\s*\{(?P<environment>[^{}]+)\}", block.raw_latex
        )
        environment = (
            environment_match.group("environment")
            if environment_match is not None
            else None
        )
        skip_reason: str | None = None
        if environment != "equation":
            skip_reason = f"unsupported_tail_environment:{environment or 'unknown'}"
        elif re.search(r"\\(?:nonumber|notag)\b", block.raw_latex):
            skip_reason = "explicit_number_suppression"
        elif re.search(r"\\tag\*?\s*\{", block.raw_latex):
            skip_reason = "explicit_tag_remains_inside_formula_markdown"
        base_offset = _source_fragment_base_offset(
            block.source_file,
            tuple(range(block.start_line, block.end_line + 1)),
            block.raw_latex,
        )
        if skip_reason is None and base_offset is None:
            skip_reason = "exact_source_span_unavailable_or_ambiguous"
        if skip_reason is not None:
            skipped.append(
                {
                    "source_block_id": block.block_id,
                    "source_file": str(block.source_file),
                    "source_lines": [block.start_line, block.end_line],
                    "reason": skip_reason,
                    "generation_source": "latex_source",
                    "pdf_text_used": False,
                }
            )
            output.append(unit)
            continue
        assert base_offset is not None
        resolution: EquationTailResolution = resolve_display_equation_tail(
            block.raw_latex,
            unit.markdown,
            source_file=block.source_file,
            aux_label_numbers=aux_label_numbers,
            source_offset=base_offset,
            start_line=block.start_line,
        )
        resolutions.append(resolution.as_report())
        candidate_values = tuple(
            dict.fromkeys(
                (candidate.policy, candidate.markdown)
                for candidate in resolution.candidates
                if candidate.markdown.startswith(unit.markdown + "\n")
            )
        )
        if resolution.status != "accepted" or not candidate_values:
            output.append(unit)
            continue
        output.append(dataclasses.replace(unit, markdown=candidate_values[0][1]))
        variants[unit.unit_id] = candidate_values
    return output, variants, {
        "policy": "single_equation_unique_aux_tail_lattice_v1",
        "status": "passed" if variants else "no_units",
        "units_considered": sum(unit.kind == "display_math" for unit in units),
        "units_with_tail_candidates": len(variants),
        "candidate_count": sum(len(value) for value in variants.values()),
        "resolutions": resolutions,
        "skipped": skipped,
        "formula_markdown_preserved_before_tail": True,
        "generation_sources": ["latex_source", "compiler_aux"],
        "pdf_text_used": False,
    }


def parse_aux_table_numbers(
    aux_paths: Iterable[Path],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Read compiler table numbers keyed by the source-derived caption."""

    values: dict[str, list[str]] = collections.defaultdict(list)
    parsed = rejected = 0
    for aux_path in aux_paths:
        if not aux_path.is_file():
            continue
        for line in aux_path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = line.find(r"\contentsline ")
            if marker < 0:
                continue
            cursor = marker + len(r"\contentsline ")
            groups: list[str] = []
            for _ in range(2):
                while cursor < len(line) and line[cursor].isspace():
                    cursor += 1
                group = page_gt.extract_balanced(line, cursor)
                if group is None:
                    break
                groups.append(group[0])
                cursor = group[1]
            if len(groups) != 2 or groups[0].strip() != "table":
                continue
            content = groups[1]
            number_marker = re.search(r"\\numberline\s*\{", content)
            if number_marker is None:
                rejected += 1
                continue
            number_group = page_gt.extract_balanced(content, number_marker.end() - 1)
            if number_group is None:
                rejected += 1
                continue
            number = page_gt.normalize_space(page_gt.latex_to_plain(number_group[0]))
            caption_source = content[number_group[1] :]
            caption_source = re.sub(r"\\ignorespaces\b", "", caption_source)
            caption = page_gt.normalize_space(page_gt.latex_to_plain(caption_source))
            if not number or not caption:
                rejected += 1
                continue
            values[caption.casefold()].append(number)
            parsed += 1
    return dict(values), {"entries_parsed": parsed, "entries_rejected": rejected}


def balanced_delimiter_end(
    value: str, start: int, opening: str, closing: str
) -> int | None:
    if start >= len(value) or value[start] != opening:
        return None
    depth = 0
    cursor = start
    while cursor < len(value):
        if value[cursor] == "\\":
            cursor += 2
            continue
        if value[cursor] == opening:
            depth += 1
        elif value[cursor] == closing:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def structural_payload_offset(window: str, begin_end: int, environment: str) -> int:
    """Skip environment arguments before inserting a zero-width locator.

    Inserting immediately after ``\\begin{longtable}`` places a PDF literal
    inside the mandatory column preamble.  Similar argument-bearing
    environments need their preamble consumed before any instrumentation.
    """

    cursor = begin_end
    while cursor < len(window) and window[cursor].isspace():
        cursor += 1
    while cursor < len(window) and window[cursor] == "[":
        end = balanced_delimiter_end(window, cursor, "[", "]")
        if end is None:
            raise ValueError(f"unbalanced optional argument for {environment}")
        cursor = end
        while cursor < len(window) and window[cursor].isspace():
            cursor += 1
    mandatory_groups = {
        "tabular": 1,
        "longtable": 1,
        "tabular*": 2,
        "tabularx": 2,
        "tabulary": 2,
        "alignat": 1,
        "alignat*": 1,
    }.get(environment, 0)
    for _ in range(mandatory_groups):
        if cursor >= len(window) or window[cursor] != "{":
            raise ValueError(f"missing mandatory environment argument for {environment}")
        end = balanced_delimiter_end(window, cursor, "{", "}")
        if end is None:
            raise ValueError(f"unbalanced mandatory argument for {environment}")
        cursor = end
        while cursor < len(window) and window[cursor].isspace():
            cursor += 1
    return cursor


def instrument_structural_ranges(
    source: str,
    units: Sequence[stable.SourceUnit],
    engine: str,
) -> str:
    """Color structural environments inside their existing TeX groups."""

    lines = source.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    edits: list[tuple[int, str, int]] = []
    for unit in units:
        left = starts[unit.start_line - 1]
        right = starts[unit.end_line] if unit.end_line < len(starts) else len(source)
        window = source[left:right]
        begin = re.search(r"\\begin\s*\{(?P<env>[^{}]+)\}", window)
        if begin is None:
            raise ValueError(f"structural begin environment unavailable: {unit.unit_id}")
        environment = begin.group("env")
        endings = list(re.finditer(r"\\end\s*\{" + re.escape(environment) + r"\}", window))
        if not endings:
            raise ValueError(f"structural end environment unavailable: {unit.unit_id}")
        end = endings[-1]
        if unit.kind == "table":
            caption = re.search(r"\\caption\*?(?:\s*\[[^\]]*\])?\s*\{", window)
            if caption is None:
                raise ValueError(f"table caption locator unavailable: {unit.unit_id}")
            caption_end = balanced_delimiter_end(window, caption.end() - 1, "{", "}")
            if caption_end is None:
                raise ValueError(f"unbalanced table caption: {unit.unit_id}")
            edits.append(
                (
                    left + caption.end(),
                    scoped_color_prefix(unit.rgb, engine),
                    0,
                )
            )
            edits.append(
                (
                    left + caption_end - 1,
                    scoped_color_suffix(engine),
                    1,
                )
            )
            continue
        payload = structural_payload_offset(window, begin.end(), environment)
        edits.append((left + payload, scoped_color_prefix(unit.rgb, engine), 0))
        edits.append((left + end.start(), scoped_color_suffix(engine), 1))
    rendered = source
    for offset, value, priority in sorted(edits, key=lambda item: (item[0], item[2]), reverse=True):
        rendered = rendered[:offset] + value + rendered[offset:]
    return rendered


def instrument_atom_ranges(
    source: str,
    probes: Sequence[stable.SourceProbe],
    locators: Mapping[str, AtomLocator],
    engine: str,
    *,
    blocked_unit_ids: Iterable[str] = (),
) -> str:
    blocked = frozenset(blocked_unit_ids)
    safe_probes = [probe for probe in probes if probe.unit_id not in blocked]
    edits: list[tuple[int, int, str]] = []
    first_by_unit: dict[str, str] = {}
    for probe in sorted(safe_probes, key=lambda item: (item.unit_id, item.ordinal)):
        first_by_unit.setdefault(probe.unit_id, probe.probe_id)
    for probe in safe_probes:
        locator = locators.get(probe.probe_id)
        if locator is None:
            continue
        prefix = scoped_color_prefix(probe.rgb, engine)
        if first_by_unit[probe.unit_id] == probe.probe_id:
            prefix = "\\leavevmode" + prefix
        edits.append((locator.source_start, 1, prefix))
        edits.append((locator.source_end, 0, scoped_color_suffix(engine)))
    rendered = source
    for offset, priority, value in sorted(edits, key=lambda item: (item[0], item[1]), reverse=True):
        rendered = rendered[:offset] + value + rendered[offset:]
    return rendered


def instrument_shadow_tree(
    clean_root: Path,
    shadow_root: Path,
    units: Sequence[stable.SourceUnit],
    probes: Sequence[stable.SourceProbe],
    atom_locators: Mapping[str, AtomLocator],
    engine: str,
    *,
    macro_expansion_mismatches: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    structural_ids = {unit.unit_id for unit in units if unit.kind in STRUCTURAL_KINDS}
    preexisting_metadata_only_ids = {
        unit.unit_id
        for unit in units
        if unit.kind in {"external_verbatim", "theorem_heading"}
        or (
            unit.kind == "heading"
            and (unit.start_line != unit.end_line or not unit.source_command)
        )
    }
    mismatch_map = macro_expansion_mismatches or {}
    source_indexes = {
        source_file: build_source_syntactic_index(source_file)
        for source_file in {unit.source_file.resolve() for unit in units}
    }
    safety_rows = [
        _unit_instrumentation_safety(
            unit,
            source_index=source_indexes.get(unit.source_file.resolve()),
            macro_expansion_mismatches=mismatch_map,
        )
        for unit in units
        if unit.unit_id not in structural_ids | preexisting_metadata_only_ids
    ]
    syntactically_blocked_ids = {
        str(row["unit_id"])
        for row in safety_rows
        if row["status"] == "metadata_only"
    }
    metadata_only_ids = preexisting_metadata_only_ids | syntactically_blocked_ids
    atom_ids = {
        probe.unit_id
        for probe in probes
        if probe.localization_mode in ATOM_LOCALIZATION_MODES
    } - metadata_only_ids
    list_ids = {
        unit.unit_id
        for unit in units
        if unit.kind in {"itemize_item", "enumerate_item", "description_item"}
    } - metadata_only_ids
    by_file: dict[Path, list[stable.SourceUnit]] = collections.defaultdict(list)
    probes_by_file: dict[Path, list[stable.SourceProbe]] = collections.defaultdict(list)
    for unit in units:
        by_file[unit.source_file.resolve()].append(unit)
    for probe in probes:
        probes_by_file[probe.source_file.resolve()].append(probe)
    for clean_path, file_units in by_file.items():
        relative = clean_path.relative_to(clean_root.resolve())
        target = shadow_root / relative
        source = target.read_text(encoding="utf-8", errors="replace")
        regular_units = [
            unit
            for unit in file_units
            if unit.unit_id
            not in structural_ids | atom_ids | metadata_only_ids | list_ids
        ]
        # The stable instrumenter has a conservative list branch for
        # itemize/enumerate.  Description Markdown differs only in its source
        # label; use the same zero-width list payload locator without changing
        # the SourceUnit kind or its GT.
        list_units = [
            dataclasses.replace(unit, kind="itemize_item")
            if unit.kind == "description_item"
            else unit
            for unit in file_units
            if unit.unit_id in list_ids
        ]
        file_probes = probes_by_file[clean_path]
        regular_probes = [
            probe
            for probe in file_probes
            if probe.unit_id not in structural_ids | atom_ids | metadata_only_ids
        ]
        # Atom offsets refer to the untouched clean file, so apply their
        # descending absolute edits first.  Subsequent legacy edits are
        # line-local and the unit overlap gate guarantees they do not share a
        # physical line with an atom-instrumented unit.
        source = instrument_atom_ranges(
            source,
            [probe for probe in file_probes if probe.unit_id in atom_ids],
            atom_locators,
            engine,
            blocked_unit_ids=metadata_only_ids,
        )
        source = stable.instrument_source_file(
            source,
            [*regular_units, *list_units],
            engine,
            regular_probes,
        )
        source = instrument_structural_ranges(
            source,
            [unit for unit in file_units if unit.unit_id in structural_ids],
            engine,
        )
        atomic_write_text(target, source)
    rejected_rows = [
        row for row in safety_rows if row["status"] == "metadata_only"
    ]
    reason_counts = collections.Counter(
        reason for row in rejected_rows for reason in row["reasons"]
    )
    return {
        "policy": "source_syntactic_executable_color_gate_v1",
        "units_total": len(units),
        "units_checked": len(safety_rows),
        "executable_color_units": len(units) - len(metadata_only_ids),
        "metadata_only_units": len(metadata_only_ids),
        "preexisting_metadata_only_units": len(preexisting_metadata_only_ids),
        "syntactic_gate_units_rejected": sum(
            any(
                reason != "macro_expanded_markdown_original_atom_mismatch"
                for reason in row["reasons"]
            )
            for row in rejected_rows
        ),
        "macro_expansion_mismatch_units_rejected": sum(
            "macro_expanded_markdown_original_atom_mismatch" in row["reasons"]
            for row in rejected_rows
        ),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "rejected_unit_provenance": rejected_rows,
        "metadata_only_unit_ids": sorted(metadata_only_ids),
        "localization_fallback": "synctex_clean_only",
        "executable_color_inserted_for_rejected_units": False,
        "source_provenance_preserved": True,
        "pdf_text_used": False,
    }


def glyph_components(chars: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    """Retain disconnected baseline runs instead of one paragraph union box."""

    ordered = sorted(
        chars,
        key=lambda char: (
            round(float(char["top"]), 2),
            float(char["x0"]),
            float(char["bottom"]),
        ),
    )
    groups: list[list[Mapping[str, Any]]] = []
    for char in ordered:
        if not groups:
            groups.append([char])
            continue
        previous = groups[-1][-1]
        same_baseline = abs(float(char["top"]) - float(previous["top"])) <= 2.0
        gap = float(char["x0"]) - float(previous["x1"])
        max_gap = max(6.0, 1.75 * float(char.get("size") or 8.0))
        if same_baseline and gap <= max_gap:
            groups[-1].append(char)
        else:
            groups.append([char])
    return [list(union_bbox([[c["x0"], c["top"], c["x1"], c["bottom"]] for c in group])) for group in groups]


def extract_color_runs(
    colored_pdf: Path,
    probes: Sequence[stable.SourceProbe],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    probe_by_rgb = {probe.rgb: probe for probe in probes}
    if len(probe_by_rgb) != len(probes):
        raise ValueError("v2 probe RGB values are not unique")
    observed: dict[str, dict[int, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    with pdfplumber.open(colored_pdf) as document:
        for page_number, page in enumerate(document.pages, start=1):
            matched = 0
            for char in page.chars:
                rgb = color_pilot.normalize_pdf_rgb(char.get("non_stroking_color"))
                probe = probe_by_rgb.get(rgb)
                if probe is None:
                    continue
                observed[probe.probe_id][page_number].append(char)
                matched += 1
            print(
                f"[color_page] page={page_number}/{len(document.pages)} "
                f"matched_characters={matched}",
                flush=True,
            )
    rows: dict[str, list[dict[str, Any]]] = {}
    mapped = multi_page = components_total = 0
    for probe in probes:
        pages: list[dict[str, Any]] = []
        for page_number, chars in sorted(observed.get(probe.probe_id, {}).items()):
            components = glyph_components(chars)
            components_total += len(components)
            pages.append(
                {
                    "page_number": page_number,
                    "bbox_points": list(union_bbox(components)),
                    "components": components,
                    "characters": len(chars),
                }
            )
        rows[probe.probe_id] = pages
        mapped += bool(pages)
        multi_page += len(pages) > 1
    return rows, {
        "probes_total": len(probes),
        "probes_mapped": mapped,
        "probes_unmapped": len(probes) - mapped,
        "probes_spanning_multiple_pages": multi_page,
        "glyph_components": components_total,
        "coverage": round(mapped / max(1, len(probes)), 8),
    }


def logical_text_stream(value: str) -> str:
    """Canonical verifier-order stream with layout-only line hyphens removed."""

    return stable.exact_visible_character_stream(value, markdown=False).replace(
        stable.OPTIONAL_LINE_END_HYPHEN, ""
    )


def compare_pdf_logical_invariance(
    clean_pdf: Path,
    colored_pdf: Path,
) -> dict[str, Any]:
    """Compare page content/order rather than Poppler glyph tokenization."""

    pages: list[dict[str, Any]] = []
    with pdfplumber.open(clean_pdf) as clean_document, pdfplumber.open(
        colored_pdf
    ) as colored_document:
        page_count_equal = len(clean_document.pages) == len(colored_document.pages)
        for page_number, (clean_page, colored_page) in enumerate(
            zip(clean_document.pages, colored_document.pages), start=1
        ):
            clean_text, clean_layout = stable.pdf_verifier_text(clean_page)
            colored_text, colored_layout = stable.pdf_verifier_text(colored_page)
            clean_stream = logical_text_stream(clean_text)
            colored_stream = logical_text_stream(colored_text)
            pages.append(
                {
                    "page_number": page_number,
                    "logical_content_and_order_equal": clean_stream == colored_stream,
                    "clean_layout": clean_layout,
                    "colored_layout": colored_layout,
                    "clean_stream_sha256": hashlib.sha256(
                        clean_stream.encode("utf-8")
                    ).hexdigest(),
                    "colored_stream_sha256": hashlib.sha256(
                        colored_stream.encode("utf-8")
                    ).hexdigest(),
                }
            )
    return {
        "page_count_equal": page_count_equal,
        "all_pages_equal": page_count_equal
        and all(page["logical_content_and_order_equal"] for page in pages),
        "pages": pages,
    }


def coarse_lane(bbox: Sequence[float], page_width: float) -> str:
    midpoint = page_width / 2.0
    tolerance = page_width * 0.025
    if float(bbox[2]) <= midpoint + tolerance:
        return "left"
    if float(bbox[0]) >= midpoint - tolerance:
        return "right"
    return "full"


def fragment_component_lane(
    fragment: LocatedFragment,
    page_width: float,
) -> str:
    """Classify a source fragment from its disconnected rendered line boxes.

    A paragraph that fills one IEEE column has a union bbox spanning much of
    the page when it continues from the bottom of the left column to the top
    of the right column.  Treating that union as a full-width block destroys
    reading order.  Individual glyph-line components retain the distinction:
    no rendered line crosses the gutter, even though the paragraph union does.
    """

    lanes = {
        coarse_lane(component, page_width)
        for component in fragment_line_components(fragment)
    }
    if "full" in lanes:
        return "full"
    if lanes == {"left", "right"}:
        return "bridge"
    if lanes == {"left"}:
        return "left"
    if lanes == {"right"}:
        return "right"
    return "unknown"


def fragment_line_components(
    fragment: LocatedFragment,
) -> tuple[tuple[float, float, float, float], ...]:
    """Merge word/glyph components on one baseline before lane inference."""

    components = sorted(
        (fragment.components or (fragment.bbox,)),
        key=lambda value: (float(value[1]), float(value[0]), float(value[3])),
    )
    lines: list[list[tuple[float, float, float, float]]] = []
    for raw in components:
        component = tuple(float(value) for value in raw)
        if not lines or abs(component[1] - lines[-1][0][1]) > 3.0:
            lines.append([component])
        else:
            lines[-1].append(component)
    return tuple(union_bbox(line) for line in lines)


def _lane_top(fragment: LocatedFragment, lane: str, page_width: float) -> float:
    values = [
        float(component[1])
        for component in fragment_line_components(fragment)
        if coarse_lane(component, page_width) == lane
    ]
    return min(values) if values else float(fragment.bbox[1])


def _order_column_segment(
    fragments: Sequence[LocatedFragment],
    page_width: float,
) -> list[LocatedFragment]:
    """Read a two-column segment left column first, then right column."""

    classified = [
        (fragment, fragment_component_lane(fragment, page_width))
        for fragment in fragments
    ]
    left = [
        fragment
        for fragment, lane in classified
        if lane in {"left", "bridge", "unknown"}
    ]
    right = [fragment for fragment, lane in classified if lane == "right"]
    left.sort(
        key=lambda item: (
            _lane_top(item, "left", page_width),
            0 if item.kind == "heading" else 1,
            item.source_start_line,
            float(item.bbox[0]),
            item.source_ordinal,
            item.fragment_id,
        )
    )
    right.sort(
        key=lambda item: (
            _lane_top(item, "right", page_width),
            0 if item.kind == "heading" else 1,
            item.source_start_line,
            float(item.bbox[0]),
            item.source_ordinal,
            item.fragment_id,
        )
    )
    return [*left, *right]


def component_lane_reading_order(
    fragments: Sequence[LocatedFragment],
    page_width: float,
) -> tuple[list[LocatedFragment], dict[str, Any]]:
    """Order full-width bands and true column lines without using PDF text.

    Only source-colored glyph geometry is consulted.  Actual full-width lines
    split the page into vertical bands.  Within each remaining band, all left
    column fragments precede all right column fragments.  A source paragraph
    crossing the column break is a single ``bridge`` fragment and is emitted
    once at the end of its left-column position.
    """

    lane_by_id = {
        fragment.fragment_id: fragment_component_lane(fragment, page_width)
        for fragment in fragments
    }
    full = sorted(
        (fragment for fragment in fragments if lane_by_id[fragment.fragment_id] == "full"),
        key=lambda item: (float(item.bbox[1]), float(item.bbox[0]), item.source_ordinal),
    )
    remaining = [
        fragment for fragment in fragments if lane_by_id[fragment.fragment_id] != "full"
    ]
    ordered: list[LocatedFragment] = []
    cursor_top = -math.inf
    tolerance = 3.0
    for spanning in full:
        before = [
            fragment
            for fragment in remaining
            if float(fragment.bbox[3]) <= float(spanning.bbox[1]) + tolerance
            and float(fragment.bbox[1]) >= cursor_top - tolerance
        ]
        if before:
            ordered.extend(_order_column_segment(before, page_width))
            selected = {fragment.fragment_id for fragment in before}
            remaining = [
                fragment for fragment in remaining if fragment.fragment_id not in selected
            ]
        ordered.append(spanning)
        cursor_top = max(cursor_top, float(spanning.bbox[3]))
    if remaining:
        ordered.extend(_order_column_segment(remaining, page_width))

    # Defensive de-duplication: a bridge participates in lane evidence for
    # both columns but must contribute its Markdown exactly once.
    unique: list[LocatedFragment] = []
    seen: set[str] = set()
    for fragment in ordered:
        if fragment.fragment_id not in seen:
            seen.add(fragment.fragment_id)
            unique.append(fragment)
    counts = collections.Counter(lane_by_id.values())
    has_columns = bool(counts["left"] or counts["bridge"]) and bool(
        counts["right"] or counts["bridge"]
    )
    if has_columns and counts["full"]:
        layout_bucket = "mixed_full_two_column"
    elif has_columns:
        layout_bucket = "two_column"
    elif counts["full"]:
        layout_bucket = "single_column"
    else:
        layout_bucket = "other"
    return unique, {
        "lane_counts": dict(sorted(counts.items())),
        "layout_bucket": layout_bucket,
    }


def split_probe_run_by_lane(
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, list[dict[str, Any]]],
    page_width: float,
) -> list[list[stable.SourceProbe]]:
    if not probes:
        return []
    probe_boxes: list[tuple[stable.SourceProbe, Sequence[float]]] = []
    for probe in probes:
        pages = rows.get(probe.probe_id, [])
        if len(pages) != 1:
            raise ValueError(f"probe does not map to exactly one page: {probe.probe_id}")
        probe_boxes.append((probe, pages[0]["bbox_points"]))

    # A token bbox cannot reveal its text lane: words near the left and right
    # edge of one ordinary full-width line would look like two columns.  First
    # aggregate consecutive source probes by visual baseline, then classify
    # the complete line box.  This preserves an actual left->right column
    # transition while avoiding token-level lane oscillation.
    line_groups: list[list[tuple[stable.SourceProbe, Sequence[float]]]] = []
    for probe, bbox in probe_boxes:
        if (
            not line_groups
            or abs(float(bbox[1]) - float(line_groups[-1][0][1][1])) > 3.0
        ):
            line_groups.append([(probe, bbox)])
        else:
            line_groups[-1].append((probe, bbox))
    line_lanes: list[str] = []
    for line in line_groups:
        bbox = union_bbox([value for _, value in line])
        line_lanes.append(coarse_lane(bbox, page_width))
    # The final short line of a full-width paragraph often lies entirely in
    # the left half.  If any line in this source run establishes a full-width
    # lane, the whole contiguous run is a single-column fragment.
    if "full" in line_lanes:
        line_lanes = ["full"] * len(line_lanes)
    probe_lane: dict[str, str] = {}
    for line, lane in zip(line_groups, line_lanes):
        for probe, _ in line:
            probe_lane[probe.probe_id] = lane
    output: list[list[stable.SourceProbe]] = []
    for probe in probes:
        lane = probe_lane[probe.probe_id]
        previous_lane = probe_lane[output[-1][-1].probe_id] if output else None
        if not output or lane != previous_lane:
            output.append([probe])
        else:
            output[-1].append(probe)
    return output


def build_fragments_for_shadow(
    units: Sequence[stable.SourceUnit],
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, list[dict[str, Any]]],
    unit_atoms: Mapping[str, tuple[SourceAtom, ...]],
    atom_locators: Mapping[str, AtomLocator],
    page_widths: Mapping[int, float],
    external_blocks: Mapping[str, ExternalVerbatimBlock] | None = None,
    structural_markdown_candidates: Mapping[
        str, tuple[tuple[str, str], ...]
    ] | None = None,
) -> tuple[dict[int, list[LocatedFragment]], dict[int, set[str]], dict[str, int]]:
    probes_by_unit: dict[str, list[stable.SourceProbe]] = collections.defaultdict(list)
    for probe in probes:
        probes_by_unit[probe.unit_id].append(probe)
    output: dict[int, list[LocatedFragment]] = collections.defaultdict(list)
    reasons: dict[int, set[str]] = collections.defaultdict(set)
    counts: collections.Counter[str] = collections.Counter()
    for unit_index, unit in enumerate(units):
        values = sorted(probes_by_unit[unit.unit_id], key=lambda item: item.ordinal)
        if not values:
            counts["units_without_probes"] += 1
            continue
        mode = values[0].localization_mode
        if any(probe.localization_mode != mode for probe in values):
            raise ValueError(f"mixed probe modes for {unit.unit_id}")
        if mode == "whole":
            pages = rows.get(values[0].probe_id, [])
            if len(pages) != 1:
                for page in pages:
                    reasons[int(page["page_number"])].add("whole_unit_not_single_page")
                counts["whole_unit_not_single_page"] += 1
                continue
            page = pages[0]
            page_number = int(page["page_number"])
            output[page_number].append(
                LocatedFragment(
                    fragment_id=f"{unit.unit_id}-whole",
                    unit_id=unit.unit_id,
                    paragraph_id=unit.paragraph_id,
                    kind=unit.kind,
                    markdown=unit.markdown,
                    probe_ids=(values[0].probe_id,),
                    source_file=unit.source_file,
                    source_start_line=unit.start_line,
                    source_ordinal=unit_index * 1_000_000,
                    page_number=page_number,
                    bbox=tuple(float(value) for value in page["bbox_points"]),
                    components=tuple(
                        tuple(float(value) for value in box)
                        for box in page.get("components", [page["bbox_points"]])
                    ),
                    structural_markdown_candidates=(
                        (structural_markdown_candidates or {}).get(
                            unit.unit_id, ()
                        )
                    ),
                )
            )
            counts["whole_mapped"] += 1
            continue
        if mode not in {
            "plain_word",
            EXTERNAL_VERBATIM_LOCALIZATION_MODE,
            *ATOM_LOCALIZATION_MODES,
        }:
            counts["unknown_mode"] += 1
            continue
        # ``page_widths`` is intentionally bounded by ``--max-pages`` in the
        # caller.  Color/Synctex alignment, however, is extracted from the
        # complete compiled PDF and can therefore contain probes on pages
        # beyond that bound.  Do not let those truncated probes reach the
        # lane split below (which needs a page width).  Whole-unit probes are
        # kept on their existing path because they never perform a lane
        # lookup; the page loop will discard out-of-range whole fragments.
        bounded_values: list[stable.SourceProbe] = []
        truncated_probe_count = 0
        for probe in values:
            probe_pages = rows.get(probe.probe_id, [])
            if len(probe_pages) == 1:
                probe_page_number = int(probe_pages[0]["page_number"])
                if probe_page_number not in page_widths:
                    truncated_probe_count += 1
                    continue
            bounded_values.append(probe)
        if truncated_probe_count:
            counts["truncated_probes"] += truncated_probe_count
        values = bounded_values
        if not values:
            counts[f"{mode}_truncated"] += 1
            continue
        page_sequence: list[int] = []
        invalid_pages: set[int] = set()
        for probe in values:
            pages = rows.get(probe.probe_id, [])
            if len(pages) != 1:
                invalid_pages.update(int(page["page_number"]) for page in pages)
                continue
            page_sequence.append(int(pages[0]["page_number"]))
        if len(page_sequence) != len(values):
            for page_number in invalid_pages:
                reasons[page_number].add(f"{mode}_probe_incomplete")
            counts[f"{mode}_incomplete"] += 1
            continue
        if any(right < left for left, right in zip(page_sequence, page_sequence[1:])):
            for page_number in set(page_sequence):
                reasons[page_number].add(f"{mode}_page_order_conflict")
            counts[f"{mode}_page_order_conflict"] += 1
            continue
        cursor = 0
        while cursor < len(values):
            page_number = page_sequence[cursor]
            end = cursor + 1
            while end < len(values) and page_sequence[end] == page_number:
                end += 1
            page_values = values[cursor:end]
            for lane_index, selected in enumerate(
                split_probe_run_by_lane(page_values, rows, page_widths[page_number])
            ):
                selected_pages = [rows[probe.probe_id][0] for probe in selected]
                boxes = [page["bbox_points"] for page in selected_pages]
                components = [
                    component
                    for page in selected_pages
                    for component in page.get("components", [page["bbox_points"]])
                ]
                if mode in ATOM_LOCALIZATION_MODES:
                    atom_ordinals = [atom_locators[probe.probe_id].atom_ordinal for probe in selected]
                    atoms = unit_atoms[unit.unit_id]
                    first_ordinal = min(atom_ordinals)
                    last_ordinal = max(atom_ordinals)
                    if mode == "source_wrapper_atom":
                        localized_ordinals = [
                            atom_locators[probe.probe_id].atom_ordinal
                            for probe in values
                        ]
                        if first_ordinal == min(localized_ordinals):
                            first_ordinal = 0
                        if last_ordinal == max(localized_ordinals):
                            last_ordinal = len(atoms) - 1
                    markdown = reconstruct_page_markdown(
                        atoms,
                        range(first_ordinal, last_ordinal + 1),
                    ).strip()
                elif mode == EXTERNAL_VERBATIM_LOCALIZATION_MODE:
                    block = (external_blocks or {}).get(unit.unit_id)
                    if block is None:
                        reasons[page_number].add("external_verbatim_block_missing")
                        continue
                    first_ordinal = min(probe.ordinal for probe in selected)
                    last_ordinal = max(probe.ordinal for probe in selected)
                    markdown = render_fenced_code(
                        block.records[first_ordinal - 1 : last_ordinal]
                    ).strip()
                else:
                    markdown = "".join(probe.markdown_fragment for probe in selected).strip()
                if not markdown:
                    reasons[page_number].add("empty_source_fragment")
                    continue
                output[page_number].append(
                    LocatedFragment(
                        fragment_id=(
                            f"{unit.unit_id}-{mode}-{selected[0].ordinal:05d}-"
                            f"{selected[-1].ordinal:05d}-lane-{lane_index:02d}"
                        ),
                        unit_id=unit.unit_id,
                        paragraph_id=unit.paragraph_id,
                        kind=unit.kind,
                        markdown=markdown,
                        probe_ids=tuple(probe.probe_id for probe in selected),
                        source_file=unit.source_file,
                        source_start_line=unit.start_line,
                        source_ordinal=unit_index * 1_000_000 + selected[0].ordinal,
                        page_number=page_number,
                        bbox=union_bbox(boxes),
                        components=tuple(tuple(float(value) for value in box) for box in components),
                    )
                )
            cursor = end
        counts[f"{mode}_mapped"] += 1
    return dict(output), reasons, dict(sorted(counts.items()))


def compose_markdown(fragments: Sequence[LocatedFragment]) -> str:
    output = ""
    previous: LocatedFragment | None = None
    for fragment in fragments:
        if not output:
            output = fragment.markdown
        elif previous is not None and previous.paragraph_id == fragment.paragraph_id:
            output = output.rstrip() + " " + fragment.markdown.lstrip()
        else:
            output = output.rstrip() + "\n\n" + fragment.markdown.lstrip()
        previous = fragment
    return output.strip() + ("\n" if output.strip() else "")


def heading_markdown_variant(
    markdown: str,
    *,
    uppercase: bool,
    trailing_colon: bool,
    terminal_label_dot: bool = False,
) -> str:
    match = re.fullmatch(r"(?P<hashes>#+)\s+(?P<body>.+)", markdown.strip(), re.DOTALL)
    if match is None:
        return markdown
    body = match.group("body").strip()
    first, separator, remainder = body.partition(" ")
    label_match = re.fullmatch(
        r"(?P<stem>(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z]))(?P<terminal>[.)]?)",
        first,
    )
    if separator and label_match is not None:
        label = first
        title = remainder
    else:
        label = ""
        title = body
    # Some classes serialize the source-derived numeric section label with a
    # terminal dot even though compiler metadata exposes only its stem (for
    # example ``1`` or ``3.1``).  Keep both finite source-derived spellings in
    # the lattice.  Existing visible punctuation is immutable: IEEE labels
    # such as ``I.``, ``A.``, and ``1)`` never become ``I..``/``1).``.
    if (
        terminal_label_dot
        and label
        and label_match is not None
        and label_match.group("terminal") == ""
        and re.fullmatch(r"\d+(?:\.\d+)*", label_match.group("stem"))
    ):
        label += "."
    # Changing the case of LaTeX math/control syntax would no longer be a
    # source-preserving serialization.  Such headings keep their source case.
    if uppercase and not re.search(r"[$\\<>]", title):
        title = title.upper()
    if trailing_colon and title and title[-1] not in ".:;!?":
        title += ":"
    return f"{match.group('hashes')} " + ((label + " ") if label else "") + title


def compose_markdown_policy(
    fragments: Sequence[LocatedFragment],
    heading_policies: Mapping[int, tuple[bool, bool, bool]],
    enumerate_parenthesis: bool,
    table_policy: tuple[bool, bool, bool, bool],
) -> str:
    rendered: list[LocatedFragment] = []
    for fragment in fragments:
        markdown = fragment.markdown
        if fragment.kind == "heading":
            match = re.match(r"^(#+)\s", markdown)
            if match is not None:
                uppercase, trailing_colon, terminal_label_dot = heading_policies.get(
                    len(match.group(1)), (False, False, False)
                )
                markdown = heading_markdown_variant(
                    markdown,
                    uppercase=uppercase,
                    trailing_colon=trailing_colon,
                    terminal_label_dot=terminal_label_dot,
                )
        if enumerate_parenthesis and fragment.kind == "enumerate_item":
            markdown = re.sub(r"^(\s*\d+)\.\s+", r"\1) ", markdown, count=1)
        if fragment.kind == "table":
            markdown = table_markdown_variant(markdown, *table_policy)
        rendered.append(dataclasses.replace(fragment, markdown=markdown))
    return compose_markdown(rendered)


def table_markdown_variant(
    markdown: str,
    uppercase_label: bool,
    uppercase_caption: bool,
    caption_period: bool,
    strip_redundant_strong: bool,
) -> str:
    parts = markdown.strip().split("\n\n", 2)
    if len(parts) < 3 or not re.fullmatch(r"Table\s+\S+", parts[0]):
        value = markdown
    else:
        label, caption, table_html = parts
        if uppercase_label:
            label = label.upper()
        if uppercase_caption:
            caption = caption.upper()
        if caption_period and caption and caption[-1] not in ".!?":
            caption += "."
        value = "\n\n".join((label, caption, table_html))
    if strip_redundant_strong:
        value = re.sub(r"</?strong>", "", value, flags=re.IGNORECASE)
    return value


def source_serialization_candidates(
    fragments: Sequence[LocatedFragment],
    *,
    max_candidates: int | None = None,
) -> list[tuple[str, str]]:
    """Finite source-derived typography lattice; PDF only verifies candidates."""

    levels = sorted(
        {
            len(match.group(1))
            for fragment in fragments
            if fragment.kind == "heading"
            and (match := re.match(r"^(#+)\s", fragment.markdown)) is not None
        }
    )
    list_options = (False, True) if any(
        fragment.kind == "enumerate_item" for fragment in fragments
    ) else (False,)
    table_options = (
        tuple(itertools.product((False, True), repeat=4))
        if any(fragment.kind == "table" for fragment in fragments)
        else ((False, False, False, False),)
    )
    heading_options: list[tuple[tuple[bool, bool, bool], ...]] = []
    for level in levels:
        has_unpunctuated_numeric_label = any(
            fragment.kind == "heading"
            and (match := re.match(r"^(#+)\s+(\d+(?:\.\d+)*)\s+", fragment.markdown))
            is not None
            and len(match.group(1)) == level
            for fragment in fragments
        )
        base = tuple(
            (uppercase, trailing_colon, False)
            for uppercase, trailing_colon in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            )
        )
        heading_options.append(
            base
            + (
                tuple(
                    (uppercase, trailing_colon, True)
                    for uppercase, trailing_colon, _ in base
                )
                if has_unpunctuated_numeric_label
                else ()
            )
        )
    structural_indices = [
        index
        for index, fragment in enumerate(fragments)
        if fragment.structural_markdown_candidates
    ]
    structural_options = [
        fragments[index].structural_markdown_candidates
        for index in structural_indices
    ]
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for structural_settings in itertools.product(*structural_options):
        selected_fragments = list(fragments)
        structural_policies: list[str] = []
        for fragment_index, (policy_name, markdown_variant) in zip(
            structural_indices, structural_settings
        ):
            selected_fragments[fragment_index] = dataclasses.replace(
                selected_fragments[fragment_index], markdown=markdown_variant
            )
            structural_policies.append(
                f"{selected_fragments[fragment_index].unit_id}:{policy_name}"
            )
        for settings in itertools.product(*heading_options):
            policies = dict(zip(levels, settings))
            for enumerate_parenthesis in list_options:
                for table_policy in table_options:
                    markdown = compose_markdown_policy(
                        selected_fragments,
                        policies,
                        enumerate_parenthesis,
                        table_policy,
                    )
                    if markdown in seen:
                        continue
                    seen.add(markdown)
                    if max_candidates is not None and len(values) >= max_candidates:
                        raise ValueError(
                            "source_serialization_candidate_limit_exceeded:"
                            f">{max_candidates}"
                        )
                    policy = ",".join(
                        f"h{level}:{'upper' if upper else 'source'}"
                        f"{'_colon' if colon else ''}"
                        f"{'_label_dot' if label_dot else ''}"
                        for level, (upper, colon, label_dot) in sorted(policies.items())
                    ) or "source"
                    policy += ";enum=paren" if enumerate_parenthesis else ";enum=dot"
                    if any(table_policy):
                        policy += ";table=" + "_".join(
                            name
                            for name, enabled in zip(
                                ("label_upper", "caption_upper", "period", "plain_th"),
                                table_policy,
                            )
                            if enabled
                        )
                    else:
                        policy += ";table=source"
                    policy += (
                        ";struct=" + ",".join(structural_policies)
                        if structural_policies
                        else ";struct=source"
                    )
                    values.append((policy, markdown))
    return values


def markdown_neutral_whitespace_cuts(value: str) -> tuple[list[int], str | None]:
    """Return whitespace cuts outside source-derived Markdown inline syntax."""

    cuts: list[int] = []
    strong = False
    emphasis = False
    math_mode = False
    code_fence = 0
    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character == "\\" and cursor + 1 < len(value):
            cursor += 2
            continue
        if character == "`" and not math_mode:
            end = cursor
            while end < len(value) and value[end] == "`":
                end += 1
            width = end - cursor
            if code_fence == 0:
                code_fence = width
            elif code_fence == width:
                code_fence = 0
            cursor = end
            continue
        if code_fence:
            cursor += 1
            continue
        if character == "$":
            if cursor + 1 < len(value) and value[cursor + 1] == "$":
                return [], "display_math_whitespace_cuts_are_unsupported"
            math_mode = not math_mode
            cursor += 1
            continue
        if not math_mode and character == "*":
            end = cursor
            while end < len(value) and value[end] == "*":
                end += 1
            width = end - cursor
            if width > 3:
                return [], "unsupported_markdown_star_run"
            if width >= 2:
                strong = not strong
            if width % 2:
                emphasis = not emphasis
            cursor = end
            continue
        if (
            character.isspace()
            and not strong
            and not emphasis
            and not math_mode
            and not code_fence
        ):
            end = cursor + 1
            while end < len(value) and value[end].isspace():
                end += 1
            if end < len(value):
                cuts.append(end)
            cursor = end
            continue
        cursor += 1
    if strong or emphasis or math_mode or code_fence:
        return [], "unbalanced_markdown_at_whitespace_frontier"
    return cuts, None


def safe_whole_unit_suffixes(
    unit: stable.SourceUnit,
) -> tuple[list[dict[str, Any]], str | None]:
    """Enumerate source/Markdown cuts without observing page text."""

    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    atoms = build_source_atoms(unit.raw_latex)
    if (
        atoms
        and not any(atom.kind == "opaque" for atom in atoms)
        and atoms_to_markdown(atoms).strip() == unit.markdown.strip()
    ):
        for atom in atoms:
            if atom.ordinal == 0 or atom.kind in {"whitespace", "opaque"}:
                continue
            suffix = reconstruct_page_markdown(
                atoms, range(atom.ordinal, len(atoms))
            ).strip()
            if suffix and suffix != unit.markdown.strip() and suffix not in seen:
                seen.add(suffix)
                values.append(
                    {
                        "suffix": suffix,
                        "cut_kind": "source_atom",
                        "cut_index": atom.ordinal,
                        "source_character_offset": atom.source_start,
                    }
                )
    whitespace_cuts, reason = markdown_neutral_whitespace_cuts(unit.markdown)
    if reason is not None:
        return [], reason
    for cut in whitespace_cuts:
        suffix = unit.markdown[cut:].strip()
        if suffix and suffix != unit.markdown.strip() and suffix not in seen:
            seen.add(suffix)
            values.append(
                {
                    "suffix": suffix,
                    "cut_kind": "markdown_neutral_whitespace",
                    "cut_index": cut,
                }
            )
    if len(values) > MAX_WHOLE_FRONTIER_CUTS:
        return [], (
            f"whole_frontier_cut_limit_exceeded:{len(values)}>"
            f"{MAX_WHOLE_FRONTIER_CUTS}"
        )
    return values, None


def _probe_page_rows(
    probe: stable.SourceProbe,
    rows: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return list(rows.get(probe.probe_id, []))


def enumerate_leading_frontier_variants(
    page_number: int,
    fragments: Sequence[LocatedFragment],
    *,
    units: Sequence[stable.SourceUnit],
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze strict source-derived page-leading continuation candidates.

    A colored probe's glyph count and page number are geometry/provenance, not
    transcription.  They may establish that a source token/unit was only
    partly localized on the preceding page, but candidate text is always a
    suffix enumerated from the already source-derived probe or unit Markdown.
    """

    if page_number <= 1 or not fragments:
        return [], {
            "status": "passed",
            "policy_version": FRONTIER_POLICY_VERSION,
            "carrier_count": 0,
            "variant_count": 0,
        }
    probes_by_unit: dict[str, list[stable.SourceProbe]] = collections.defaultdict(list)
    for probe in probes:
        probes_by_unit[probe.unit_id].append(probe)
    unit_by_id = {unit.unit_id: unit for unit in units}
    unit_index = {unit.unit_id: index for index, unit in enumerate(units)}
    current_unit_ids = {fragment.unit_id for fragment in fragments}
    if not current_unit_ids.issubset(unit_index):
        return [], {
            "status": "failed",
            "policy_version": FRONTIER_POLICY_VERSION,
            "reason": "fragment_unit_missing_from_source_order",
        }
    global_top = min(fragment.bbox[1] for fragment in fragments)
    carriers: list[dict[str, Any]] = []

    plain_carriers: list[tuple[stable.SourceProbe, stable.SourceProbe, dict[str, Any]]] = []
    for unit_id in sorted(current_unit_ids, key=unit_index.__getitem__):
        values = sorted(probes_by_unit.get(unit_id, []), key=lambda probe: probe.ordinal)
        if not values or values[0].localization_mode != "plain_word":
            continue
        current_values = [
            probe
            for probe in values
            if len(_probe_page_rows(probe, rows)) == 1
            and int(_probe_page_rows(probe, rows)[0]["page_number"]) == page_number
        ]
        if not current_values:
            continue
        first = current_values[0]
        first_row = _probe_page_rows(first, rows)[0]
        if float(first_row["bbox_points"][1]) > global_top + 2.0:
            continue
        if first.ordinal <= 1:
            continue
        previous = next(
            (probe for probe in values if probe.ordinal == first.ordinal - 1), None
        )
        if previous is None:
            continue
        previous_rows = _probe_page_rows(previous, rows)
        if (
            len(previous_rows) != 1
            or int(previous_rows[0]["page_number"]) != page_number - 1
        ):
            continue
        token = previous.markdown_fragment.strip()
        expected = len(
            stable.exact_visible_character_stream(token, markdown=True).replace(
                stable.OPTIONAL_LINE_END_HYPHEN, ""
            )
        )
        observed = int(previous_rows[0].get("characters") or 0)
        if expected <= 1 or observed >= expected:
            continue
        plain_carriers.append((previous, first, previous_rows[0]))
    if len(plain_carriers) > MAX_BOUNDARY_CARRIERS:
        return [], {
            "status": "failed",
            "policy_version": FRONTIER_POLICY_VERSION,
            "reason": f"ambiguous_plain_word_carriers={len(plain_carriers)}",
        }
    if plain_carriers:
        previous, first, previous_row = plain_carriers[0]
        token = previous.markdown_fragment.strip()
        if re.fullmatch(r"[^\W\d_]+", token, flags=re.UNICODE) is None:
            return [], {
                "status": "failed",
                "policy_version": FRONTIER_POLICY_VERSION,
                "reason": "plain_word_carrier_is_not_a_safe_alphabetic_token",
            }
        suffixes = [
            {"suffix": token[cut:], "cut_index": cut}
            for cut in range(1, len(token))
            if token[cut:]
        ]
        if len(suffixes) > MAX_TOKEN_FRONTIER_CUTS:
            return [], {
                "status": "failed",
                "policy_version": FRONTIER_POLICY_VERSION,
                "reason": (
                    f"token_frontier_cut_limit_exceeded:{len(suffixes)}>"
                    f"{MAX_TOKEN_FRONTIER_CUTS}"
                ),
            }
        carriers.append(
            {
                "kind": "plain_word_token_suffix",
                "unit_id": previous.unit_id,
                "paragraph_id": previous.paragraph_id,
                "probe_id": previous.probe_id,
                "next_probe_id": first.probe_id,
                "source_token_span": list(previous.token_span) if previous.token_span else None,
                "source_token": token,
                "observed_colored_characters": int(previous_row.get("characters") or 0),
                "source_visible_characters": len(token),
                "join": "same_paragraph_space",
                "suffixes": suffixes,
            }
        )

    earliest_index = min(unit_index[unit_id] for unit_id in current_unit_ids)
    if earliest_index > 0:
        previous_unit = units[earliest_index - 1]
        previous_probes = probes_by_unit.get(previous_unit.unit_id, [])
        if (
            len(previous_probes) == 1
            and previous_probes[0].localization_mode == "whole"
        ):
            whole_rows = _probe_page_rows(previous_probes[0], rows)
            if (
                len(whole_rows) == 1
                and int(whole_rows[0]["page_number"]) == page_number - 1
            ):
                expected = len(
                    stable.exact_visible_character_stream(
                        previous_unit.markdown, markdown=True
                    ).replace(stable.OPTIONAL_LINE_END_HYPHEN, "")
                )
                observed = int(whole_rows[0].get("characters") or 0)
                if 0 < observed < expected:
                    suffixes, reason = safe_whole_unit_suffixes(previous_unit)
                    if reason is not None:
                        return [], {
                            "status": "failed",
                            "policy_version": FRONTIER_POLICY_VERSION,
                            "reason": reason,
                            "carrier_unit_id": previous_unit.unit_id,
                        }
                    carriers.append(
                        {
                            "kind": "whole_unit_suffix",
                            "unit_id": previous_unit.unit_id,
                            "paragraph_id": previous_unit.paragraph_id,
                            "probe_id": previous_probes[0].probe_id,
                            "observed_colored_characters": observed,
                            "source_visible_characters": expected,
                            "join": "new_paragraph",
                            "suffixes": suffixes,
                        }
                    )

    if len(carriers) > MAX_BOUNDARY_CARRIERS:
        return [], {
            "status": "failed",
            "policy_version": FRONTIER_POLICY_VERSION,
            "reason": f"ambiguous_boundary_carriers={len(carriers)}",
            "carrier_kinds": [carrier["kind"] for carrier in carriers],
        }
    if not carriers:
        return [], {
            "status": "passed",
            "policy_version": FRONTIER_POLICY_VERSION,
            "carrier_count": 0,
            "variant_count": 0,
        }
    carrier = carriers[0]
    variants: list[dict[str, Any]] = []
    for suffix in carrier["suffixes"]:
        text = str(suffix["suffix"])
        provenance = {
            key: value
            for key, value in carrier.items()
            if key not in {"suffixes", "source_token"}
        }
        provenance.update(
            {
                key: value for key, value in suffix.items() if key != "suffix"
            }
        )
        provenance["suffix_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        provenance["suffix_characters"] = len(text)
        variants.append(
            {
                "text": text,
                "join": carrier["join"],
                "provenance": provenance,
            }
        )
    return variants, {
        "status": "passed",
        "policy_version": FRONTIER_POLICY_VERSION,
        "carrier_count": 1,
        "carrier_kind": carrier["kind"],
        "carrier_unit_id": carrier["unit_id"],
        "variant_count": len(variants),
    }


def apply_leading_frontier(markdown: str, variant: Mapping[str, Any] | None) -> str:
    if variant is None:
        return markdown
    prefix = str(variant["text"]).strip()
    if not prefix:
        return markdown
    separator = " " if variant["join"] == "same_paragraph_space" else "\n\n"
    return prefix + separator + markdown.lstrip()


def compact_frontier_provenance(
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if provenance is None:
        return None
    return dict(provenance)


def project_math_visible_braces(markdown: str) -> tuple[str, int]:
    """Project escaped visible math braces without changing stored GT."""

    replacements = 0

    def project(match: re.Match[str]) -> str:
        nonlocal replacements
        body = match.group("body")
        left = body.count(r"\{")
        right = body.count(r"\}")
        replacements += left + right
        body = body.replace(r"\{", "｛").replace(r"\}", "｝")
        return match.group("open") + body + match.group("close")

    inline_dollar = re.compile(
        r"(?P<open>(?<!\\)\$(?!\$))(?P<body>.*?)(?P<close>(?<!\\)\$)",
        flags=re.DOTALL,
    )
    inline_paren = re.compile(
        r"(?P<open>\\\()(?P<body>.*?)(?P<close>\\\))",
        flags=re.DOTALL,
    )
    projected = inline_dollar.sub(project, markdown)
    projected = inline_paren.sub(project, projected)
    return projected, replacements


def project_fenced_code_info_strings(markdown: str) -> tuple[str, int]:
    """Hide non-printed fenced-code info strings from the PDF verifier.

    The stored Markdown remains untouched.  We retain both fences and the
    exact source-derived code body so the stable Markdown projector continues
    to protect literal Markdown punctuation inside the code block.
    """

    opening = re.compile(
        r"(?m)^(?P<indent> {0,3})(?P<fence>`{3,})(?P<info>[^\r\n]*)$"
    )
    edits: list[tuple[int, int, str]] = []
    cursor = 0
    while match := opening.search(markdown, cursor):
        fence = match.group("fence")
        closing = re.compile(
            rf"(?m)^ {{0,3}}{re.escape(fence)}[ \t]*$"
        ).search(markdown, match.end())
        if closing is None:
            break
        if match.group("info").strip():
            edits.append(
                (
                    match.start(),
                    match.end(),
                    match.group("indent") + fence,
                )
            )
        cursor = closing.end()
    projected = markdown
    for start, end, replacement in reversed(edits):
        projected = projected[:start] + replacement + projected[end:]
    return projected, len(edits)


def experimental_verifier_result(markdown: str, pdf_text: str) -> dict[str, Any]:
    visible_flow = project_source_verifier_visible_flow(markdown)
    projected, replacements = project_math_visible_braces(
        visible_flow.projected_markdown
    )
    projected, fenced_info_strings = project_fenced_code_info_strings(projected)
    result = stable.verifier_result(projected, pdf_text)
    result["experimental_projection"] = {
        "version": VERIFIER_PROJECTION_VERSION,
        "source_visible_flow_version": MATH_VISIBLE_FLOW_PROJECTION_VERSION,
        "source_visible_flow": dict(visible_flow.provenance),
        "math_visible_brace_replacements": replacements,
        "fenced_code_info_strings_hidden": fenced_info_strings,
        "ground_truth_changed": False,
        "pdf_text_used_for_ground_truth": False,
    }
    return result


def candidate_orders(
    fragments: Sequence[LocatedFragment],
    *,
    page_width: float,
    page_height: float,
) -> tuple[list[tuple[str, list[LocatedFragment]]], dict[str, Any]]:
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    graph = build_layout_graph(
        [
            LayoutFragment(
                fragment_id=fragment.fragment_id,
                source_ordinal=fragment.source_ordinal,
                bbox=fragment.bbox,
                page_number=fragment.page_number,
                kind=fragment.kind,
                source_group_id=fragment.paragraph_id,
            )
            for fragment in fragments
        ],
        page_width=page_width,
        page_height=page_height,
    )
    candidates: list[tuple[str, list[LocatedFragment]]] = []
    if graph.ordered_fragment_ids:
        candidates.append(("banded_layout_graph", [by_id[value] for value in graph.ordered_fragment_ids]))
    component_order, component_report = component_lane_reading_order(
        fragments, page_width
    )
    if component_order:
        candidates.append(("component_lane_reading_order", component_order))
    candidates.extend(
        [
            (
                "source_ordinal",
                sorted(fragments, key=lambda item: (item.source_ordinal, item.fragment_id)),
            ),
            (
                "geometry_yx",
                sorted(fragments, key=lambda item: (item.bbox[1], item.bbox[0], item.source_ordinal)),
            ),
        ]
    )
    unique: list[tuple[str, list[LocatedFragment]]] = []
    seen: set[tuple[str, ...]] = set()
    for name, ordered in candidates:
        identity = tuple(item.fragment_id for item in ordered)
        if identity not in seen:
            seen.add(identity)
            unique.append((name, ordered))
    graph_report = graph.as_dict()
    graph_report["union_bbox_layout_bucket"] = (
        "mixed_full_two_column"
        if graph.layout_kind == "two_column" and any(band.full_span for band in graph.bands)
        else graph.layout_kind
    )
    graph_report["component_lane_model"] = component_report
    graph_report["layout_bucket"] = component_report["layout_bucket"]
    return unique, graph_report


def freeze_page_source_candidates(
    fragments: Sequence[LocatedFragment],
    *,
    page_width: float,
    page_height: float,
    frontier_variants: Sequence[Mapping[str, Any]] = (),
    frontier_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the bounded source candidate set before PDF text is read."""

    orders, layout = candidate_orders(
        fragments, page_width=page_width, page_height=page_height
    )
    report = dict(
        frontier_report
        or {
            "status": "passed",
            "policy_version": FRONTIER_POLICY_VERSION,
            "carrier_count": 0,
            "variant_count": 0,
        }
    )
    if report.get("status") != "passed":
        return {
            "status": "failed",
            "reason": str(report.get("reason") or "frontier_enumeration_failed"),
            "candidates": [],
            "candidate_count": 0,
            "layout": layout,
            "frontier": report,
        }
    variants: list[Mapping[str, Any] | None] = [None, *frontier_variants]
    candidates: list[dict[str, Any]] = []
    for name, ordered in orders:
        remaining_serializations = (
            MAX_PAGE_SOURCE_CANDIDATES - len(candidates)
        ) // len(variants)
        if remaining_serializations <= 0:
            return {
                "status": "failed",
                "reason": (
                    "page_candidate_limit_exceeded:>"
                    f"{MAX_PAGE_SOURCE_CANDIDATES}"
                ),
                "candidates": [],
                "candidate_count": 0,
                "layout": layout,
                "frontier": report,
            }
        try:
            serializations = source_serialization_candidates(
                ordered,
                max_candidates=remaining_serializations,
            )
        except ValueError as error:
            if not str(error).startswith(
                "source_serialization_candidate_limit_exceeded:"
            ):
                raise
            return {
                "status": "failed",
                "reason": (
                    "page_candidate_limit_exceeded:>"
                    f"{MAX_PAGE_SOURCE_CANDIDATES}"
                ),
                "candidates": [],
                "candidate_count": 0,
                "layout": layout,
                "frontier": report,
            }
        for serialization_policy, base_markdown in serializations:
            for variant in variants:
                markdown = apply_leading_frontier(base_markdown, variant)
                markdown_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
                fragment_ids = [fragment.fragment_id for fragment in ordered]
                order_identity = json.dumps(
                    {"order": name, "fragment_ids": fragment_ids},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                order_id = "order-" + hashlib.sha256(
                    order_identity.encode("utf-8")
                ).hexdigest()[:12]
                provenance = (
                    compact_frontier_provenance(variant.get("provenance"))
                    if variant is not None
                    else None
                )
                frontier_identity = json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                frontier_id = (
                    "frontier-none"
                    if provenance is None
                    else "frontier-"
                    + hashlib.sha256(frontier_identity.encode("utf-8")).hexdigest()[:12]
                )
                identity = json.dumps(
                    {
                        "order_id": order_id,
                        "serialization_policy": serialization_policy,
                        "frontier_id": frontier_id,
                        "markdown_sha256": markdown_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                candidates.append(
                    {
                        "candidate_id": "sfv2c-"
                        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                        "order": name,
                        "order_id": order_id,
                        "serialization_policy": serialization_policy,
                        "fragment_ids": fragment_ids,
                        "frontier": provenance,
                        "frontier_id": frontier_id,
                        "markdown": markdown,
                        "markdown_sha256": markdown_sha256,
                    }
                )
                if len(candidates) > MAX_PAGE_SOURCE_CANDIDATES:
                    return {
                        "status": "failed",
                        "reason": (
                            f"page_candidate_limit_exceeded:{len(candidates)}>"
                            f"{MAX_PAGE_SOURCE_CANDIDATES}"
                        ),
                        "candidates": [],
                        "candidate_count": 0,
                        "layout": layout,
                        "frontier": report,
                    }
    return {
        "status": "passed",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "layout": layout,
        "frontier": report,
    }


def compact_verifier_summary(verifier: Mapping[str, Any]) -> dict[str, Any]:
    projection = verifier.get("experimental_projection")
    return {
        "status": verifier.get("status"),
        "match_mode": verifier.get("match_mode"),
        "exact_character_stream": bool(
            verifier.get("exact_ordered_character_stream_match")
        ),
        "math_brace_projection_count": int(
            projection.get("math_visible_brace_replacements") or 0
        )
        if isinstance(projection, Mapping)
        else 0,
    }


def verify_frozen_page_candidates(
    frozen: Mapping[str, Any],
    pdf_text: str,
    *,
    pdf_page: Any | None = None,
) -> dict[str, Any]:
    """Use PDF text only to exact-select an already frozen source set."""

    if frozen.get("status") != "passed":
        return {
            "status": "rejected",
            "reason": str(frozen.get("reason") or "source_candidate_freeze_failed"),
            "attempts": [],
            "candidate_count": int(frozen.get("candidate_count") or 0),
            "layout": frozen.get("layout", {}),
            "frontier": frozen.get("frontier", {}),
        }
    attempts: list[dict[str, Any]] = []
    exact_by_markdown: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    order_catalog: dict[str, dict[str, Any]] = {}
    policy_catalog: dict[str, str] = {}
    frontier_catalog: dict[str, dict[str, Any] | None] = {}
    folio_projection = None
    if pdf_page is not None:
        folio_projection = project_bottom_margin_folio(
            "",
            pdf_text,
            page=pdf_page,
            stream_projector=logical_text_stream,
        )
    for candidate in frozen.get("candidates", []):
        markdown = str(candidate["markdown"])
        verifier = experimental_verifier_result(markdown, pdf_text)
        verifier_text_projection: dict[str, Any] | None = None
        if (
            verifier["status"] != "passed"
            and folio_projection is not None
            and folio_projection.projection_applied
        ):
            projected = experimental_verifier_result(
                markdown, folio_projection.verifier_text
            )
            if projected["status"] == "passed":
                verifier = projected
                verified_projection = project_bottom_margin_folio(
                    markdown,
                    pdf_text,
                    page=pdf_page,
                    stream_projector=logical_text_stream,
                )
                verifier_text_projection = dict(verified_projection.provenance)
                verifier["pdf_layout_projection"] = verifier_text_projection
        order_catalog.setdefault(
            str(candidate["order_id"]),
            {
                "order": candidate["order"],
                "fragment_ids": candidate["fragment_ids"],
            },
        )
        frontier_catalog.setdefault(
            str(candidate["frontier_id"]), candidate["frontier"]
        )
        policy = str(candidate["serialization_policy"])
        policy_id = "policy-" + hashlib.sha256(policy.encode("utf-8")).hexdigest()[:10]
        policy_catalog.setdefault(policy_id, policy)
        compact = {
            "candidate_id": candidate["candidate_id"],
            "order_id": candidate["order_id"],
            "policy_id": policy_id,
            "frontier_id": candidate["frontier_id"],
            "markdown_sha256": candidate["markdown_sha256"],
            "verifier": compact_verifier_summary(verifier),
        }
        if verifier_text_projection is not None:
            compact["pdf_layout_projection"] = {
                "version": verifier_text_projection.get("version"),
                "status": verifier_text_projection.get("status"),
                "reason": verifier_text_projection.get("reason"),
                "projection_applied": True,
            }
        attempts.append(compact)
        if verifier["status"] == "passed":
            exact_by_markdown[markdown].append(
                {**candidate, "verifier": verifier}
            )
    if len(exact_by_markdown) == 1:
        markdown, matching = next(iter(exact_by_markdown.items()))
        selected = matching[0]
        return {
            "status": "passed",
            "markdown": markdown,
            "selected_candidate_id": selected["candidate_id"],
            "selected_order": selected["order"],
            "selected_serialization_policy": selected["serialization_policy"],
            "selected_frontier": selected["frontier"],
            "fragment_ids": selected["fragment_ids"],
            "verifier": selected["verifier"],
            "attempts": attempts,
            "candidate_provenance": {
                "orders": order_catalog,
                "policies": policy_catalog,
                "frontiers": frontier_catalog,
                "verifier_projection_version": VERIFIER_PROJECTION_VERSION,
                "folio_projection_version": FOLIO_PROJECTION_VERSION,
            },
            "candidate_count": int(frozen.get("candidate_count") or 0),
            "layout": frozen.get("layout", {}),
            "frontier": frozen.get("frontier", {}),
        }
    return {
        "status": "rejected",
        "reason": "no_exact_candidate" if not exact_by_markdown else "ambiguous_exact_markdown",
        "attempts": attempts,
        "candidate_provenance": {
            "orders": order_catalog,
            "policies": policy_catalog,
            "frontiers": frontier_catalog,
            "verifier_projection_version": VERIFIER_PROJECTION_VERSION,
            "folio_projection_version": FOLIO_PROJECTION_VERSION,
        },
        "candidate_count": int(frozen.get("candidate_count") or 0),
        "layout": frozen.get("layout", {}),
        "frontier": frozen.get("frontier", {}),
    }


def choose_exact_page_candidate(
    fragments: Sequence[LocatedFragment],
    *,
    page_width: float,
    page_height: float,
    pdf_text: str,
) -> dict[str, Any]:
    frozen = freeze_page_source_candidates(
        fragments,
        page_width=page_width,
        page_height=page_height,
    )
    return verify_frozen_page_candidates(frozen, pdf_text)


def probe_json(
    probe: stable.SourceProbe,
    clean_root: Path,
    atom_locators: Mapping[str, AtomLocator],
) -> dict[str, Any]:
    value = probe.as_json(clean_root)
    locator = atom_locators.get(probe.probe_id)
    if locator is not None:
        value["source_character_span"] = [locator.source_start, locator.source_end]
        value["source_atom_ordinal"] = locator.atom_ordinal
    return value


def exact_pdf_shadow_gate(
    geometry: Mapping[str, Any],
    logical_invariance: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a shadow to be exactly layout- and text-equivalent to clean.

    The general geometry comparator deliberately allows a small tolerance for
    diagnostic experiments.  External-verbatim localization is stricter: its
    executable hook is trusted only when every projected glyph coordinate is
    byte-for-byte stable after the comparator's six-decimal projection, and
    the independent logical page stream is unchanged on every page.
    """

    def exact_zero(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) == 0.0
        )

    reasons: list[str] = []
    geometry_pages = list(geometry.get("pages") or ())
    logical_pages = list(logical_invariance.get("pages") or ())
    if geometry.get("status") != "passed":
        reasons.append("geometry_comparison_failed")
    if geometry.get("page_count_equal") is not True:
        reasons.append("geometry_page_count_changed")
    if geometry.get("character_text_equal") is not True:
        reasons.append("glyph_text_or_order_changed")
    if geometry.get("geometry_equal") is not True:
        reasons.append("glyph_geometry_changed")
    if not exact_zero(geometry.get("max_geometry_shift_points")):
        reasons.append("glyph_geometry_not_exact")
    if not geometry_pages or len(geometry_pages) != int(
        geometry.get("pages_compared") or 0
    ):
        reasons.append("geometry_pages_incomplete")
    if any(
        page.get("character_count_equal") is not True
        or page.get("character_text_equal") is not True
        or page.get("geometry_equal") is not True
        or not exact_zero(page.get("max_geometry_shift_points"))
        for page in geometry_pages
    ):
        reasons.append("page_glyph_geometry_not_exact")
    if logical_invariance.get("page_count_equal") is not True:
        reasons.append("logical_page_count_changed")
    if not logical_pages:
        reasons.append("logical_pages_incomplete")
    if logical_invariance.get("all_pages_equal") is not True or any(
        page.get("logical_content_and_order_equal") is not True
        for page in logical_pages
    ):
        reasons.append("logical_content_or_order_changed")
    unique_reasons = sorted(set(reasons))
    return {
        "status": "passed" if not unique_reasons else "failed",
        "reasons": unique_reasons,
        "requires_zero_geometry_shift": True,
        "max_geometry_shift_points": geometry.get("max_geometry_shift_points"),
        "logical_pages_compared": len(logical_pages),
    }


def best_synctex_shadow(
    shadows: Sequence[ShadowCandidate],
) -> ShadowCandidate | None:
    """Select the highest-coverage trusted SyncTeX row set."""

    priority = {"synctex_clean": 0, "synctex_atom_lines": 1}
    candidates = [shadow for shadow in shadows if shadow.shadow_id in priority]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda shadow: (
            float(shadow.color_summary.get("coverage") or 0.0),
            priority[shadow.shadow_id],
        ),
    )


def alignment_summary_for_rows(
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    locator: str,
) -> dict[str, Any]:
    mapped = multi_page = components_total = 0
    for probe in probes:
        pages = rows.get(probe.probe_id, ())
        mapped += bool(pages)
        multi_page += len(pages) > 1
        components_total += sum(
            len(page.get("components") or ()) for page in pages
        )
    total = len(probes)
    return {
        "locator": locator,
        "probes_total": total,
        "probes_mapped": mapped,
        "probes_unmapped": total - mapped,
        "probes_spanning_multiple_pages": multi_page,
        "glyph_components": components_total,
        "coverage": round(mapped / max(1, total), 8),
        "pdf_text_used": False,
    }


INVARIANT_UNIT_HYBRID_POLICY = "fail_closed_invariant_unit_geometry_merge_v1"
INVARIANT_UNIT_HYBRID_DONOR_PRIORITY = (
    "synctex_clean",
    "synctex_atom_lines",
)
INVARIANT_UNIT_HYBRID_REJECTION_EXAMPLES = 64


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_unit_geometry_merge_record(unit: stable.SourceUnit) -> dict[str, Any]:
    """Return a root-independent source-only identity for one unit.

    Paths are deliberately omitted from the digest: ``unit_id``, exact source
    line numbers, raw-source hash, and Markdown hash already identify the
    immutable source record while keeping a derived hybrid ID reproducible
    after moving the same project to another machine.
    """

    return {
        "unit_id": unit.unit_id,
        "paragraph_id": unit.paragraph_id,
        "kind": unit.kind,
        "source_lines": list(unit.source_lines),
        "raw_latex_sha256": hashlib.sha256(
            unit.raw_latex.encode("utf-8")
        ).hexdigest(),
        "markdown_sha256": hashlib.sha256(
            unit.markdown.encode("utf-8")
        ).hexdigest(),
        "source_command": unit.source_command,
    }


def _probe_geometry_merge_record(
    probe: stable.SourceProbe,
    locator: AtomLocator | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "probe_id": probe.probe_id,
        "unit_id": probe.unit_id,
        "paragraph_id": probe.paragraph_id,
        "kind": probe.kind,
        "source_file": str(probe.source_file.resolve()),
        "source_lines": list(probe.source_lines),
        "markdown_fragment_sha256": hashlib.sha256(
            probe.markdown_fragment.encode("utf-8")
        ).hexdigest(),
        "rgb": list(probe.rgb),
        "ordinal": probe.ordinal,
        "total": probe.total,
        "localization_mode": probe.localization_mode,
        "token_span": list(probe.token_span) if probe.token_span else None,
    }
    if locator is not None:
        value["atom_locator"] = {
            "source_start": locator.source_start,
            "source_end": locator.source_end,
            "atom_ordinal": locator.atom_ordinal,
        }
    return value


def _source_atom_geometry_merge_record(atom: SourceAtom) -> dict[str, Any]:
    return {
        "ordinal": atom.ordinal,
        "source_start": atom.source_start,
        "source_end": atom.source_end,
        "kind": atom.kind,
        "style_stack": list(atom.style_stack),
        "markdown_fragment_sha256": hashlib.sha256(
            atom.markdown_fragment.encode("utf-8")
        ).hexdigest(),
        "raw_source_sha256": hashlib.sha256(
            atom.raw_source.encode("utf-8")
        ).hexdigest(),
        "visible_text_sha256": hashlib.sha256(
            atom.visible_text.encode("utf-8")
        ).hexdigest(),
    }


def invariant_shadow_probe_schema(
    shadow: ShadowCandidate,
) -> tuple[dict[str, tuple[stable.SourceProbe, ...]], str, dict[str, str]]:
    """Freeze the exact probe schema used by a possible hybrid.

    A donor is accepted only when this complete ordered schema hash equals the
    base hash.  Matching probe IDs alone is intentionally insufficient.
    """

    grouped: dict[str, list[stable.SourceProbe]] = collections.defaultdict(list)
    seen_ids: set[str] = set()
    ordered_records: list[dict[str, Any]] = []
    for probe in shadow.probes:
        if probe.probe_id in seen_ids:
            raise ValueError(f"duplicate probe id: {probe.probe_id}")
        seen_ids.add(probe.probe_id)
        grouped[probe.unit_id].append(probe)
        record = _probe_geometry_merge_record(
            probe, shadow.atom_locators.get(probe.probe_id)
        )
        registered_mode = shadow.modes.get(probe.unit_id)
        if registered_mode != probe.localization_mode:
            raise ValueError(
                f"probe mode registry mismatch: {probe.probe_id}:"
                f"{registered_mode!r}!={probe.localization_mode!r}"
            )
        record["registered_mode"] = registered_mode
        ordered_records.append(record)
    grouped_frozen: dict[str, tuple[stable.SourceProbe, ...]] = {}
    unit_hashes: dict[str, str] = {}
    for unit_id, probes in grouped.items():
        values = tuple(sorted(probes, key=lambda item: item.ordinal))
        ordinals = [probe.ordinal for probe in values]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError(f"duplicate probe ordinal in unit: {unit_id}")
        grouped_frozen[unit_id] = values
        unit_hashes[unit_id] = _sha256_json(
            {
                "probes": [
                    {
                        **_probe_geometry_merge_record(
                            probe, shadow.atom_locators.get(probe.probe_id)
                        ),
                        "registered_mode": shadow.modes.get(probe.unit_id),
                    }
                    for probe in values
                ],
                "atoms": [
                    _source_atom_geometry_merge_record(atom)
                    for atom in shadow.unit_atoms.get(unit_id, ())
                ],
            }
        )
    whole_schema = {
        "probes": ordered_records,
        "unit_atoms": {
            unit_id: [
                _source_atom_geometry_merge_record(atom)
                for atom in shadow.unit_atoms.get(unit_id, ())
            ]
            for unit_id in sorted(grouped_frozen)
        },
    }
    return grouped_frozen, _sha256_json(whole_schema), unit_hashes


def _finite_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return None
    return bbox


def _unit_unique_geometry(
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    page_widths: Mapping[int, float],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate that observed rows form one unambiguous page/lane carrier."""

    observed: list[tuple[stable.SourceProbe, int, tuple[float, float, float, float]]] = []
    missing: list[str] = []
    for probe in probes:
        values = rows.get(probe.probe_id, ())
        if not values:
            missing.append(probe.probe_id)
            continue
        if len(values) != 1:
            return {
                "status": "rejected",
                "reason": "probe_row_not_unique",
                "probe_id": probe.probe_id,
                "row_count": len(values),
            }
        row = values[0]
        try:
            page_number = int(row["page_number"])
        except (KeyError, TypeError, ValueError):
            return {
                "status": "rejected",
                "reason": "invalid_row_schema",
                "probe_id": probe.probe_id,
            }
        bbox = _finite_bbox(row.get("bbox_points"))
        if page_number < 1 or bbox is None or page_number not in page_widths:
            return {
                "status": "rejected",
                "reason": "invalid_row_schema",
                "probe_id": probe.probe_id,
            }
        observed.append((probe, page_number, bbox))
    if require_complete and missing:
        return {
            "status": "incomplete",
            "reason": "probe_rows_incomplete",
            "missing_probe_ids": missing,
        }
    if not observed:
        return {
            "status": "empty",
            "reason": "probe_rows_empty",
            "missing_probe_ids": missing,
        }
    pages = {page_number for _probe, page_number, _bbox in observed}
    if len(pages) != 1:
        return {
            "status": "rejected",
            "reason": "unit_rows_span_multiple_pages",
            "page_numbers": sorted(pages),
        }
    page_number = next(iter(pages))
    lanes = {
        coarse_lane(bbox, page_widths[page_number])
        for _probe, _page_number, bbox in observed
    }
    if len(lanes) != 1:
        return {
            "status": "rejected",
            "reason": "unit_rows_span_multiple_lanes",
            "page_number": page_number,
            "lanes": sorted(lanes),
        }
    lane = next(iter(lanes))
    return {
        "status": "complete" if not missing else "partial",
        "page_number": page_number,
        "lane": lane,
        "observed_probe_ids": [probe.probe_id for probe, _page, _bbox in observed],
        "missing_probe_ids": missing,
    }


def _donor_page_bbox_rows(
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Copy only page membership and the union bbox from a trusted donor."""

    output: dict[str, list[dict[str, Any]]] = {}
    for probe in probes:
        row = rows[probe.probe_id][0]
        bbox = _finite_bbox(row.get("bbox_points"))
        assert bbox is not None  # validated by _unit_unique_geometry
        output[probe.probe_id] = [
            {
                "page_number": int(row["page_number"]),
                "bbox_points": list(bbox),
            }
        ]
    return output


def _copy_alignment_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Deep-copy row geometry without sharing mutable lists with a base."""

    output: dict[str, list[dict[str, Any]]] = {}
    for probe_id, values in rows.items():
        copied: list[dict[str, Any]] = []
        for row in values:
            value = dict(row)
            if isinstance(value.get("bbox_points"), Sequence):
                value["bbox_points"] = list(value["bbox_points"])
            if isinstance(value.get("components"), Sequence):
                value["components"] = [
                    list(component) for component in value["components"]
                ]
            if isinstance(value.get("source_lines"), Sequence):
                value["source_lines"] = list(value["source_lines"])
            copied.append(value)
        output[probe_id] = copied
    return output


def derive_invariant_unit_hybrid_shadows(
    shadows: Sequence[ShadowCandidate],
    units: Sequence[stable.SourceUnit],
    page_widths: Mapping[int, float],
) -> tuple[list[ShadowCandidate], dict[str, Any]]:
    """Derive fail-closed per-unit geometry hybrids from exact SyncTeX.

    The base shadow remains immutable.  For an empty/incomplete base unit, an
    exact clean-SyncTeX donor (then atom-line SyncTeX) may replace *every* row
    in that unit.  The donor contributes only page number and bbox; probes,
    atoms, Markdown, source ordering, and all GT material remain those of the
    base ``SourceUnit`` graph.  No PDF text is read here.
    """

    unit_by_id: dict[str, stable.SourceUnit] = {}
    duplicate_unit_ids: set[str] = set()
    for unit in units:
        if unit.unit_id in unit_by_id:
            duplicate_unit_ids.add(unit.unit_id)
        unit_by_id[unit.unit_id] = unit
    rejection_counts: collections.Counter[str] = collections.Counter()
    rejection_examples: list[dict[str, Any]] = []

    def reject(reason: str, **values: Any) -> None:
        rejection_counts[reason] += 1
        if len(rejection_examples) < INVARIANT_UNIT_HYBRID_REJECTION_EXAMPLES:
            rejection_examples.append({"reason": reason, **values})

    if duplicate_unit_ids:
        for unit_id in sorted(duplicate_unit_ids):
            reject("duplicate_source_unit_id", unit_id=unit_id)
        return [], {
            "status": "rejected",
            "policy": INVARIANT_UNIT_HYBRID_POLICY,
            "donor_priority": list(INVARIANT_UNIT_HYBRID_DONOR_PRIORITY),
            "hybrids": [],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rejection_examples": rejection_examples,
            "ground_truth_source": "SourceUnit",
            "donor_fields_used": ["page_number", "bbox_points"],
            "pdf_text_used": False,
        }

    shadow_by_id = {shadow.shadow_id: shadow for shadow in shadows}
    donor_catalog: list[dict[str, Any]] = []
    for donor_id in INVARIANT_UNIT_HYBRID_DONOR_PRIORITY:
        donor = shadow_by_id.get(donor_id)
        if donor is None:
            donor_catalog.append(
                {"shadow_id": donor_id, "status": "unavailable"}
            )
            continue
        gate = exact_pdf_shadow_gate(donor.geometry, donor.logical_invariance)
        try:
            grouped, schema_hash, unit_hashes = invariant_shadow_probe_schema(
                donor
            )
        except ValueError as error:
            reject(
                "donor_probe_schema_invalid",
                donor_shadow_id=donor_id,
                detail=str(error),
            )
            donor_catalog.append(
                {
                    "shadow_id": donor_id,
                    "status": "rejected",
                    "exact_invariance_gate": gate,
                    "reason": "donor_probe_schema_invalid",
                }
            )
            continue
        status = "eligible" if gate["status"] == "passed" else "rejected"
        if status == "rejected":
            reject(
                "donor_global_exact_invariance_failed",
                donor_shadow_id=donor_id,
                gate_reasons=gate["reasons"],
            )
        donor_catalog.append(
            {
                "shadow_id": donor_id,
                "status": status,
                "shadow": donor,
                "grouped": grouped,
                "schema_sha256": schema_hash,
                "unit_probe_schema_sha256": unit_hashes,
                "exact_invariance_gate": gate,
            }
        )

    eligible_donors = [
        value for value in donor_catalog if value.get("status") == "eligible"
    ]
    hybrids: list[ShadowCandidate] = []
    hybrid_reports: list[dict[str, Any]] = []
    donor_ids = set(INVARIANT_UNIT_HYBRID_DONOR_PRIORITY)
    for base in shadows:
        if base.shadow_id in donor_ids or base.shadow_id.startswith(
            "synctex_atom_external_color"
        ):
            continue
        try:
            base_grouped, base_schema_hash, base_unit_hashes = (
                invariant_shadow_probe_schema(base)
            )
        except ValueError as error:
            reject(
                "base_probe_schema_invalid",
                base_shadow_id=base.shadow_id,
                detail=str(error),
            )
            continue
        compatible_donors = [
            value
            for value in eligible_donors
            if value["schema_sha256"] == base_schema_hash
        ]
        if not compatible_donors:
            if eligible_donors:
                reject(
                    "complete_probe_schema_hash_mismatch",
                    base_shadow_id=base.shadow_id,
                    base_probe_schema_sha256=base_schema_hash,
                    donor_probe_schema_sha256={
                        value["shadow_id"]: value["schema_sha256"]
                        for value in eligible_donors
                    },
                )
            continue
        merged_rows = _copy_alignment_rows(base.color_rows)
        replacements: list[dict[str, Any]] = []
        donor_counts: collections.Counter[str] = collections.Counter()
        for unit_id, base_probes in base_grouped.items():
            unit = unit_by_id.get(unit_id)
            if unit is None:
                reject(
                    "probe_unit_missing_from_source_units",
                    base_shadow_id=base.shadow_id,
                    unit_id=unit_id,
                )
                continue
            base_state = _unit_unique_geometry(
                base_probes,
                base.color_rows,
                page_widths,
                require_complete=False,
            )
            if base_state["status"] == "complete":
                continue
            if base_state["status"] == "rejected":
                reject(
                    str(base_state["reason"]),
                    base_shadow_id=base.shadow_id,
                    unit_id=unit_id,
                    role="base",
                )
                continue
            if any(
                probe.localization_mode == EXTERNAL_VERBATIM_LOCALIZATION_MODE
                for probe in base_probes
            ):
                reject(
                    "external_verbatim_unit_forbidden",
                    base_shadow_id=base.shadow_id,
                    unit_id=unit_id,
                )
                continue

            selected_donor: dict[str, Any] | None = None
            selected_state: dict[str, Any] | None = None
            selected_probes: tuple[stable.SourceProbe, ...] | None = None
            for donor_value in compatible_donors:
                donor_id = str(donor_value["shadow_id"])
                donor_grouped = donor_value["grouped"]
                donor_probes = donor_grouped.get(unit_id)
                if donor_probes is None:
                    reject(
                        "donor_unit_probe_set_missing",
                        base_shadow_id=base.shadow_id,
                        donor_shadow_id=donor_id,
                        unit_id=unit_id,
                    )
                    continue
                donor_unit_hash = donor_value[
                    "unit_probe_schema_sha256"
                ].get(unit_id)
                if donor_unit_hash != base_unit_hashes.get(unit_id):
                    reject(
                        "unit_probe_schema_hash_mismatch",
                        base_shadow_id=base.shadow_id,
                        donor_shadow_id=donor_id,
                        unit_id=unit_id,
                    )
                    continue
                donor_state = _unit_unique_geometry(
                    donor_probes,
                    donor_value["shadow"].color_rows,
                    page_widths,
                    require_complete=True,
                )
                if donor_state["status"] == "complete":
                    selected_donor = donor_value
                    selected_state = donor_state
                    selected_probes = donor_probes
                    break
                reason = str(donor_state["reason"])
                reject(
                    reason,
                    base_shadow_id=base.shadow_id,
                    donor_shadow_id=donor_id,
                    unit_id=unit_id,
                    role="donor",
                )
                # Missing clean rows may be refined by atom-line SyncTeX.  A
                # contradictory row, page, lane, or schema is a hard unit
                # rejection and must never be hidden by trying another donor.
                if donor_state["status"] == "rejected":
                    selected_donor = None
                    selected_state = None
                    selected_probes = None
                    break
            if selected_donor is None or selected_state is None or selected_probes is None:
                continue
            if base_state["status"] == "partial" and (
                base_state.get("page_number") != selected_state["page_number"]
                or base_state.get("lane") != selected_state["lane"]
            ):
                reject(
                    "base_donor_page_or_lane_conflict",
                    base_shadow_id=base.shadow_id,
                    donor_shadow_id=selected_donor["shadow_id"],
                    unit_id=unit_id,
                    base_page_number=base_state.get("page_number"),
                    donor_page_number=selected_state["page_number"],
                    base_lane=base_state.get("lane"),
                    donor_lane=selected_state["lane"],
                )
                continue
            donor_rows = _donor_page_bbox_rows(
                selected_probes,
                selected_donor["shadow"].color_rows,
            )
            # Atomic unit replacement: even valid partial base rows are
            # discarded, preventing mixed locators within one semantic unit.
            for probe in base_probes:
                merged_rows[probe.probe_id] = donor_rows[probe.probe_id]
            donor_id = str(selected_donor["shadow_id"])
            donor_counts[donor_id] += 1
            source_unit_record = _source_unit_geometry_merge_record(unit)
            replacements.append(
                {
                    "unit_id": unit_id,
                    "source_unit_sha256": _sha256_json(source_unit_record),
                    "source_unit_markdown_sha256": source_unit_record[
                        "markdown_sha256"
                    ],
                    "probe_schema_sha256": base_unit_hashes[unit_id],
                    "probe_count": len(base_probes),
                    "base_row_state": base_state["status"],
                    "donor_shadow_id": donor_id,
                    "page_number": selected_state["page_number"],
                    "lane": selected_state["lane"],
                    "donor_geometry_sha256": _sha256_json(donor_rows),
                    "donor_fields_used": ["page_number", "bbox_points"],
                    "ground_truth_source": "SourceUnit",
                    "pdf_text_used": False,
                }
            )
        if not replacements:
            continue
        identity = {
            "policy": INVARIANT_UNIT_HYBRID_POLICY,
            "base_shadow_id": base.shadow_id,
            "base_probe_schema_sha256": base_schema_hash,
            "unit_donors": [
                [value["unit_id"], value["donor_shadow_id"]]
                for value in replacements
            ],
        }
        hybrid_id = (
            f"{base.shadow_id}__invariant_unit_hybrid_"
            f"{_sha256_json(identity)[:16]}"
        )
        summary = alignment_summary_for_rows(
            base.probes,
            merged_rows,
            locator="invariant_unit_hybrid_page_bbox",
        )
        summary.update(
            {
                "policy": INVARIANT_UNIT_HYBRID_POLICY,
                "base_shadow_id": base.shadow_id,
                "units_replaced": len(replacements),
                "donor_counts": dict(sorted(donor_counts.items())),
                "ground_truth_source": "SourceUnit",
                "donor_fields_used": ["page_number", "bbox_points"],
                "pdf_text_used": False,
            }
        )
        hybrids.append(
            ShadowCandidate(
                shadow_id=hybrid_id,
                probes=list(base.probes),
                atom_locators=dict(base.atom_locators),
                unit_atoms=dict(base.unit_atoms),
                modes=dict(base.modes),
                colored_pdf=base.colored_pdf,
                geometry=base.geometry,
                logical_invariance=base.logical_invariance,
                color_rows=merged_rows,
                color_summary=summary,
            )
        )
        hybrid_reports.append(
            {
                "shadow_id": hybrid_id,
                "base_shadow_id": base.shadow_id,
                "base_probe_schema_sha256": base_schema_hash,
                "alignment": summary,
                "unit_replacements": replacements,
                "hybrid_geometry_sha256": _sha256_json(merged_rows),
                "base_shadow_unchanged": True,
                "ground_truth_source": "SourceUnit",
                "donor_fields_used": ["page_number", "bbox_points"],
                "pdf_text_used": False,
            }
        )

    public_donors = [
        {
            key: value
            for key, value in donor.items()
            if key not in {"shadow", "grouped", "unit_probe_schema_sha256"}
        }
        for donor in donor_catalog
    ]
    return hybrids, {
        "status": "derived" if hybrids else "skipped",
        "policy": INVARIANT_UNIT_HYBRID_POLICY,
        "donor_priority": list(INVARIANT_UNIT_HYBRID_DONOR_PRIORITY),
        "donors": public_donors,
        "base_shadows_considered": [
            shadow.shadow_id
            for shadow in shadows
            if shadow.shadow_id not in donor_ids
            and not shadow.shadow_id.startswith("synctex_atom_external_color")
        ],
        "hybrids": hybrid_reports,
        "hybrids_created": len(hybrids),
        "units_replaced": sum(
            len(value["unit_replacements"]) for value in hybrid_reports
        ),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejection_examples": rejection_examples,
        "rejection_examples_truncated": sum(rejection_counts.values())
        > len(rejection_examples),
        "base_shadows_immutable": True,
        "ground_truth_source": "SourceUnit",
        "donor_fields_used": ["page_number", "bbox_points"],
        "pdf_text_used": False,
        "pdf_text_role_after_candidate_freeze": "exact_verifier_only",
    }


def suppress_untrusted_external_synctex_rows(
    probes: Sequence[stable.SourceProbe],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Prevent native SyncTeX from bypassing the external color gate."""

    external_ids = {
        probe.probe_id
        for probe in probes
        if probe.localization_mode == EXTERNAL_VERBATIM_LOCALIZATION_MODE
    }
    output: dict[str, list[dict[str, Any]]] = {}
    observed_but_untrusted = 0
    for probe in probes:
        pages = [dict(page) for page in rows.get(probe.probe_id, ())]
        if probe.probe_id in external_ids:
            observed_but_untrusted += bool(pages)
            pages = []
        output[probe.probe_id] = pages
    summary = alignment_summary_for_rows(
        probes,
        output,
        locator="synctex_clean_source_line_external_verbatim_suppressed",
    )
    summary.update(
        {
            "external_line_probes_suppressed": len(external_ids),
            "external_synctex_rows_observed_but_untrusted": observed_but_untrusted,
            "requires_external_color_gate": bool(external_ids),
        }
    )
    return output, summary


def merge_external_verbatim_alignment(
    probes: Sequence[stable.SourceProbe],
    base_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    external_color_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Replace external-line SyncTeX rows with exact color-derived rows.

    Empty external physical lines intentionally have no probe; their source
    records remain between adjacent visible records when fenced Markdown is
    reconstructed.  Every *visible* external line must have exactly one page
    observation, otherwise the entire merged shadow fails closed.
    """

    external_ids = {
        probe.probe_id
        for probe in probes
        if probe.localization_mode == EXTERNAL_VERBATIM_LOCALIZATION_MODE
    }
    if not external_ids:
        raise ValueError("no external verbatim line probes to merge")
    invalid_external: list[str] = []
    merged: dict[str, list[dict[str, Any]]] = {}
    for probe in probes:
        source_rows = (
            external_color_rows
            if probe.probe_id in external_ids
            else base_rows
        )
        pages = [dict(page) for page in source_rows.get(probe.probe_id, ())]
        if probe.probe_id in external_ids and len(pages) != 1:
            invalid_external.append(f"{probe.probe_id}:{len(pages)}_pages")
        merged[probe.probe_id] = pages
    if invalid_external:
        sample = ", ".join(invalid_external[:10])
        raise RuntimeError(
            "external verbatim color localization is incomplete or multi-page: "
            f"{sample}"
        )
    summary = alignment_summary_for_rows(
        probes,
        merged,
        locator="synctex_plus_external_verbatim_color",
    )
    summary.update({
        "external_line_probes": len(external_ids),
        "external_line_probes_mapped_exactly_once": len(external_ids),
        "external_rows_replace_synctex_rows": True,
    })
    return merged, summary


def main() -> int:
    args = parse_args()
    if args.max_pages <= 0 or args.dpi <= 0 or args.compile_timeout <= 0:
        raise ValueError("max-pages, dpi, and compile-timeout must be positive")
    if args.min_eligible_visible_characters < 1:
        raise ValueError("min eligible visible characters must be positive")
    stable_guard = assert_stable_files(REPO_ROOT)
    color_pilot.LATEXMK = args.latexmk.expanduser().absolute()
    page_gt.PDFTOPPM = args.pdftoppm.expanduser().resolve()
    for name, executable in (
        ("latexmk", color_pilot.LATEXMK),
        ("pdftoppm", page_gt.PDFTOPPM),
    ):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"required executable unavailable: {name}={executable}")
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"experimental paper output already exists: {output_dir}")
    if not (source_dir / args.main_tex).is_file():
        raise FileNotFoundError(source_dir / args.main_tex)
    started = time.monotonic()
    print(
        f"[start] contract={EXPERIMENTAL_CONTRACT} paper={args.paper_id} "
        f"source={source_dir} main={args.main_tex} output={output_dir}",
        flush=True,
    )
    output_dir.mkdir(parents=True)
    clean_root = output_dir / "source_clean"
    shutil.copytree(source_dir, clean_root)
    if args.drop_references:
        reference_report = color_pilot.strip_references_tree(clean_root)
    else:
        reference_report = {"status": "disabled", "residuals": []}
    if args.drop_figures:
        figure_report = stable.strip_ignored_figures_tree(clean_root)
        figure_policy = "drop_figures"
    else:
        figure_report = {"status": "disabled"}
        figure_policy = "keep_figures"
    main_path = clean_root / args.main_tex
    atomic_write_text(
        main_path,
        stable.inject_compile_support(
            main_path.read_text(encoding="utf-8", errors="replace")
        ),
    )
    clean_build = output_dir / "build_clean"
    clean_pdf = color_pilot.run_compile(
        source_root=clean_root,
        main_tex=args.main_tex,
        build_dir=clean_build,
        log_path=output_dir / "logs" / "clean.log",
        label="source-first-v2-clean",
        timeout_seconds=args.compile_timeout,
        engine=args.engine,
    )
    executed_project = stable.compiled_project_sources(clean_root, clean_build)
    executed = [
        path
        for path in executed_project
        if path.suffix.casefold() in {".tex", ".ltx"}
    ]
    if not executed:
        raise RuntimeError("clean compile exposed no executed project TeX sources")
    execution_ir = build_execution_ir(main_path, fls_sources=executed)
    safe_macro_registry = collect_safe_macros(executed)
    external_verbatim_ir = build_external_verbatim_ir(
        clean_root,
        executed,
        strict=False,
    )
    visible_wrapper_macros, visible_wrapper_definition_report = (
        collect_safe_visible_wrapper_macros(executed_project)
    )
    math_macros = page_gt.collect_simple_math_macros(clean_root)
    blocks = page_gt.parse_source_blocks(clean_root, math_macros)
    paragraphs = page_gt.parse_source_paragraphs(clean_root, blocks, executed)
    try:
        compiler_labels, compiler_heading_report = compiler_heading_labels(
            clean_root=clean_root,
            clean_pdf=clean_pdf,
            main_tex=args.main_tex,
            blocks=blocks,
            output_dir=output_dir,
            timeout_seconds=args.compile_timeout,
            engine=args.engine,
        )
    except Exception as error:  # noqa: BLE001 - metadata is an optional shadow
        compiler_labels = {}
        compiler_heading_report = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        print(
            f"[heading_metadata_done] status=failed error={type(error).__name__}:{error}",
            flush=True,
        )
    else:
        print(
            f"[heading_metadata_done] status={compiler_heading_report['status']} "
            f"labels={compiler_heading_report['labels_resolved']}/"
            f"{compiler_heading_report['records_requested']}",
            flush=True,
        )
    aux_paths = sorted(clean_build.glob("*.aux"))
    structural_aux_paths = sorted(clean_build.rglob("*.aux"))
    references = stable.parse_aux_references(aux_paths)
    aux_label_number_candidates, structural_aux_report = (
        parse_aux_label_number_candidates(structural_aux_paths)
    )
    structural_sources: dict[Path, str] = {}
    structural_source_read_rejections: list[dict[str, Any]] = []
    for source_path in executed_project:
        if source_path.suffix.casefold() not in {".tex", ".ltx", ".sty", ".cls"}:
            continue
        try:
            structural_sources[source_path.resolve()] = source_path.read_text(
                encoding="utf-8", errors="strict"
            )
        except (OSError, UnicodeError) as error:
            structural_source_read_rejections.append(
                {
                    "source_file": str(source_path),
                    "reason": f"exact UTF-8 structural source unavailable:{error}",
                    "pdf_text_used": False,
                }
            )
    try:
        theorem_ir = build_theorem_ir_from_sources(
            structural_sources,
            aux_label_number_candidates,
        )
        theorem_parser_report: dict[str, Any] = theorem_ir.as_report()
    except Exception as error:  # noqa: BLE001 - optional IR fails closed
        theorem_ir = None
        theorem_parser_report = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "generation_sources": ["latex_source", "compiler_aux"],
            "pdf_text_used": False,
        }
    table_numbers, table_number_report = parse_aux_table_numbers(aux_paths)
    heading_templates, heading_report = stable.parse_unique_titleformat_labels(
        executed_project
    )
    (
        heading_admission_blocks,
        serialized_heading_titles,
        safe_heading_rejections,
        safe_heading_admission_report,
    ) = prepare_heading_blocks_for_safe_admission(
        blocks,
        safe_macro_registry,
        references=references,
    )
    (
        paragraph_units,
        source_rejections,
        visible_wrapper_invocations,
        visible_wrapper_unit_report,
    ) = build_source_units_with_visible_wrappers(
        paragraphs,
        references=references,
        macros=visible_wrapper_macros,
        safe_macros=safe_macro_registry,
        execution_ir=execution_ir,
    )
    external_units, external_blocks_by_unit = external_verbatim_units(
        external_verbatim_ir,
        color_index_offset=len(paragraph_units),
    )
    heading_units, heading_rejections = stable.build_heading_units(
        heading_admission_blocks,
        aux_paths,
        color_index_offset=len(paragraph_units) + len(external_units),
        heading_label_templates=heading_templates,
    )
    heading_units, heading_markdown_application_report = apply_compiler_heading_labels(
        heading_units,
        heading_admission_blocks,
        compiler_labels,
        serialized_heading_titles,
    )
    heading_units, compiler_heading_recovery_report = recover_compiler_labeled_headings(
        heading_units,
        heading_admission_blocks,
        compiler_labels,
        serialized_heading_titles,
    )
    if theorem_ir is None:
        theorem_heading_units: list[stable.SourceUnit] = []
        theorem_heading_variants: dict[
            str, tuple[tuple[str, str], ...]
        ] = {}
        theorem_heading_rejections: list[dict[str, Any]] = []
        theorem_heading_report: dict[str, Any] = {
            "status": "disabled_after_parser_failure",
            "generation_sources": ["latex_source", "compiler_aux"],
            "pdf_text_used": False,
        }
    else:
        (
            theorem_heading_units,
            theorem_heading_variants,
            theorem_heading_rejections,
            theorem_heading_report,
        ) = build_theorem_heading_units(
            theorem_ir,
            color_index_offset=(
                len(paragraph_units) + len(external_units) + len(heading_units)
            ),
        )
    recovered_heading_ids = {
        str(item["source_block_id"])
        for item in compiler_heading_recovery_report["recovered_headings"]
    }
    heading_rejections = [
        item
        for item in heading_rejections
        if str(item.get("source_block_id")) not in recovered_heading_ids
    ]
    heading_rejections = [
        *safe_heading_rejections,
        *heading_rejections,
        *heading_markdown_application_report["rejections"],
    ]
    paragraph_macro_admission_report = visible_wrapper_unit_report[
        "safe_macro_admission"
    ]
    source_macro_admission_report = {
        "policy": "safe_macro_pre_admission_v1",
        "total": (
            int(paragraph_macro_admission_report["total"])
            + int(safe_heading_admission_report["total"])
        ),
        "changed": (
            int(paragraph_macro_admission_report["changed"])
            + int(safe_heading_admission_report["changed"])
        ),
        "successful": (
            int(paragraph_macro_admission_report["successful"])
            + int(safe_heading_admission_report["successful"])
        ),
        "rejected": (
            int(paragraph_macro_admission_report["rejected"])
            + int(safe_heading_admission_report["rejected"])
        ),
        "provenance": [
            *paragraph_macro_admission_report["provenance"],
            *safe_heading_admission_report["provenance"],
        ],
        "paragraphs": paragraph_macro_admission_report,
        "headings": safe_heading_admission_report,
        "heading_markdown_application": heading_markdown_application_report,
        "original_source_provenance_preserved": True,
        "pdf_text_used": False,
    }
    regular_before_overlap = [
        *paragraph_units,
        *external_units,
        *heading_units,
        *theorem_heading_units,
    ]
    regular_units, overlap_rejections = stable.reject_line_overlaps(
        regular_before_overlap
    )
    struct_units, struct_rejections = structural_units(
        blocks,
        color_index_offset=len(regular_before_overlap),
        table_numbers=table_numbers,
        references=references,
    )
    (
        struct_units,
        equation_tail_variants,
        equation_tail_report,
    ) = apply_display_equation_tail_ir(
        struct_units,
        blocks,
        aux_label_number_candidates,
    )
    structural_markdown_candidates = {
        **theorem_heading_variants,
        **equation_tail_variants,
    }
    units, execution_rejections, execution_report = order_units_by_execution(
        [*regular_units, *struct_units], execution_ir
    )
    source_rejections = [
        *source_rejections,
        *heading_rejections,
        *(rejection.as_json() for rejection in external_verbatim_ir.rejections),
        *(
            (rejection.as_json() for rejection in theorem_ir.rejections)
            if theorem_ir is not None
            else ()
        ),
        *theorem_heading_rejections,
        *overlap_rejections,
        *struct_rejections,
        *execution_rejections,
    ]
    structural_source_ir_report = {
        "policy": "strict_source_aux_structural_ir_v1",
        "generation_sources": ["latex_source", "compiler_aux"],
        "pdf_text_used_for_ground_truth": False,
        "source_files_considered": len(structural_sources),
        "source_read_rejections": structural_source_read_rejections,
        "aux_label_numbers": structural_aux_report,
        "theorem_parser": theorem_parser_report,
        "theorem_heading_admission": theorem_heading_report,
        "display_equation_tail_admission": equation_tail_report,
        "candidate_units": len(structural_markdown_candidates),
        "candidate_count": sum(
            len(values) for values in structural_markdown_candidates.values()
        ),
        "pdf_candidate_role": "exact_selection_and_rejection_only",
    }
    structural_source_ir_summary = {
        "policy": structural_source_ir_report["policy"],
        "report": "structural_source_ir.json",
        "source_files_considered": len(structural_sources),
        "source_read_rejections": len(structural_source_read_rejections),
        "aux_labels": structural_aux_report["labels"],
        "aux_ambiguous_labels": structural_aux_report["ambiguous_labels"],
        "theorem_blocks_admitted": (
            len(theorem_ir.blocks) if theorem_ir is not None else 0
        ),
        "theorem_heading_units": len(theorem_heading_units),
        "equation_tail_units": len(equation_tail_variants),
        "candidate_units": len(structural_markdown_candidates),
        "candidate_count": structural_source_ir_report["candidate_count"],
        "generation_sources": ["latex_source", "compiler_aux"],
        "pdf_text_used_for_ground_truth": False,
        "pdf_candidate_role": "exact_selection_and_rejection_only",
    }
    if not units:
        raise RuntimeError("v2 source parser produced no renderable units")
    macro_expansion_instrumentation_mismatches = (
        build_macro_expansion_instrumentation_mismatches(
            units,
            source_macro_admission_report["provenance"],
        )
    )

    shadows: list[ShadowCandidate] = []
    shadow_attempts: list[dict[str, Any]] = []
    shadow_instrumentation_safety: list[dict[str, Any]] = []
    shadows_root = output_dir / "shadows"
    synctex_report: dict[str, Any] = {"status": "not_attempted"}
    synctex_atom_line_report: dict[str, Any] = {"status": "not_attempted"}
    external_verbatim_color_report: dict[str, Any] = {"status": "not_attempted"}
    sync_base_probes, _sync_base_modes = stable.build_source_probes(
        units,
        word_probe_kinds={
            "paragraph",
            "itemize_item",
            "enumerate_item",
            "description_item",
        },
    )
    (
        sync_probes,
        sync_atom_locators,
        sync_unit_atoms,
        sync_modes,
    ) = build_atom_probes(
        units,
        sync_base_probes,
        visible_wrapper_invocations,
        include_plain_source_atoms=True,
    )
    sync_probes, sync_modes = replace_external_verbatim_line_probes(
        sync_probes,
        sync_modes,
        external_blocks_by_unit,
    )
    sync_mode_counts = collections.Counter(sync_modes.values())

    # Primary advanced locator: compile the exact clean source with engine
    # metadata enabled.  No color commands, TeX groups, or visible markers are
    # inserted, so complex class and macro behavior remains untouched.
    try:
        synctex_pdf, synctex_path = run_synctex_compile(
            source_root=clean_root,
            main_tex=args.main_tex,
            build_dir=output_dir / "build_synctex",
            log_path=output_dir / "logs" / "synctex_clean.log",
            timeout_seconds=args.compile_timeout,
            engine=args.engine,
        )
        synctex_geometry = color_pilot.compare_pdf_geometry(clean_pdf, synctex_pdf)
        synctex_invariance = compare_pdf_logical_invariance(clean_pdf, synctex_pdf)
        if synctex_geometry.get("status") != "passed":
            raise RuntimeError("SyncTeX clean compile changed PDF glyph geometry")
        if not synctex_invariance.get("page_count_equal") or not all(
            bool(page.get("logical_content_and_order_equal"))
            for page in synctex_invariance.get("pages", [])
        ):
            raise RuntimeError("SyncTeX clean compile changed PDF logical page content")
        sync_index = parse_synctex(synctex_path, source_root=clean_root)
        sync_raw_rows, sync_raw_summary = synctex_alignment_for_probes(
            sync_index,
            sync_probes,
            sync_atom_locators,
        )
        sync_rows, sync_summary = suppress_untrusted_external_synctex_rows(
            sync_probes, sync_raw_rows
        )
        sync_root = shadows_root / "synctex_clean"
        atomic_write_json(
            sync_root / "synctex_index.json",
            sync_index.as_json(clean_root),
        )
        atomic_write_jsonl(
            sync_root / "source_probes.jsonl",
            (
                probe_json(probe, clean_root, sync_atom_locators)
                for probe in sync_probes
            ),
        )
        atomic_write_jsonl(
            sync_root / "synctex_page_alignment.jsonl",
            (
                {"probe_id": probe_id, "pages": pages}
                for probe_id, pages in sync_rows.items()
            ),
        )
        shadows.append(
            ShadowCandidate(
                shadow_id="synctex_clean",
                probes=list(sync_probes),
                atom_locators=sync_atom_locators,
                unit_atoms=sync_unit_atoms,
                modes=sync_modes,
                colored_pdf=synctex_pdf,
                geometry=synctex_geometry,
                logical_invariance=synctex_invariance,
                color_rows=sync_rows,
                color_summary=sync_summary,
            )
        )
        synctex_report = {
            "status": "compiled",
            "metadata": sync_index.as_json(clean_root),
            "geometry": synctex_geometry,
            "logical_invariance": synctex_invariance,
            "alignment": sync_summary,
            "raw_alignment_before_external_gate": sync_raw_summary,
            "unit_modes": dict(sorted(sync_mode_counts.items())),
            "source_mutated_for_localization": False,
            "pdf_text_used_for_localization": False,
        }
        shadow_attempts.append({"shadow_id": "synctex_clean", **synctex_report})
        print(
            f"[synctex_locator_done] status=compiled "
            f"coverage={sync_summary['coverage']:.4f} "
            f"records={sync_index.records_indexed}",
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - locator is fail-closed
        synctex_report = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "source_mutated_for_localization": False,
            "pdf_text_used_for_localization": False,
        }
        shadow_attempts.append({"shadow_id": "synctex_clean", **synctex_report})
        print(
            f"[synctex_locator_done] status=failed "
            f"error={type(error).__name__}:{error}",
            flush=True,
        )

    # Fine source identity locator.  Comment-newline markers refine SyncTeX's
    # native line granularity to parsed atom boundaries without inserting any
    # executable or visible TeX material.  Trust it only when every clean PDF
    # glyph remains at the same coordinate.
    try:
        fine_container = shadows_root / "synctex_atom_lines"
        fine_root = fine_container / "source"
        fine_overrides, fine_instrumentation = (
            instrument_synctex_line_identity_tree(
                clean_root,
                fine_root,
                sync_probes,
                sync_atom_locators,
            )
        )
        fine_pdf, fine_synctex_path = run_synctex_compile(
            source_root=fine_root,
            main_tex=args.main_tex,
            build_dir=fine_container / "build",
            log_path=output_dir / "logs" / "synctex_atom_lines.log",
            timeout_seconds=args.compile_timeout,
            engine=args.engine,
        )
        fine_geometry = color_pilot.compare_pdf_geometry(clean_pdf, fine_pdf)
        fine_invariance = compare_pdf_logical_invariance(clean_pdf, fine_pdf)
        if fine_geometry.get("status") != "passed":
            raise RuntimeError("SyncTeX atom-line compile changed PDF glyph geometry")
        if not fine_invariance.get("page_count_equal") or not all(
            bool(page.get("logical_content_and_order_equal"))
            for page in fine_invariance.get("pages", [])
        ):
            raise RuntimeError("SyncTeX atom-line compile changed logical page content")
        fine_index = parse_synctex(fine_synctex_path, source_root=fine_root)
        fine_raw_rows, fine_raw_summary = synctex_alignment_for_probes(
            fine_index,
            sync_probes,
            sync_atom_locators,
            line_overrides=fine_overrides,
        )
        fine_rows, fine_summary = suppress_untrusted_external_synctex_rows(
            sync_probes, fine_raw_rows
        )
        atomic_write_json(
            fine_container / "synctex_index.json",
            fine_index.as_json(fine_root),
        )
        atomic_write_jsonl(
            fine_container / "source_probes.jsonl",
            (
                probe_json(probe, clean_root, sync_atom_locators)
                for probe in sync_probes
            ),
        )
        atomic_write_jsonl(
            fine_container / "synctex_page_alignment.jsonl",
            (
                {"probe_id": probe_id, "pages": pages}
                for probe_id, pages in fine_rows.items()
            ),
        )
        shadows.append(
            ShadowCandidate(
                shadow_id="synctex_atom_lines",
                probes=list(sync_probes),
                atom_locators=sync_atom_locators,
                unit_atoms=sync_unit_atoms,
                modes=sync_modes,
                colored_pdf=fine_pdf,
                geometry=fine_geometry,
                logical_invariance=fine_invariance,
                color_rows=fine_rows,
                color_summary=fine_summary,
            )
        )
        synctex_atom_line_report = {
            "status": "compiled",
            "instrumentation": fine_instrumentation,
            "metadata": fine_index.as_json(fine_root),
            "geometry": fine_geometry,
            "logical_invariance": fine_invariance,
            "alignment": fine_summary,
            "raw_alignment_before_external_gate": fine_raw_summary,
            "unit_modes": dict(sorted(sync_mode_counts.items())),
            "source_mutated_for_localization": True,
            "source_mutation_visible_or_executable": False,
            "pdf_text_used_for_localization": False,
        }
        shadow_attempts.append(
            {"shadow_id": "synctex_atom_lines", **synctex_atom_line_report}
        )
        print(
            f"[synctex_atom_lines_done] status=compiled "
            f"coverage={fine_summary['coverage']:.4f} "
            f"markers={fine_instrumentation['markers_inserted']} "
            f"records={fine_index.records_indexed}",
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - locator is fail-closed
        synctex_atom_line_report = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "source_mutated_for_localization": True,
            "source_mutation_visible_or_executable": False,
            "pdf_text_used_for_localization": False,
        }
        shadow_attempts.append(
            {"shadow_id": "synctex_atom_lines", **synctex_atom_line_report}
        )
        print(
            f"[synctex_atom_lines_done] status=failed "
            f"error={type(error).__name__}:{error}",
            flush=True,
        )

    # SyncTeX identifies ordinary source atoms well, but ``verbatiminput``
    # consumes another file through a package hook whose metadata often points
    # only to the invocation.  Compile one dedicated shadow that colors each
    # source-derived visible external line.  Its rows replace only those
    # external probes in the best already-gated SyncTeX row set.  The merged
    # candidate enters the same immutable-candidate and exact-verifier path as
    # every other shadow below.
    external_line_probes = [
        probe
        for probe in sync_probes
        if probe.localization_mode == EXTERNAL_VERBATIM_LOCALIZATION_MODE
    ]
    external_shadow_id = "synctex_atom_external_color"
    if not external_line_probes:
        external_verbatim_color_report = {
            "status": "skipped",
            "reason": "no_external_verbatim_line_probes",
            "pdf_text_used_for_localization": False,
        }
        shadow_attempts.append(
            {"shadow_id": external_shadow_id, **external_verbatim_color_report}
        )
        print(
            "[external_verbatim_color_done] status=skipped "
            "reason=no_external_verbatim_line_probes",
            flush=True,
        )
    else:
        base_synctex_shadow = best_synctex_shadow(shadows)
        try:
            if base_synctex_shadow is None:
                raise RuntimeError("no exact-invariant SyncTeX base shadow available")
            external_container = shadows_root / external_shadow_id
            external_root = external_container / "source"
            external_instrumentation = instrument_external_verbatim_color_tree(
                clean_root,
                external_root,
                external_blocks_by_unit,
                external_line_probes,
                args.engine,
            )
            if external_instrumentation["unsupported_blocks"]:
                raise RuntimeError(
                    "unsupported external verbatim blocks in color shadow: "
                    f"{external_instrumentation['unsupported_blocks']}"
                )
            if external_instrumentation["lines_instrumented"] != len(
                external_line_probes
            ):
                raise RuntimeError(
                    "external verbatim color instrumentation did not cover every "
                    f"visible line: {external_instrumentation['lines_instrumented']}"
                    f"/{len(external_line_probes)}"
                )
            external_pdf = color_pilot.run_compile(
                source_root=external_root,
                main_tex=args.main_tex,
                build_dir=external_container / "build",
                log_path=output_dir
                / "logs"
                / "synctex_atom_external_color.log",
                label="source-first-v2-synctex-atom-external-color",
                timeout_seconds=args.compile_timeout,
                engine=args.engine,
            )
            external_geometry = color_pilot.compare_pdf_geometry(
                clean_pdf, external_pdf
            )
            external_invariance = compare_pdf_logical_invariance(
                clean_pdf, external_pdf
            )
            external_gate = exact_pdf_shadow_gate(
                external_geometry, external_invariance
            )
            if external_gate["status"] != "passed":
                raise RuntimeError(
                    "external verbatim color shadow failed exact invariance gate: "
                    + ",".join(external_gate["reasons"])
                )
            external_rows, external_color_summary = extract_color_runs(
                external_pdf, external_line_probes
            )
            merged_rows, merged_summary = merge_external_verbatim_alignment(
                sync_probes,
                base_synctex_shadow.color_rows,
                external_rows,
            )
            merged_summary["base_synctex_shadow_id"] = (
                base_synctex_shadow.shadow_id
            )
            merged_summary["external_color_coverage"] = external_color_summary[
                "coverage"
            ]
            atomic_write_json(
                external_container / "instrumentation.json",
                external_instrumentation,
            )
            atomic_write_jsonl(
                external_container / "source_probes.jsonl",
                (
                    probe_json(probe, clean_root, sync_atom_locators)
                    for probe in sync_probes
                ),
            )
            atomic_write_jsonl(
                external_container / "color_page_alignment.jsonl",
                (
                    {"probe_id": probe_id, "pages": pages}
                    for probe_id, pages in merged_rows.items()
                ),
            )
            shadows.append(
                ShadowCandidate(
                    shadow_id=external_shadow_id,
                    probes=list(sync_probes),
                    atom_locators=sync_atom_locators,
                    unit_atoms=sync_unit_atoms,
                    modes=sync_modes,
                    colored_pdf=external_pdf,
                    geometry=external_geometry,
                    logical_invariance=external_invariance,
                    color_rows=merged_rows,
                    color_summary=merged_summary,
                )
            )
            external_verbatim_color_report = {
                "status": "compiled",
                "base_synctex_shadow_id": base_synctex_shadow.shadow_id,
                "instrumentation": external_instrumentation,
                "geometry": external_geometry,
                "logical_invariance": external_invariance,
                "exact_invariance_gate": external_gate,
                "external_color_alignment": external_color_summary,
                "merged_alignment": merged_summary,
                "unit_modes": dict(sorted(sync_mode_counts.items())),
                "source_mutated_for_localization": True,
                "source_mutation_visible_or_executable": True,
                "source_content_from_external_files": True,
                "pdf_text_used_for_localization": False,
            }
            shadow_attempts.append(
                {"shadow_id": external_shadow_id, **external_verbatim_color_report}
            )
            print(
                f"[external_verbatim_color_done] status=compiled "
                f"base={base_synctex_shadow.shadow_id} "
                f"external_lines={len(external_line_probes)} "
                f"coverage={merged_summary['coverage']:.4f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - locator is fail-closed
            external_verbatim_color_report = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "external_line_probes": len(external_line_probes),
                "source_mutated_for_localization": True,
                "source_mutation_visible_or_executable": True,
                "source_content_from_external_files": True,
                "pdf_text_used_for_localization": False,
            }
            shadow_attempts.append(
                {"shadow_id": external_shadow_id, **external_verbatim_color_report}
            )
            print(
                f"[external_verbatim_color_done] status=failed "
                f"error={type(error).__name__}:{error}",
                flush=True,
            )

    tier_specs = [
        (
            "source_atoms",
            {
                "paragraph",
                "itemize_item",
                "enumerate_item",
                "description_item",
            },
            True,
        ),
        (
            "legacy_words",
            {
                "paragraph",
                "itemize_item",
                "enumerate_item",
                "description_item",
            },
            False,
        ),
        ("whole_units", set(), False),
    ]
    for tier_index, (shadow_id, word_kinds, use_atoms) in enumerate(tier_specs, start=1):
        base_probes, base_modes = stable.build_source_probes(
            units, word_probe_kinds=word_kinds
        )
        if use_atoms or visible_wrapper_invocations:
            probes, atom_locators, unit_atoms, modes = build_atom_probes(
                units,
                base_probes,
                visible_wrapper_invocations,
                include_plain_source_atoms=use_atoms,
            )
        else:
            probes = base_probes
            atom_locators = {}
            unit_atoms = {}
            modes = base_modes
        mode_counts = collections.Counter(modes.values())
        shadow_root = shadows_root / shadow_id / "source"
        shadow_build = shadows_root / shadow_id / "build"
        print(
            f"[shadow_start] shadow={shadow_id} tier={tier_index}/{len(tier_specs)} "
            f"units={len(units)} probes={len(probes)} modes={dict(mode_counts)}",
            flush=True,
        )
        shutil.copytree(clean_root, shadow_root)
        instrumentation_safety: dict[str, Any] = {
            "status": "not_attempted",
            "policy": "source_syntactic_executable_color_gate_v1",
        }
        try:
            instrumentation_safety = instrument_shadow_tree(
                clean_root,
                shadow_root,
                units,
                probes,
                atom_locators,
                args.engine,
                macro_expansion_mismatches=(
                    macro_expansion_instrumentation_mismatches
                ),
            )
            shadow_instrumentation_safety.append(
                {"shadow_id": shadow_id, **instrumentation_safety}
            )
            atomic_write_json(
                shadows_root / shadow_id / "instrumentation_safety.json",
                instrumentation_safety,
            )
            colored_pdf = color_pilot.run_compile(
                source_root=shadow_root,
                main_tex=args.main_tex,
                build_dir=shadow_build,
                log_path=output_dir / "logs" / f"shadow_{shadow_id}.log",
                label=f"source-first-v2-{shadow_id}",
                timeout_seconds=args.compile_timeout,
                engine=args.engine,
            )
            geometry = color_pilot.compare_pdf_geometry(clean_pdf, colored_pdf)
            if not geometry["page_count_equal"]:
                raise RuntimeError("shadow page count differs from clean compile")
            logical_invariance = compare_pdf_logical_invariance(clean_pdf, colored_pdf)
            if not logical_invariance["page_count_equal"]:
                raise RuntimeError("shadow logical page count differs from clean compile")
            color_rows, color_summary = extract_color_runs(colored_pdf, probes)
            atomic_write_jsonl(
                shadows_root / shadow_id / "source_probes.jsonl",
                (probe_json(probe, clean_root, atom_locators) for probe in probes),
            )
            atomic_write_jsonl(
                shadows_root / shadow_id / "color_page_alignment.jsonl",
                (
                    {"probe_id": probe_id, "pages": pages}
                    for probe_id, pages in color_rows.items()
                ),
            )
            shadows.append(
                ShadowCandidate(
                    shadow_id=shadow_id,
                    probes=list(probes),
                    atom_locators=atom_locators,
                    unit_atoms=unit_atoms,
                    modes=modes,
                    colored_pdf=colored_pdf,
                    geometry=geometry,
                    logical_invariance=logical_invariance,
                    color_rows=color_rows,
                    color_summary=color_summary,
                )
            )
            shadow_attempts.append(
                {
                    "shadow_id": shadow_id,
                    "status": "compiled",
                    "geometry": geometry,
                    "logical_invariance": logical_invariance,
                    "color_alignment": color_summary,
                    "unit_modes": dict(sorted(mode_counts.items())),
                    "instrumentation_safety": instrumentation_safety,
                }
            )
            print(
                f"[shadow_done] shadow={shadow_id} status=compiled "
                f"coverage={color_summary['coverage']:.4f} "
                f"text_equal={geometry['character_text_equal']} "
                f"metadata_only={instrumentation_safety['metadata_only_units']}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - every tier is fail-closed
            if not any(
                row.get("shadow_id") == shadow_id
                for row in shadow_instrumentation_safety
            ):
                shadow_instrumentation_safety.append(
                    {"shadow_id": shadow_id, **instrumentation_safety}
                )
            shadow_attempts.append(
                {
                    "shadow_id": shadow_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "unit_modes": dict(sorted(mode_counts.items())),
                    "instrumentation_safety": instrumentation_safety,
                }
            )
            print(
                f"[shadow_done] shadow={shadow_id} status=failed "
                f"error={type(error).__name__}:{error}",
                flush=True,
            )
    if not shadows:
        raise RuntimeError("every v2 locator shadow failed")

    invariant_shadow_geometry_report: dict[str, Any] = {
        "status": "not_attempted",
        "policy": INVARIANT_UNIT_HYBRID_POLICY,
    }
    with pdfplumber.open(clean_pdf) as document:
        page_limit = min(len(document.pages), args.max_pages)
        page_widths = {
            index: float(document.pages[index - 1].width)
            for index in range(1, page_limit + 1)
        }
        page_heights = {
            index: float(document.pages[index - 1].height)
            for index in range(1, page_limit + 1)
        }
        hybrid_shadows, invariant_shadow_geometry_report = (
            derive_invariant_unit_hybrid_shadows(
                shadows,
                units,
                page_widths,
            )
        )
        for hybrid in hybrid_shadows:
            hybrid_root = shadows_root / hybrid.shadow_id
            atomic_write_jsonl(
                hybrid_root / "source_probes.jsonl",
                (
                    probe_json(probe, clean_root, hybrid.atom_locators)
                    for probe in hybrid.probes
                ),
            )
            atomic_write_jsonl(
                hybrid_root / "hybrid_page_alignment.jsonl",
                (
                    {"probe_id": probe_id, "pages": pages}
                    for probe_id, pages in hybrid.color_rows.items()
                ),
            )
            derivation = next(
                value
                for value in invariant_shadow_geometry_report["hybrids"]
                if value["shadow_id"] == hybrid.shadow_id
            )
            atomic_write_json(hybrid_root / "derivation.json", derivation)
            shadow_attempts.append(
                {
                    "shadow_id": hybrid.shadow_id,
                    "status": "derived",
                    "policy": INVARIANT_UNIT_HYBRID_POLICY,
                    "base_shadow_id": derivation["base_shadow_id"],
                    "units_replaced": len(derivation["unit_replacements"]),
                    "donor_counts": hybrid.color_summary["donor_counts"],
                    "coverage": hybrid.color_summary["coverage"],
                    "ground_truth_source": "SourceUnit",
                    "donor_fields_used": ["page_number", "bbox_points"],
                    "pdf_text_used": False,
                }
            )
        shadows.extend(hybrid_shadows)
        atomic_write_json(
            output_dir / "invariant_shadow_geometry_fallback.json",
            invariant_shadow_geometry_report,
        )
        print(
            f"[invariant_unit_hybrid_done] "
            f"status={invariant_shadow_geometry_report['status']} "
            f"hybrids={len(hybrid_shadows)} "
            f"units_replaced={invariant_shadow_geometry_report.get('units_replaced', 0)} "
            f"rejections={sum(invariant_shadow_geometry_report.get('rejection_counts', {}).values())}",
            flush=True,
        )
        shadow_fragments: dict[str, dict[int, list[LocatedFragment]]] = {}
        shadow_reasons: dict[str, dict[int, set[str]]] = {}
        shadow_localization: dict[str, dict[str, int]] = {}
        for shadow in shadows:
            fragments, reasons, summary = build_fragments_for_shadow(
                units,
                shadow.probes,
                shadow.color_rows,
                shadow.unit_atoms,
                shadow.atom_locators,
                page_widths,
                external_blocks=external_blocks_by_unit,
                structural_markdown_candidates=structural_markdown_candidates,
            )
            shadow_fragments[shadow.shadow_id] = fragments
            shadow_reasons[shadow.shadow_id] = reasons
            shadow_localization[shadow.shadow_id] = summary

        # Freeze every source/geometry candidate before the first clean-page
        # text extraction.  PDF text below can only exact-select this bounded
        # immutable set; it cannot influence a carrier, cut, order, or policy.
        shadow_frozen_candidates: dict[str, dict[int, dict[str, Any]]] = {}
        for shadow in shadows:
            frozen_pages: dict[int, dict[str, Any]] = {}
            fragments_by_page = shadow_fragments[shadow.shadow_id]
            for page_number in range(1, page_limit + 1):
                fragments = fragments_by_page.get(page_number, [])
                if not fragments:
                    continue
                frontier_variants, frontier_report = (
                    enumerate_leading_frontier_variants(
                        page_number,
                        fragments,
                        units=units,
                        probes=shadow.probes,
                        rows=shadow.color_rows,
                    )
                )
                frozen_pages[page_number] = freeze_page_source_candidates(
                    fragments,
                    page_width=page_widths[page_number],
                    page_height=page_heights[page_number],
                    frontier_variants=frontier_variants,
                    frontier_report=frontier_report,
                )
            shadow_frozen_candidates[shadow.shadow_id] = frozen_pages

        passed_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        complex_passed = 0
        eligible_pages = 0
        pages_dir = output_dir / "pages"
        for page_number in range(1, page_limit + 1):
            page = document.pages[page_number - 1]
            pdf_text, pdf_layout = stable.pdf_verifier_text(page)
            visible_characters = len(
                stable.exact_visible_character_stream(pdf_text, markdown=False).replace(
                    stable.OPTIONAL_LINE_END_HYPHEN, ""
                )
            )
            eligible = visible_characters >= args.min_eligible_visible_characters
            eligible_pages += eligible
            exact_options: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            attempts: list[dict[str, Any]] = []
            for shadow in shadows:
                logical_page = next(
                    (
                        row
                        for row in shadow.logical_invariance["pages"]
                        if int(row["page_number"]) == page_number
                    ),
                    None,
                )
                invariant = bool(
                    logical_page
                    and logical_page["logical_content_and_order_equal"]
                )
                fragments = shadow_fragments[shadow.shadow_id].get(page_number, [])
                reasons = sorted(shadow_reasons[shadow.shadow_id].get(page_number, set()))
                if not invariant or not fragments or reasons:
                    attempts.append(
                        {
                            "shadow_id": shadow.shadow_id,
                            "status": "ineligible",
                            "shadow_text_invariant": invariant,
                            "fragments": len(fragments),
                            "reasons": [
                                *(("shadow_text_mismatch",) if not invariant else ()),
                                *(("no_source_fragments",) if not fragments else ()),
                                *reasons,
                            ],
                        }
                    )
                    continue
                frozen = shadow_frozen_candidates[shadow.shadow_id].get(
                    page_number,
                    {
                        "status": "failed",
                        "reason": "source_candidates_not_frozen",
                        "candidates": [],
                        "candidate_count": 0,
                    },
                )
                result = verify_frozen_page_candidates(
                    frozen,
                    pdf_text,
                    pdf_page=page,
                )
                result["shadow_id"] = shadow.shadow_id
                result["shadow_coverage"] = shadow.color_summary["coverage"]
                attempts.append(result)
                if result["status"] == "passed":
                    exact_options[result["markdown"]].append(result)
            selected: dict[str, Any] | None = None
            rejection_reasons: list[str] = []
            if not eligible:
                rejection_reasons.append("below_fixed_visible_character_threshold")
            elif len(exact_options) == 1:
                options = next(iter(exact_options.values()))
                selected = max(
                    options,
                    key=lambda value: (
                        float(value["shadow_coverage"]),
                        value["shadow_id"] == "synctex_atom_external_color",
                        value["shadow_id"] == "synctex_atom_lines",
                        value["shadow_id"] == "synctex_clean",
                        value["shadow_id"] == "source_atoms",
                    ),
                )
            elif not exact_options:
                rejection_reasons.append("no_unique_exact_source_candidate")
            else:
                rejection_reasons.append("multiple_exact_markdown_serializations")
            status = "passed" if selected is not None and not rejection_reasons else "rejected"
            layout_report = selected["layout"] if selected else next(
                (
                    attempt.get("layout")
                    for attempt in attempts
                    if isinstance(attempt.get("layout"), dict)
                ),
                {"layout_bucket": "unknown", "layout_kind": "unknown"},
            )
            layout_bucket = str(layout_report.get("layout_bucket") or "unknown")
            if layout_bucket not in {
                "single_column",
                "two_column",
                "mixed_full_two_column",
                "other",
                "unknown",
            }:
                layout_bucket = "unknown"
            row: dict[str, Any] = {
                "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
                "contract": EXPERIMENTAL_CONTRACT,
                "probe_policy_version": PROBE_POLICY_VERSION,
                "layout_policy_version": LAYOUT_POLICY_VERSION,
                "data_id": f"{args.paper_id}_page_{page_number:04d}_sfspanv2",
                "paper_id": args.paper_id,
                "page_number": page_number,
                "status": status,
                "rejection_reasons": rejection_reasons,
                "generation_source": "latex_source",
                "page_provenance": "compiled_source_metadata_span_graph",
                "pdf_role": "independent_verifier_only",
                "layout": layout_report,
                "layout_bucket": layout_bucket,
                "pdf_verifier_layout_diagnostic": pdf_layout,
                "clean": True,
                "eligible_text_page": eligible,
                "visible_characters": visible_characters,
                "source_first_passed": status == "passed",
                "source_first_verifier_exact": status == "passed",
                "edit_accepted": False,
                "verifier_exact": False,
                "shadow_attempts": attempts,
            }
            if selected is not None:
                markdown = str(selected["markdown"])
                stem = f"page_{page_number:04d}"
                markdown_path = pages_dir / f"{stem}.md"
                image_path = pages_dir / f"{stem}.png"
                metadata_path = pages_dir / f"{stem}.json"
                atomic_write_text(markdown_path, markdown)
                page_gt.render_page_png(clean_pdf, page_number, image_path, args.dpi)
                row.update(
                    {
                        "selected_shadow_id": selected["shadow_id"],
                        "selected_order": selected["selected_order"],
                        "selected_serialization_policy": selected[
                            "selected_serialization_policy"
                        ],
                        "source_fragment_ids": selected["fragment_ids"],
                        "verifier": selected["verifier"],
                        "markdown": markdown_path.relative_to(output_dir).as_posix(),
                        "image": image_path.relative_to(output_dir).as_posix(),
                        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                    }
                )
                atomic_write_json(metadata_path, row)
                passed_rows.append(row)
                if layout_bucket in {"two_column", "mixed_full_two_column", "other"}:
                    complex_passed += 1
            else:
                rejected_rows.append(row)
            ledger_rows.append(
                {
                    "page_id": row["data_id"],
                    "paper_id": args.paper_id,
                    "page_number": page_number,
                    "candidate": True,
                    "clean": True,
                    "eligible_text_page": eligible,
                    "source_first_passed": status == "passed",
                    "source_first_verifier_exact": status == "passed",
                    "edit_accepted": False,
                    "verifier_exact": False,
                    "layout": layout_bucket,
                    "status": status,
                    "rejection_reasons": rejection_reasons,
                }
            )
            completed = page_number
            elapsed = max(time.monotonic() - started, 1e-9)
            throughput = completed / elapsed
            eta = (page_limit - completed) / throughput if throughput else 0.0
            print(
                f"[page_done] page={page_number}/{page_limit} status={status} "
                f"layout={layout_bucket} accepted={len(passed_rows)} "
                f"rejected={len(rejected_rows)} pct={100*completed/page_limit:.1f}% "
                f"throughput={throughput:.3f}_pages/s elapsed={color_pilot.elapsed_text(elapsed)} "
                f"eta={color_pilot.elapsed_text(eta)}",
                flush=True,
            )

    atomic_write_jsonl(
        output_dir / "source_units.jsonl",
        (unit.as_json(clean_root) for unit in units),
    )
    atomic_write_json(
        output_dir / "external_verbatim_ir.json",
        external_verbatim_ir.as_json(),
    )
    atomic_write_json(
        output_dir / "structural_source_ir.json",
        structural_source_ir_report,
    )
    atomic_write_jsonl(output_dir / "source_rejections.jsonl", source_rejections)
    atomic_write_jsonl(output_dir / "pages_passed.jsonl", passed_rows)
    atomic_write_jsonl(output_dir / "pages_rejected.jsonl", rejected_rows)
    atomic_write_jsonl(output_dir / "page_ledger_v2.jsonl", ledger_rows)
    source_first_yield = len(passed_rows) / max(1, eligible_pages)
    all_clean_pages_yield = len(passed_rows) / max(1, len(ledger_rows))
    exact_rate = (
        sum(
            row.get("verifier", {}).get("exact_ordered_character_stream_match") is True
            for row in passed_rows
        )
        / max(1, len(passed_rows))
    )
    target = {
        "overall_source_first_yield_gt_0_30": source_first_yield > 0.30,
        "accepted_complex_layout_pages_gt_0": complex_passed > 0,
        "accepted_exact_verifier_rate_1_0": bool(passed_rows) and exact_rate == 1.0,
    }
    target["passed"] = all(target.values())
    report = {
        "schema_version": EXPERIMENTAL_SCHEMA_VERSION,
        "contract": EXPERIMENTAL_CONTRACT,
        "probe_policy_version": PROBE_POLICY_VERSION,
        "layout_policy_version": LAYOUT_POLICY_VERSION,
        "status": "passed" if passed_rows else "failed",
        "paper_id": args.paper_id,
        "source_dir": str(source_dir),
        "main_tex": args.main_tex.as_posix(),
        "compile_engine": args.engine,
        "clean_pdf": str(clean_pdf),
        "reference_removal": reference_report,
        "figure_policy": figure_policy,
        "figure_removal": figure_report,
        "heading_label_resolution": heading_report,
        "compiler_heading_label_resolution": compiler_heading_report,
        "compiler_heading_recovery": compiler_heading_recovery_report,
        "compiler_table_number_resolution": table_number_report,
        "structural_source_ir": structural_source_ir_summary,
        "visible_wrapper_definitions": visible_wrapper_definition_report,
        "visible_wrapper_units": visible_wrapper_unit_report,
        "safe_macro_registry": safe_macro_registry.as_report(),
        "source_macro_admission": source_macro_admission_report,
        "macro_expansion_instrumentation_mismatches": {
            "units": len(macro_expansion_instrumentation_mismatches),
            "provenance": list(
                macro_expansion_instrumentation_mismatches.values()
            ),
            "pdf_text_used": False,
        },
        "source_execution_ir": execution_report,
        "synctex_clean_locator": synctex_report,
        "synctex_atom_line_locator": synctex_atom_line_report,
        "external_verbatim_color_locator": external_verbatim_color_report,
        "invariant_shadow_geometry_fallback": {
            "status": invariant_shadow_geometry_report.get("status"),
            "policy": invariant_shadow_geometry_report.get("policy"),
            "audit": "invariant_shadow_geometry_fallback.json",
            "hybrids_created": invariant_shadow_geometry_report.get(
                "hybrids_created", 0
            ),
            "units_replaced": invariant_shadow_geometry_report.get(
                "units_replaced", 0
            ),
            "rejection_counts": invariant_shadow_geometry_report.get(
                "rejection_counts", {}
            ),
            "base_shadows_immutable": True,
            "ground_truth_source": "SourceUnit",
            "donor_fields_used": ["page_number", "bbox_points"],
            "pdf_text_used": False,
        },
        "stable_guard": stable_guard,
        "source_paragraphs_total": len(paragraphs),
        "source_units_renderable": len(units),
        "source_units_rejected": len(source_rejections),
        "structural_units": len(struct_units),
        "theorem_heading_units": len(theorem_heading_units),
        "external_verbatim": {
            "blocks": len(external_verbatim_ir.blocks),
            "line_records": len(external_verbatim_ir.records),
            "units": len(external_units),
            "rejections": len(external_verbatim_ir.rejections),
            "generation_source": "latex_source",
            "pdf_text_used": False,
        },
        "shadow_attempts": shadow_attempts,
        "shadow_instrumentation_safety": shadow_instrumentation_safety,
        "shadow_localization": shadow_localization,
        "pages_total": len(ledger_rows),
        "eligible_clean_text_pages": eligible_pages,
        "pages_passed": len(passed_rows),
        "pages_rejected": len(rejected_rows),
        "accepted_complex_layout_pages": complex_passed,
        "source_first_yield": round(source_first_yield, 8),
        "all_clean_pages_yield": round(all_clean_pages_yield, 8),
        "accepted_exact_verifier_rate": round(exact_rate, 8),
        "target": target,
        "pdf_used_for_generation": False,
        "pdf_used_for_verification": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "validation_report_v2.json", report)
    print(
        f"[finish] status={report['status']} pages={len(passed_rows)}/{eligible_pages} "
        f"yield={100*source_first_yield:.2f}% complex={complex_passed} "
        f"target_passed={target['passed']} elapsed={color_pilot.elapsed_text(time.monotonic()-started)} "
        f"output={output_dir}",
        flush=True,
    )
    return 0 if passed_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
