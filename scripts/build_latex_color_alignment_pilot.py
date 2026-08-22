#!/usr/bin/env python3
"""Build a clean/color-shadow LaTeX alignment pilot.

The script colors conservative visible source tokens with unique RGB values,
compiles clean and colored copies of the same source tree, reads vector text
colors from the colored PDF, and verifies that color instrumentation did not
change pagination, extracted characters, or character geometry.

This is deliberately a provenance/positioning pilot, not a general LaTeX to
Markdown converter. Unsupported macro expansions remain unclaimed and must be
handled by later compiler hooks or rejected by a coverage gate.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable

import pdfplumber


DEFAULT_PAPER_DIR = Path("outputs/arxiv_latex_recompile_2000/papers/2510.03415v3")
DEFAULT_OUTPUT_DIR = Path("output/pdf/latex_color_alignment_pilot_2510.03415v3")
DEFAULT_TARGET_SOURCE = Path("intro.tex")
DEFAULT_MAIN_TEX = Path("main.tex")
DEFAULT_START_LINE = 4
DEFAULT_END_LINE = 115
LATEXMK = Path(shutil.which("latexmk") or "/Library/TeX/texbin/latexmk")
PDFTOPPM = Path(shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm")
HEARTBEAT_SECONDS = 15.0
GEOMETRY_TOLERANCE_POINTS = 0.05
TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+(?:\.\d+)*|[.,;:!?()]",
    re.UNICODE,
)
NONVISIBLE_ARGUMENT_COMMANDS = {
    "begin",
    "bibliography",
    "bibliographystyle",
    "cite",
    "citealp",
    "citeauthor",
    "citep",
    "citet",
    "citeyear",
    "end",
    "eqref",
    "include",
    "input",
    "label",
    "pageref",
    "ref",
    "url",
}
CITATION_COMMANDS = {
    "autocite",
    "autocites",
    "bibentry",
    "cite",
    "citea",
    "citeaffixed",
    "citealias",
    "citealp",
    "citealt",
    "citeasnoun",
    "citeauthor",
    "citen",
    "citenp",
    "citenum",
    "citep",
    "citepos",
    "citeposs",
    "citet",
    "citetext",
    "citeyear",
    "citeyearnp",
    "citeyearpar",
    "footcite",
    "footcites",
    "footfullcite",
    "fullcite",
    "nocite",
    "notecite",
    "onlinecite",
    "parencite",
    "parencites",
    "shortcite",
    "shortcitep",
    "smartcite",
    "smartcites",
    "supercite",
    "textcite",
    "textcites",
}
BIBLIOGRAPHY_COMMANDS = {
    "addbibresource",
    "bibliography",
    "bibliographystyle",
    "printbibliography",
    "putbib",
}
CITATION_CONFIGURATION_COMMANDS = {
    "atnextcite",
    "ateverycite",
    "ateverycitekey",
    "citestyle",
    "declarecitecommand",
    "declaremulticitecommand",
    "defcitealias",
    "setcitestyle",
}
REFERENCE_ENVIRONMENTS = ("thebibliography", "references")


@dataclasses.dataclass(frozen=True)
class ColoredToken:
    token_id: str
    text: str
    rgb: tuple[int, int, int]
    source_file: str
    source_line: int
    source_column: int
    kind: str

    def as_json(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "text": self.text,
            "rgb": list(self.rgb),
            "hex": "#" + "".join(f"{component:02x}" for component in self.rgb),
            "source_file": self.source_file,
            "source_line": self.source_line,
            "source_column": self.source_column,
            "kind": self.kind,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX)
    parser.add_argument("--target-source", type=Path, default=DEFAULT_TARGET_SOURCE)
    parser.add_argument("--start-line", type=int, default=DEFAULT_START_LINE)
    parser.add_argument("--end-line", type=int, default=DEFAULT_END_LINE)
    parser.add_argument(
        "--mode",
        choices=("token", "paragraph"),
        default="token",
        help="color individual literal tokens or whole source paragraphs",
    )
    parser.add_argument(
        "--drop-references",
        action="store_true",
        help="remove inline citation commands and bibliography output before both compiles",
    )
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--compile-timeout", type=int, default=300)
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for count, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}m{seconds % 60:02d}s"


def tex_comment_start(line: str) -> int:
    for index, character in enumerate(line):
        if character != "%":
            continue
        slashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2 == 0:
            return index
    return len(line)


def deterministic_rgb(index: int) -> tuple[int, int, int]:
    """Return a unique deterministic 24-bit color for realistic page token counts."""

    if index < 0 or index >= 2**24:
        raise ValueError(f"color index out of range: {index}")
    value = ((index + 1) * 0x9E3779) & 0xFFFFFF
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


def balanced_group_end(value: str, start: int, opening: str, closing: str) -> int:
    if start >= len(value) or value[start] != opening:
        return start
    depth = 0
    index = start
    while index < len(value):
        character = value[index]
        if character == "\\":
            index += 2
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(value)


def tex_blank_preserving_lines(value: str) -> str:
    """Remove TeX input while retaining source line numbers without adding paragraphs."""

    output: list[str] = []
    for index, character in enumerate(value):
        if character == "\n":
            if index > 0 and value[index - 1] == "\r":
                output.append("%\r\n")
            else:
                output.append("%\n")
    return "".join(output)


def consume_tex_command_groups(
    value: str,
    start: int,
) -> tuple[int, list[tuple[str, str]]]:
    """Consume a command's star and all immediately following []/{} groups."""

    index = start
    if index < len(value) and value[index] == "*":
        index += 1
    groups: list[tuple[str, str]] = []
    while index < len(value):
        probe = index
        while probe < len(value) and value[probe].isspace():
            probe += 1
        if probe >= len(value) or value[probe] not in "[{":
            break
        opening = value[probe]
        closing = "]" if opening == "[" else "}"
        end = balanced_group_end(value, probe, opening, closing)
        if end <= probe or end > len(value):
            break
        groups.append((opening, value[probe + 1 : end - 1]))
        index = end
    return index, groups


def strip_reference_content(source: str) -> tuple[str, dict[str, Any]]:
    """Remove visible citations and bibliography output from one TeX source.

    Comments are retained. Removed multiline constructs are replaced with TeX
    comment-only lines so downstream source line numbers stay stable.
    """

    stats: dict[str, Any] = {
        "citation_commands": collections.Counter(),
        "bibliography_commands": collections.Counter(),
        "reference_environments": collections.Counter(),
        "reference_headings": 0,
        "input_bbl_commands": 0,
    }
    transformed = source
    for environment in REFERENCE_ENVIRONMENTS:
        pattern = re.compile(
            r"\\begin\s*\{" + re.escape(environment) + r"\}.*?"
            r"\\end\s*\{" + re.escape(environment) + r"\}",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def replace_environment(match: re.Match[str], *, name: str = environment) -> str:
            stats["reference_environments"][name] += 1
            return tex_blank_preserving_lines(match.group(0))

        transformed = pattern.sub(replace_environment, transformed)

    heading_pattern = re.compile(
        r"\\(?:chapter|section|subsection)\*?\s*"
        r"\{\s*(?:references|bibliography)\s*\}",
        flags=re.IGNORECASE,
    )

    def replace_heading(match: re.Match[str]) -> str:
        stats["reference_headings"] += 1
        return tex_blank_preserving_lines(match.group(0))

    transformed = heading_pattern.sub(replace_heading, transformed)

    output: list[str] = []
    index = 0
    while index < len(transformed):
        character = transformed[index]
        if character == "%":
            line_end = transformed.find("\n", index)
            if line_end < 0:
                output.append(transformed[index:])
                break
            output.append(transformed[index : line_end + 1])
            index = line_end + 1
            continue
        if character != "\\":
            output.append(character)
            index += 1
            continue
        command_start = index
        if index + 1 < len(transformed) and (
            transformed[index + 1].isalpha() or transformed[index + 1] == "@"
        ):
            command_end = index + 2
            while command_end < len(transformed) and (
                transformed[command_end].isalpha() or transformed[command_end] == "@"
            ):
                command_end += 1
        else:
            command_end = min(len(transformed), index + 2)
        command = transformed[index + 1 : command_end]
        command_key = command.casefold()
        arguments_end, groups = consume_tex_command_groups(transformed, command_end)
        braced_groups = [content for opening, content in groups if opening == "{"]
        remove = False
        if command_key in CITATION_COMMANDS:
            if not braced_groups:
                raise ValueError(f"citation command has no required argument: \\{command}")
            stats["citation_commands"][command] += 1
            remove = True
        elif command_key in BIBLIOGRAPHY_COMMANDS:
            stats["bibliography_commands"][command] += 1
            remove = True
        elif command_key in {"input", "include"} and braced_groups:
            candidate = braced_groups[0].strip().casefold()
            if candidate.endswith(".bbl"):
                stats["input_bbl_commands"] += 1
                remove = True
        if not remove:
            output.append(transformed[command_start:command_end])
            index = command_end
            continue
        if output and output[-1] == "~":
            output.pop()
        removed = transformed[command_start:arguments_end]
        output.append(tex_blank_preserving_lines(removed))
        index = arguments_end

    serializable = dict(stats)
    for key in (
        "citation_commands",
        "bibliography_commands",
        "reference_environments",
    ):
        serializable[key] = dict(sorted(stats[key].items()))
    return "".join(output), serializable


def visible_reference_markers(source: str) -> list[str]:
    """Return citation/bibliography markers remaining outside comments."""

    markers: list[str] = []
    for line in source.splitlines():
        visible = line[: tex_comment_start(line)]
        for match in re.finditer(r"\\([A-Za-z@]+)", visible):
            command = match.group(1)
            command_key = command.casefold()
            if (
                command_key in CITATION_COMMANDS
                or command_key in BIBLIOGRAPHY_COMMANDS
                or command_key == "bibitem"
            ):
                markers.append("\\" + command)
            elif (
                command_key not in CITATION_CONFIGURATION_COMMANDS
                and (
                    command_key.startswith("cite")
                    or command_key.endswith("cite")
                    or command_key.endswith("cites")
                )
            ):
                markers.append("unknown-citation-command:\\" + command)
        if re.search(
            r"\\begin\s*\{\s*(?:thebibliography|references)\s*\}",
            visible,
            flags=re.IGNORECASE,
        ):
            markers.append("reference-environment")
    return markers


def strip_references_tree(source_root: Path) -> dict[str, Any]:
    """Rewrite a copied source tree and fail if visible reference markers remain."""

    files: list[dict[str, Any]] = []
    totals: collections.Counter[str] = collections.Counter()
    residuals: list[dict[str, Any]] = []
    candidates = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".tex", ".ltx"}
    )
    for file_index, path in enumerate(candidates, start=1):
        original = path.read_text(encoding="utf-8", errors="replace")
        transformed, stats = strip_reference_content(original)
        markers = visible_reference_markers(transformed)
        if transformed != original:
            atomic_write_text(path, transformed)
        citation_count = sum(stats["citation_commands"].values())
        bibliography_count = sum(stats["bibliography_commands"].values())
        environment_count = sum(stats["reference_environments"].values())
        totals.update(
            {
                "citation_commands": citation_count,
                "bibliography_commands": bibliography_count,
                "reference_environments": environment_count,
                "reference_headings": stats["reference_headings"],
                "input_bbl_commands": stats["input_bbl_commands"],
                "files_changed": int(transformed != original),
            }
        )
        if markers:
            residuals.append(
                {
                    "source_file": path.relative_to(source_root).as_posix(),
                    "markers": sorted(set(markers)),
                }
            )
        if transformed != original or markers:
            files.append(
                {
                    "source_file": path.relative_to(source_root).as_posix(),
                    "changed": transformed != original,
                    "stats": stats,
                    "residual_markers": sorted(set(markers)),
                }
            )
        print(
            f"[reference_scan] file={file_index}/{len(candidates)} "
            f"changed={totals['files_changed']} citations={totals['citation_commands']} "
            f"bibliography={totals['bibliography_commands']} current={path.name}",
            flush=True,
        )
    support_candidates = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".sty", ".cls", ".def"}
    )
    # Style files commonly choose a bibliography style or register a .bib
    # database without printing anything. Only reject support-file constructs
    # that can actually emit a bibliography by themselves.
    bibliography_marker_names = {
        "\\bibliography",
        "\\printbibliography",
        "\\putbib",
        "reference-environment",
    }
    for path in support_candidates:
        source = path.read_text(encoding="utf-8", errors="replace")
        markers = [
            marker
            for marker in visible_reference_markers(source)
            if marker in bibliography_marker_names
        ]
        visible = "\n".join(
            line[: tex_comment_start(line)] for line in source.splitlines()
        )
        if re.search(
            r"\\(?:input|include)\s*\{[^{}]*\.bbl\s*\}",
            visible,
            flags=re.IGNORECASE,
        ):
            markers.append("input-bbl-in-support-file")
        if markers:
            residuals.append(
                {
                    "source_file": path.relative_to(source_root).as_posix(),
                    "markers": sorted(set(markers)),
                }
            )
    report = {
        "status": "passed" if not residuals else "failed",
        "files_scanned": len(candidates),
        "support_files_audited": len(support_candidates),
        "totals": dict(totals),
        "files": files,
        "residuals": residuals,
    }
    if residuals:
        raise RuntimeError(
            "reference removal left visible markers: "
            + json.dumps(residuals[:20], ensure_ascii=False)
        )
    return report


def consume_nonvisible_command_arguments(value: str, start: int) -> int:
    index = start
    while index < len(value) and value[index].isspace() and value[index] != "\n":
        index += 1
    if index < len(value) and value[index] == "[":
        index = balanced_group_end(value, index, "[", "]")
        while index < len(value) and value[index].isspace() and value[index] != "\n":
            index += 1
    if index < len(value) and value[index] == "{":
        index = balanced_group_end(value, index, "{", "}")
    return index


def colorize_source_range(
    source: str,
    *,
    source_file: str,
    start_line: int,
    end_line: int,
) -> tuple[str, list[ColoredToken]]:
    """Color conservative visible tokens in an inclusive source-line range."""

    if start_line < 1 or end_line < start_line:
        raise ValueError("invalid source line range")
    lines = source.splitlines(keepends=True)
    if end_line > len(lines):
        raise ValueError(f"end line {end_line} exceeds source length {len(lines)}")
    output: list[str] = []
    tokens: list[ColoredToken] = []
    in_inline_math = False
    in_display_math = False
    for line_number, line in enumerate(lines, start=1):
        if line_number < start_line or line_number > end_line:
            output.append(line)
            continue
        comment_at = tex_comment_start(line)
        visible = line[:comment_at]
        comment = line[comment_at:]
        rendered: list[str] = []
        index = 0
        while index < len(visible):
            if visible.startswith("$$", index):
                in_display_math = not in_display_math
                rendered.append("$$")
                index += 2
                continue
            character = visible[index]
            if character == "$" and not in_display_math:
                in_inline_math = not in_inline_math
                rendered.append(character)
                index += 1
                continue
            if character == "\\":
                if index + 1 < len(visible) and visible[index + 1].isalpha():
                    end = index + 2
                    while end < len(visible) and (
                        visible[end].isalpha() or visible[end] == "@"
                    ):
                        end += 1
                    command = visible[index + 1 : end]
                else:
                    end = min(len(visible), index + 2)
                    command = visible[index + 1 : end]
                if command == "[":
                    in_display_math = True
                elif command == "]":
                    in_display_math = False
                rendered.append(visible[index:end])
                index = end
                if command in NONVISIBLE_ARGUMENT_COMMANDS:
                    argument_end = consume_nonvisible_command_arguments(visible, index)
                    rendered.append(visible[index:argument_end])
                    index = argument_end
                continue
            if not in_inline_math and not in_display_math:
                match = TOKEN_PATTERN.match(visible, index)
                if match is not None:
                    text = match.group(0)
                    token_index = len(tokens)
                    rgb = deterministic_rgb(token_index)
                    token = ColoredToken(
                        token_id=f"tok-{token_index + 1:06d}",
                        text=text,
                        rgb=rgb,
                        source_file=source_file,
                        source_line=line_number,
                        source_column=index + 1,
                        kind="word" if any(char.isalpha() for char in text) else "punctuation",
                    )
                    tokens.append(token)
                    rendered.append(
                        "{\\color[RGB]{"
                        + ",".join(str(component) for component in rgb)
                        + "}"
                        + text
                        + "}"
                    )
                    index = match.end()
                    continue
            rendered.append(character)
            index += 1
        output.append("".join(rendered) + comment)
    return "".join(output), tokens


def colorize_paragraphs(
    source: str,
    *,
    source_file: str,
    start_line: int,
    end_line: int,
) -> tuple[str, list[ColoredToken]]:
    """Color each blank-line-delimited source paragraph as one unit.

    The color command is inserted only at paragraph boundaries. Keeping all
    words inside one color span avoids the kerning, ligature, and italic-
    correction changes caused by wrapping every word in a separate TeX group.
    """

    if start_line < 1 or end_line < start_line:
        raise ValueError("invalid source line range")
    lines = source.splitlines(keepends=True)
    if end_line > len(lines):
        raise ValueError(f"end line {end_line} exceeds source length {len(lines)}")

    selected = lines[start_line - 1 : end_line]
    output: list[str] = list(lines[: start_line - 1])
    blocks: list[ColoredToken] = []
    cursor = 0
    while cursor < len(selected):
        if selected[cursor].strip() == "":
            output.append(selected[cursor])
            cursor += 1
            continue
        block_start = cursor
        while cursor < len(selected) and selected[cursor].strip() != "":
            cursor += 1
        block_lines = selected[block_start:cursor]
        raw_block = "".join(block_lines)
        visible_projection = "".join(
            line[: tex_comment_start(line)].strip() for line in block_lines
        )
        if not visible_projection:
            output.extend(block_lines)
            continue
        block_index = len(blocks)
        rgb = deterministic_rgb(block_index)
        blocks.append(
            ColoredToken(
                token_id=f"par-{block_index + 1:06d}",
                text=raw_block.rstrip("\r\n"),
                rgb=rgb,
                source_file=source_file,
                source_line=start_line + block_start,
                source_column=1,
                kind="paragraph",
            )
        )
        color_command = "{\\color[RGB]{" + ",".join(str(value) for value in rgb) + "}"
        output.append(color_command)
        output.extend(block_lines)
        # End the paragraph while the local color/group is still active. The
        # original following blank line is preserved below.
        if raw_block.endswith(("\n", "\r")):
            output.append("\\par}\n")
        else:
            output.append("\\par}")

    output.extend(lines[end_line:])
    return "".join(output), blocks


def run_compile(
    *,
    source_root: Path,
    main_tex: Path,
    build_dir: Path,
    log_path: Path,
    label: str,
    timeout_seconds: int,
    engine: str = "pdflatex",
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = {
        "pdflatex": "-pdf",
        "xelatex": "-xelatex",
        "latex_dvips_ps2pdf": "-pdfps",
    }
    if engine not in flags:
        raise ValueError(f"unsupported LaTeX engine: {engine}")
    command = [
        str(LATEXMK),
        "-norc",
        "-g",
        flags[engine],
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        f"-outdir={build_dir}",
        str(main_tex),
    ]
    started = time.monotonic()
    print(
        f"[compile_start] label={label} engine={engine} source={source_root} main={main_tex} "
        f"timeout={timeout_seconds}s",
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
                raise TimeoutError(f"{label} compile timed out after {elapsed:.1f}s")
            if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                print(
                    f"[compile_progress] label={label} elapsed={elapsed_text(elapsed)} "
                    f"log_bytes={log_path.stat().st_size if log_path.exists() else 0}",
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            time.sleep(0.25)
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{label} compile failed rc={process.returncode}\n{tail}")
    pdf_path = build_dir / main_tex.with_suffix(".pdf").name
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} compile produced no PDF: {pdf_path}")
    print(
        f"[compile_done] label={label} pdf={pdf_path} bytes={pdf_path.stat().st_size} "
        f"elapsed={elapsed_text(elapsed)}",
        flush=True,
    )
    return pdf_path


def normalize_pdf_rgb(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, (int, float)):
        channel = float(value)
        if 0.0 <= channel <= 1.000001:
            normalized = int(round(channel * 255))
        elif 0.0 <= channel <= 255.0:
            normalized = int(round(channel))
        else:
            return None
        return (normalized, normalized, normalized)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    channels = [float(channel) for channel in value]
    if all(0.0 <= channel <= 1.000001 for channel in channels):
        return tuple(int(round(channel * 255)) for channel in channels)
    if all(0.0 <= channel <= 255.0 for channel in channels):
        return tuple(int(round(channel)) for channel in channels)
    return None


def character_projection(char: dict[str, Any]) -> tuple[str, float, float, float, float]:
    return (
        str(char.get("text", "")),
        float(char["x0"]),
        float(char["top"]),
        float(char["x1"]),
        float(char["bottom"]),
    )


def compare_pdf_geometry(clean_pdf: Path, colored_pdf: Path) -> dict[str, Any]:
    page_rows: list[dict[str, Any]] = []
    global_max_shift = 0.0
    total_clean_chars = 0
    total_colored_chars = 0
    all_text_equal = True
    all_geometry_equal = True
    with pdfplumber.open(clean_pdf) as clean_document, pdfplumber.open(
        colored_pdf
    ) as colored_document:
        page_count_equal = len(clean_document.pages) == len(colored_document.pages)
        for page_index in range(min(len(clean_document.pages), len(colored_document.pages))):
            clean_chars = [character_projection(char) for char in clean_document.pages[page_index].chars]
            colored_chars = [
                character_projection(char) for char in colored_document.pages[page_index].chars
            ]
            total_clean_chars += len(clean_chars)
            total_colored_chars += len(colored_chars)
            count_equal = len(clean_chars) == len(colored_chars)
            text_equal = count_equal and all(
                left[0] == right[0] for left, right in zip(clean_chars, colored_chars)
            )
            page_max_shift = math.inf
            if count_equal:
                page_max_shift = max(
                    (
                        max(abs(left[index] - right[index]) for index in range(1, 5))
                        for left, right in zip(clean_chars, colored_chars)
                    ),
                    default=0.0,
                )
                global_max_shift = max(global_max_shift, page_max_shift)
            geometry_equal = count_equal and page_max_shift <= GEOMETRY_TOLERANCE_POINTS
            all_text_equal = all_text_equal and text_equal
            all_geometry_equal = all_geometry_equal and geometry_equal
            page_rows.append(
                {
                    "page_number": page_index + 1,
                    "clean_characters": len(clean_chars),
                    "colored_characters": len(colored_chars),
                    "character_count_equal": count_equal,
                    "character_text_equal": text_equal,
                    "max_geometry_shift_points": (
                        round(page_max_shift, 6) if math.isfinite(page_max_shift) else None
                    ),
                    "geometry_equal": geometry_equal,
                }
            )
    status = (
        "passed"
        if page_count_equal and all_text_equal and all_geometry_equal
        else "failed"
    )
    return {
        "status": status,
        "page_count_equal": page_count_equal,
        "pages_compared": len(page_rows),
        "character_text_equal": all_text_equal,
        "geometry_equal": all_geometry_equal,
        "geometry_tolerance_points": GEOMETRY_TOLERANCE_POINTS,
        "max_geometry_shift_points": round(global_max_shift, 6),
        "clean_characters": total_clean_chars,
        "colored_characters": total_colored_chars,
        "pages": page_rows,
    }


def token_text_projection(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def extract_color_alignment(
    colored_pdf: Path,
    tokens: list[ColoredToken],
    *,
    require_text_match: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token_by_rgb = {token.rgb: token for token in tokens}
    observed: dict[tuple[int, int, int], dict[int, list[dict[str, Any]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    unknown_colored_characters: collections.Counter[tuple[int, int, int]] = (
        collections.Counter()
    )
    page_character_counts: dict[int, int] = {}
    with pdfplumber.open(colored_pdf) as document:
        for page_index, page in enumerate(document.pages, start=1):
            page_character_counts[page_index] = len(page.chars)
            for char in page.chars:
                rgb = normalize_pdf_rgb(char.get("non_stroking_color"))
                if rgb in token_by_rgb:
                    observed[rgb][page_index].append(char)
                elif rgb is not None and rgb not in {(0, 0, 0), (255, 255, 255)}:
                    unknown_colored_characters[rgb] += 1
            print(
                f"[color_extract_page] page={page_index}/{len(document.pages)} "
                f"characters={len(page.chars)} matched_colors={sum(1 for rgb in observed if page_index in observed[rgb])}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    mapped = text_matched = multi_page = 0
    page_token_counts: collections.Counter[int] = collections.Counter()
    source_line_pages: dict[int, set[int]] = collections.defaultdict(set)
    for token in tokens:
        pages: list[dict[str, Any]] = []
        extracted_text_parts: list[str] = []
        for page_number, chars in sorted(observed.get(token.rgb, {}).items()):
            extracted_text = "".join(str(char.get("text", "")) for char in chars)
            extracted_text_parts.append(extracted_text)
            bbox = [
                min(float(char["x0"]) for char in chars),
                min(float(char["top"]) for char in chars),
                max(float(char["x1"]) for char in chars),
                max(float(char["bottom"]) for char in chars),
            ]
            pages.append(
                {
                    "page_number": page_number,
                    "bbox_points": [round(value, 3) for value in bbox],
                    "pdf_text": extracted_text,
                    "characters": len(chars),
                }
            )
            page_token_counts[page_number] += 1
            source_line_pages[token.source_line].add(page_number)
        is_mapped = bool(pages)
        if is_mapped:
            mapped += 1
        if len(pages) > 1:
            multi_page += 1
        extracted_projection = token_text_projection("".join(extracted_text_parts))
        source_projection = token_text_projection(token.text)
        is_text_matched: bool | None = (
            is_mapped and extracted_projection == source_projection
            if require_text_match
            else None
        )
        if is_text_matched is True:
            text_matched += 1
        row = token.as_json()
        row.update(
            {
                "status": "mapped" if is_mapped else "unmapped",
                "text_match": is_text_matched,
                "pages": pages,
            }
        )
        rows.append(row)
    coverage = mapped / len(tokens) if tokens else 0.0
    text_match_rate = (
        text_matched / len(tokens) if tokens and require_text_match else None
    )
    summary = {
        "status": (
            "passed"
            if mapped == len(tokens)
            and (not require_text_match or text_matched == len(tokens))
            else "failed"
        ),
        "tokens_total": len(tokens),
        "tokens_mapped": mapped,
        "tokens_unmapped": len(tokens) - mapped,
        "token_coverage": round(coverage, 6),
        "tokens_text_matched": text_matched if require_text_match else None,
        "token_text_match_rate": (
            round(text_match_rate, 6) if text_match_rate is not None else None
        ),
        "text_match_required": require_text_match,
        "tokens_spanning_multiple_pages": multi_page,
        "page_token_counts": {
            str(page): count for page, count in sorted(page_token_counts.items())
        },
        "source_line_pages": {
            str(line): sorted(pages) for line, pages in sorted(source_line_pages.items())
        },
        "page_character_counts": {
            str(page): count for page, count in sorted(page_character_counts.items())
        },
        "unknown_nonblack_color_character_counts": {
            "#" + "".join(f"{channel:02x}" for channel in rgb): count
            for rgb, count in unknown_colored_characters.most_common()
        },
    }
    return rows, summary


def render_pages(pdf_path: Path, pages: Iterable[int], output_dir: Path, label: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_number in sorted(set(pages)):
        target_prefix = output_dir / f"{label}_page_{page_number:04d}"
        command = [
            str(PDFTOPPM),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-r",
            str(dpi),
            "-png",
            str(pdf_path),
            str(target_prefix),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(
            f"[render_page] label={label} page={page_number} output={target_prefix}.png",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    paper_dir = args.paper_dir.resolve()
    source_root = (paper_dir / "source").resolve()
    output_dir = args.output_dir.resolve()
    main_tex = args.main_tex
    target_source = args.target_source
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; use a new path to preserve the prior pilot: {output_dir}"
        )
    if not LATEXMK.is_file() or not PDFTOPPM.is_file():
        raise FileNotFoundError(f"required tools missing: latexmk={LATEXMK} pdftoppm={PDFTOPPM}")
    for relative in (main_tex, target_source):
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    print(
        f"[start] paper={paper_dir.name} source={source_root} target={target_source} "
        f"lines={args.start_line}-{args.end_line} mode={args.mode} "
        f"drop_references={args.drop_references} output={output_dir}",
        flush=True,
    )
    clean_source = output_dir / "source_clean"
    colored_source = output_dir / "source_colored"
    clean_source.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_root, clean_source)
    reference_report: dict[str, Any] = {
        "status": "disabled",
        "files_scanned": 0,
        "totals": {},
        "files": [],
        "residuals": [],
    }
    if args.drop_references:
        reference_report = strip_references_tree(clean_source)
        atomic_write_json(output_dir / "reference_removal_report.json", reference_report)
        print(
            f"[references_removed] status={reference_report['status']} "
            f"files={reference_report['files_scanned']} "
            f"citations={reference_report['totals'].get('citation_commands', 0)} "
            f"bibliography={reference_report['totals'].get('bibliography_commands', 0)}",
            flush=True,
        )
    # The colored tree is copied from the transformed clean tree so both PDFs
    # are guaranteed to use exactly the same reference-removal policy.
    shutil.copytree(clean_source, colored_source)
    target_path = colored_source / target_source
    original = target_path.read_text(encoding="utf-8")
    colorizer = colorize_source_range if args.mode == "token" else colorize_paragraphs
    colored, tokens = colorizer(
        original,
        source_file=target_source.as_posix(),
        start_line=args.start_line,
        end_line=args.end_line,
    )
    if not tokens:
        raise RuntimeError("no eligible source tokens were colored")
    atomic_write_text(target_path, colored)
    write_jsonl(output_dir / "source_tokens.jsonl", (token.as_json() for token in tokens))
    print(
        f"[instrument_done] mode={args.mode} units={len(tokens)} "
        f"words={sum(t.kind == 'word' for t in tokens)} "
        f"punctuation={sum(t.kind == 'punctuation' for t in tokens)} "
        f"paragraphs={sum(t.kind == 'paragraph' for t in tokens)} target={target_path}",
        flush=True,
    )
    clean_build_pdf = run_compile(
        source_root=clean_source,
        main_tex=main_tex,
        build_dir=output_dir / "build_clean",
        log_path=output_dir / "logs/clean.log",
        label="clean",
        timeout_seconds=args.compile_timeout,
    )
    colored_build_pdf = run_compile(
        source_root=colored_source,
        main_tex=main_tex,
        build_dir=output_dir / "build_colored",
        log_path=output_dir / "logs/colored.log",
        label="colored",
        timeout_seconds=args.compile_timeout,
    )
    clean_pdf = output_dir / "clean.pdf"
    colored_pdf = output_dir / "colored.pdf"
    shutil.copy2(clean_build_pdf, clean_pdf)
    shutil.copy2(colored_build_pdf, colored_pdf)
    geometry = compare_pdf_geometry(clean_pdf, colored_pdf)
    print(
        f"[geometry] status={geometry['status']} pages={geometry['pages_compared']} "
        f"clean_chars={geometry['clean_characters']} colored_chars={geometry['colored_characters']} "
        f"max_shift_points={geometry['max_geometry_shift_points']}",
        flush=True,
    )
    alignment_rows, alignment = extract_color_alignment(
        colored_pdf,
        tokens,
        require_text_match=args.mode == "token",
    )
    write_jsonl(output_dir / "token_page_alignment.jsonl", alignment_rows)
    mapped_pages = sorted(
        {
            page["page_number"]
            for row in alignment_rows
            for page in row.get("pages", [])
        }
    )
    render_pages(clean_pdf, mapped_pages, output_dir / "rendered", "clean", args.dpi)
    render_pages(colored_pdf, mapped_pages, output_dir / "rendered", "colored", args.dpi)
    report = {
        "schema_version": 1,
        "status": (
            "passed"
            if geometry["status"] == "passed" and alignment["status"] == "passed"
            else "failed"
        ),
        "paper_id": paper_dir.name,
        "source_root": str(source_root),
        "main_tex": main_tex.as_posix(),
        "target_source": target_source.as_posix(),
        "source_line_range": [args.start_line, args.end_line],
        "instrumentation_mode": args.mode,
        "drop_references": args.drop_references,
        "reference_removal": reference_report,
        "clean_pdf": str(clean_pdf),
        "colored_pdf": str(colored_pdf),
        "geometry_validation": geometry,
        "color_alignment": alignment,
        "mapped_pages": mapped_pages,
        "limitations": (
            [
                "only conservative literal source tokens in the requested range are colored",
                "visible zero-argument macro expansions are not yet assigned source token IDs",
                "math contents are intentionally skipped in token mode",
                "semantic Markdown serialization is outside this positioning pilot",
            ]
            if args.mode == "token"
            else [
                "paragraph mode proves source-block-to-page/bbox provenance, not token identity",
                "links or macros that set their own colors can override the paragraph color",
                "semantic Markdown serialization is outside this positioning pilot",
            ]
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    print(
        f"[final] status={report['status']} tokens={alignment['tokens_mapped']}/"
        f"{alignment['tokens_total']} text_matched={alignment['tokens_text_matched']} "
        f"pages={mapped_pages} geometry={geometry['status']} "
        f"elapsed={elapsed_text(time.monotonic() - started)} output={output_dir}",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
