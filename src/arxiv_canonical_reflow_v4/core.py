"""Source-derived canonical reflow primitives.

V4 deliberately avoids recovering Markdown from a PDF.  A safe subset of the
existing immutable LaTeX document AST is serialized twice: once to Markdown
ground truth and once to canonical LaTeX.  The latter is compiled only to
produce an image and to reject pages whose rendered text does not agree with
the source-derived verifier text.

The module does not edit an arXiv source tree and does not import a V1/V2 page
pipeline.  It may consume the frozen V3 source AST because that AST is the
project's source parser, not its original-page placement algorithm.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser

from arxiv_source_first_v3.ast_ir import SourceAtom, reconstruct_markdown
from arxiv_source_first_v3.document_ast import DocumentAst, DocumentBlockNode
from arxiv_source_first_v3.table_ast import is_serializable_table_metadata

PIPELINE_VERSION = "arxiv_canonical_reflow_v4_8_direct_edit"


class CanonicalReflowError(ValueError):
    """A source block cannot be represented by the canonical renderer."""


@dataclass(frozen=True, slots=True)
class CanonicalBlock:
    block_id: str
    node_id: str
    kind: str
    markdown: str
    latex: str
    verifier_text: str
    weight: int
    source_char_span: tuple[int, int]
    source_files: tuple[str, ...]
    has_table: bool = False


@dataclass(frozen=True, slots=True)
class CanonicalPage:
    page_id: str
    paper_id: str
    ordinal: int
    layout: str
    blocks: tuple[CanonicalBlock, ...]

    @property
    def markdown(self) -> str:
        return "\n\n".join(block.markdown for block in self.blocks).strip()

    @property
    def verifier_text(self) -> str:
        return "\n".join(
            block.verifier_text for block in self.blocks if block.verifier_text.strip()
        ).strip()

    @property
    def has_table(self) -> bool:
        return any(block.has_table for block in self.blocks)


@dataclass(frozen=True, slots=True)
class PageBuildResult:
    page_id: str
    status: str
    reason: str | None
    pdf: str | None
    image: str | None
    markdown: str
    layout: str
    has_table: bool
    verifier_recall: float | None
    verifier_precision: float | None


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}

_STYLE_WRAPPERS = {
    "strong": (r"\textbf{", "}"),
    "em": (r"\emph{", "}"),
    "body_em": (r"\emph{", "}"),
    "code": (r"\texttt{", "}"),
    "sup": (r"\textsuperscript{", "}"),
    "smallcaps": (r"\textsc{", "}"),
}


def escape_latex_text(value: str) -> str:
    """Escape literal visible text for canonical XeLaTeX."""

    return "".join(_LATEX_ESCAPES.get(char, char) for char in value)


def _style_latex(value: str, styles: Sequence[str]) -> str:
    rendered = value
    for style in reversed(tuple(styles)):
        wrapper = _STYLE_WRAPPERS.get(style)
        if wrapper is None:
            raise CanonicalReflowError(f"unsupported inline style: {style}")
        rendered = wrapper[0] + rendered + wrapper[1]
    return rendered


def _balanced_math_fragment(markdown: str) -> str | None:
    value = markdown.strip()
    if value.startswith("$$") and value.endswith("$$") and len(value) >= 4:
        return value[2:-2].strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        return value[1:-1]
    return None


def _render_source_macro_fragment(markdown: str) -> str:
    """Render a conservative Markdown subset emitted by ``ast_ir``.

    Source macros may combine literal text with inline formulas.  Formatting
    is intentionally restricted; unknown markup is escaped as text rather
    than executed as TeX.
    """

    wrappers = {
        "strong": r"\textbf{",
        "em": r"\emph{",
        "code": r"\texttt{",
        "sup": r"\textsuperscript{",
    }

    def tagged(start: int) -> tuple[str, int] | None:
        for tag, opening in wrappers.items():
            opener = f"<{tag}>"
            closer = f"</{tag}>"
            if not markdown.startswith(opener, start):
                continue
            end = markdown.find(closer, start + len(opener))
            if end < 0:
                return None
            inner = markdown[start + len(opener) : end]
            return opening + _render_source_macro_fragment(inner) + "}", end + len(
                closer
            )
        return None

    output: list[str] = []
    cursor = 0
    while cursor < len(markdown):
        tag = tagged(cursor)
        if tag is not None:
            output.append(tag[0])
            cursor = tag[1]
            continue
        if markdown[cursor] == "$":
            closing = markdown.find("$", cursor + 1)
            if closing > cursor + 1:
                output.append("$" + markdown[cursor + 1 : closing] + "$")
                cursor = closing + 1
                continue
        if markdown.startswith("**", cursor):
            closing = markdown.find("**", cursor + 2)
            if closing > cursor + 2:
                inner = markdown[cursor + 2 : closing]
                output.append(r"\textbf{" + _render_source_macro_fragment(inner) + "}")
                cursor = closing + 2
                continue
        if markdown[cursor] == "*":
            closing = markdown.find("*", cursor + 1)
            if closing > cursor + 1:
                inner = markdown[cursor + 1 : closing]
                output.append(r"\emph{" + _render_source_macro_fragment(inner) + "}")
                cursor = closing + 1
                continue
        output.append(escape_latex_text(markdown[cursor]))
        cursor += 1
    return "".join(output)


def _atom_latex(atom: SourceAtom) -> str:
    if atom.is_whitespace:
        return " "
    if atom.kind == "math":
        math = _balanced_math_fragment(atom.markdown_fragment)
        if math is None:
            raise CanonicalReflowError("math atom lacks balanced Markdown delimiters")
        value = "$" + math + "$"
    elif atom.raw_source.startswith(r"\\") and atom.visible_text == "\n":
        value = r"\\ "
    elif atom.kind == "source_macro":
        value = _render_source_macro_fragment(atom.markdown_fragment)
    elif atom.kind in {
        "text",
        "reference",
        "url",
        "literal",
        "footnote_callout",
    }:
        value = escape_latex_text(atom.visible_text)
    else:
        if not atom.visible_text:
            return ""
        value = escape_latex_text(atom.visible_text)
    return _style_latex(value, atom.style_stack)


def _inline_ir_latex(node: DocumentBlockNode) -> str:
    ir = node.inline_ir
    if ir is None:
        raise CanonicalReflowError("block has no inline IR")
    if ir.opaque_atoms:
        raise CanonicalReflowError("block contains opaque LaTeX")
    if ir.footnotes:
        raise CanonicalReflowError("block contains unresolved footnotes")
    rendered = "".join(_atom_latex(atom) for atom in ir.atoms if not atom.footnote_path)
    rendered = re.sub(r"[ \t\r\f\v]+", " ", rendered).strip()
    if not rendered:
        raise CanonicalReflowError("block has no canonical visible content")
    return rendered


def _inline_verifier_text(node: DocumentBlockNode) -> str:
    if node.inline_ir is None:
        return ""
    chunks: list[str] = []
    for atom in node.inline_ir.atoms:
        if atom.footnote_path or atom.is_whitespace:
            if chunks and not chunks[-1].endswith(" "):
                chunks.append(" ")
            continue
        fragments = atom.verifier_fragments
        if fragments:
            chunks.extend(text for text, verifiable in fragments if verifiable)
        elif atom.kind in {"text", "reference", "source_macro"}:
            chunks.append(atom.visible_text)
    return " ".join("".join(chunks).split())


@dataclass(slots=True)
class _HtmlCell:
    tag: str
    attrs: dict[str, str]
    parts: list[str]


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_HtmlCell]] = []
        self._row: list[_HtmlCell] | None = None
        self._cell: _HtmlCell | None = None
        self._format_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            if self._row is not None:
                raise CanonicalReflowError("nested table row")
            self._row = []
        elif tag in {"th", "td"}:
            if self._row is None or self._cell is not None:
                raise CanonicalReflowError("malformed table cell")
            self._cell = _HtmlCell(
                tag=tag,
                attrs={key: value or "" for key, value in attrs},
                parts=[],
            )
        elif self._cell is not None and tag in {"strong", "em", "code", "sup"}:
            self._cell.parts.append(
                {
                    "strong": r"\textbf{",
                    "em": r"\emph{",
                    "code": r"\texttt{",
                    "sup": r"\textsuperscript{",
                }[tag]
            )
            self._format_stack.append(tag)
        elif self._cell is not None and tag == "br":
            self._cell.parts.append(r"\newline{}")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"strong", "em", "code", "sup"} and self._cell is not None:
            if not self._format_stack or self._format_stack[-1] != tag:
                raise CanonicalReflowError("unbalanced table inline markup")
            self._format_stack.pop()
            self._cell.parts.append("}")
        elif tag in {"th", "td"}:
            if self._row is None or self._cell is None or self._cell.tag != tag:
                raise CanonicalReflowError("malformed table cell ending")
            if self._format_stack:
                raise CanonicalReflowError("unclosed table inline markup")
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr":
            if self._row is None or self._cell is not None:
                raise CanonicalReflowError("malformed table row ending")
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is None:
            return
        self._cell.parts.append(_render_text_with_inline_math(data))

    def close(self) -> None:
        super().close()
        if self._row is not None or self._cell is not None or self._format_stack:
            raise CanonicalReflowError("unterminated table markup")


def _render_text_with_inline_math(value: str) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(value):
        if value[cursor] == "$":
            closing = value.find("$", cursor + 1)
            if closing > cursor + 1:
                output.append("$" + value[cursor + 1 : closing] + "$")
                cursor = closing + 1
                continue
            # A literal unmatched dollar sign is text, not an executable math
            # delimiter.  Escaping and advancing also guarantees termination.
            output.append(r"\$")
            cursor += 1
            continue
        next_math = value.find("$", cursor)
        end = len(value) if next_math < 0 else next_math
        literal = value[cursor:end]
        for piece in re.split(r"(\s+)", literal):
            escaped = escape_latex_text(piece)
            if piece and not piece.isspace() and len(piece) >= 18:
                escaped = r"\seqsplit{" + escaped + "}"
            output.append(escaped)
        cursor = end
    return "".join(output)


def _positive_span(attrs: Mapping[str, str], name: str) -> int:
    raw = attrs.get(name, "1")
    if not raw.isdigit() or int(raw) < 1:
        raise CanonicalReflowError(f"invalid table {name}")
    return int(raw)


def render_table_html_to_latex(table_html: str) -> str:
    """Render the strict clean table HTML subset to canonical LaTeX."""

    parser = _TableHtmlParser()
    parser.feed(table_html)
    parser.close()
    if not parser.rows:
        raise CanonicalReflowError("table has no rows")
    column_count = max(
        sum(_positive_span(cell.attrs, "colspan") for cell in row)
        for row in parser.rows
    )
    if column_count < 1 or column_count > 16:
        raise CanonicalReflowError("table column count is outside canonical limit")

    active_rowspans = [0] * column_count
    latex_rows: list[str] = []
    for row in parser.rows:
        slots: list[str | None] = [None] * column_count
        for index, remaining in enumerate(active_rowspans):
            if remaining > 0:
                slots[index] = ""
        cursor = 0
        for cell in row:
            while cursor < column_count and slots[cursor] is not None:
                cursor += 1
            colspan = _positive_span(cell.attrs, "colspan")
            rowspan = _positive_span(cell.attrs, "rowspan")
            if cursor + colspan > column_count:
                raise CanonicalReflowError("table spans exceed inferred columns")
            if any(
                slots[index] is not None for index in range(cursor, cursor + colspan)
            ):
                raise CanonicalReflowError("overlapping table spans")
            body = "".join(cell.parts).strip()
            if cell.tag == "th" and body:
                body = r"\textbf{" + body + "}"
            if rowspan > 1:
                body = rf"\multirow{{{rowspan}}}{{*}}{{{body}}}"
            if colspan > 1:
                body = rf"\multicolumn{{{colspan}}}{{c}}{{{body}}}"
            slots[cursor] = body
            for index in range(cursor + 1, cursor + colspan):
                slots[index] = "__SPAN__"
            if rowspan > 1:
                for index in range(cursor, cursor + colspan):
                    active_rowspans[index] = max(active_rowspans[index], rowspan)
            cursor += colspan
        cells = [value for value in slots if value != "__SPAN__"]
        latex_rows.append(" & ".join(value or "" for value in cells) + r" \\")
        active_rowspans = [max(0, value - 1) for value in active_rowspans]

    columns = " ".join([">{\\raggedright\\arraybackslash}X"] * column_count)
    size = r"\scriptsize" if column_count >= 6 else r"\small"
    return "\n".join(
        [
            r"\begin{center}",
            size,
            rf"\begin{{tabularx}}{{\linewidth}}{{{columns}}}",
            r"\toprule",
            *[
                row + ("\n" + r"\midrule" if index == 0 else "")
                for index, row in enumerate(latex_rows)
            ],
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{center}",
        ]
    )


def _node_is_rejected(document: DocumentAst, node: DocumentBlockNode) -> bool:
    for rejection in document.rejections:
        if rejection.node_id == node.node_id:
            return True
        if (
            rejection.span.char_start < node.span.char_end
            and node.span.char_start < rejection.span.char_end
        ):
            return True
    return False


def _list_prefix(document: DocumentAst, node: DocumentBlockNode) -> tuple[str, str]:
    if node.parent_node_id is None:
        return "", ""
    parent = document.get_node(node.parent_node_id)
    if parent.kind != "list_item_region" or not parent.child_node_ids:
        return "", ""
    if parent.child_node_ids[0] != node.node_id:
        return "", ""
    list_kind = dict(parent.metadata).get("list_kind")
    if list_kind == "itemize":
        return "- ", r"\textbullet{} "
    if list_kind != "enumerate" or parent.parent_node_id is None:
        return "", ""
    container = document.get_node(parent.parent_node_id)
    item_ids = [
        child_id
        for child_id in container.child_node_ids
        if document.get_node(child_id).kind == "list_item_region"
    ]
    try:
        index = item_ids.index(parent.node_id) + 1
    except ValueError:
        return "", ""
    prefix = f"{index}. "
    return prefix, escape_latex_text(prefix)


def _source_files(node: DocumentBlockNode) -> tuple[str, ...]:
    return tuple(sorted({origin.source_path for origin in node.origins}))


def _block_id(paper_id: str, node: DocumentBlockNode) -> str:
    digest = hashlib.sha256(
        f"{paper_id}:{node.node_id}:{node.span.char_start}:{node.span.char_end}".encode()
    ).hexdigest()[:16]
    return f"b_{digest}"


def _inline_block(
    document: DocumentAst,
    node: DocumentBlockNode,
    *,
    paper_id: str,
) -> CanonicalBlock:
    if node.inline_ir is None:
        raise CanonicalReflowError("inline block lacks source IR")
    markdown = reconstruct_markdown(node.inline_ir).strip()
    latex = _inline_ir_latex(node)
    prefix_md, prefix_tex = _list_prefix(document, node)
    markdown = prefix_md + markdown
    latex = prefix_tex + latex

    if node.kind == "heading":
        level = node.heading_level or 1
        if not 1 <= level <= 6:
            raise CanonicalReflowError("heading level is outside Markdown range")
        markdown = f"{'#' * level} {markdown}"
        command = (
            "section" if level == 1 else "subsection" if level == 2 else "subsubsection"
        )
        latex = rf"\{command}*{{{latex}}}"
    elif node.kind == "frontmatter" and node.command_name == "title":
        markdown = f"# {markdown}"
        latex = rf"\begin{{center}}\LARGE\bfseries {latex}\end{{center}}"
    elif node.kind == "frontmatter" and node.command_name == "author":
        latex = rf"\begin{{center}}{latex}\end{{center}}"
    elif node.kind == "caption":
        # A plain source-derived line avoids inventing a compiler caption number.
        latex = rf"\par\small\emph{{{latex}}}\par"
    elif node.kind == "display_math":
        math = _balanced_math_fragment(markdown)
        if math is None:
            raise CanonicalReflowError("display formula lacks canonical delimiters")
        latex = "\\[\n" + math + "\n\\]"
    else:
        latex = latex + r"\par"

    verifier = _inline_verifier_text(node)
    if node.kind in {"heading", "frontmatter", "caption"} and not verifier:
        verifier = " ".join(
            atom.visible_text
            for atom in node.inline_ir.atoms
            if atom.visible_text and not atom.footnote_path
        )
    weight = max(1, len(verifier) or len(markdown))
    return CanonicalBlock(
        block_id=_block_id(paper_id, node),
        node_id=node.node_id,
        kind=node.kind,
        markdown=markdown,
        latex=latex,
        verifier_text=verifier,
        weight=weight,
        source_char_span=node.span.char_span,
        source_files=_source_files(node),
    )


def _table_block(node: DocumentBlockNode, *, paper_id: str) -> CanonicalBlock:
    metadata = dict(node.metadata)
    if not is_serializable_table_metadata(metadata):
        raise CanonicalReflowError("table lacks strict source-derived HTML")
    table_html = metadata["table_html"]
    latex = render_table_html_to_latex(table_html)
    verifier = metadata["table_visible_text"]
    return CanonicalBlock(
        block_id=_block_id(paper_id, node),
        node_id=node.node_id,
        kind="table",
        markdown=table_html,
        latex=latex,
        verifier_text=verifier,
        weight=max(300, len(verifier) * 2),
        source_char_span=node.span.char_span,
        source_files=_source_files(node),
        has_table=True,
    )


def blocks_from_document(
    document: DocumentAst,
    *,
    paper_id: str,
) -> tuple[tuple[CanonicalBlock, ...], tuple[dict[str, object], ...]]:
    """Convert independently safe AST leaves and audit every skipped leaf."""

    accepted: list[CanonicalBlock] = []
    rejected: list[dict[str, object]] = []
    allowed = {
        "paragraph",
        "heading",
        "display_math",
        "caption",
        "frontmatter",
        "table",
    }
    for node in document.source_ordered_leaf_nodes:
        if node.kind not in allowed:
            rejected.append(
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "reason": "unsupported_kind",
                }
            )
            continue
        if _node_is_rejected(document, node):
            rejected.append(
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "reason": "overlapping_ast_rejection",
                }
            )
            continue
        try:
            block = (
                _table_block(node, paper_id=paper_id)
                if node.kind == "table"
                else _inline_block(document, node, paper_id=paper_id)
            )
        except Exception as exc:  # noqa: BLE001 - every unsafe leaf is audited independently
            rejected.append(
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if block.markdown.strip() and block.latex.strip():
            accepted.append(block)
    return tuple(accepted), tuple(rejected)


def bundle_blocks(
    blocks: Sequence[CanonicalBlock],
) -> tuple[tuple[CanonicalBlock, ...], ...]:
    """Keep source elements that must not straddle canonical pages together.

    Tables and their adjacent captions form one indivisible bundle.  A heading
    is then attached to the following bundle so the dense packer cannot leave
    an orphan heading at the bottom of a page.  No source text is rewritten or
    reordered by this operation.
    """

    adjacent: list[tuple[CanonicalBlock, ...]] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if (
            block.kind == "caption"
            and index + 1 < len(blocks)
            and blocks[index + 1].has_table
        ):
            adjacent.append((block, blocks[index + 1]))
            index += 2
            continue
        if (
            block.has_table
            and index + 1 < len(blocks)
            and blocks[index + 1].kind == "caption"
        ):
            adjacent.append((block, blocks[index + 1]))
            index += 2
            continue
        adjacent.append((block,))
        index += 1

    bundled: list[tuple[CanonicalBlock, ...]] = []
    index = 0
    while index < len(adjacent):
        current = adjacent[index]
        if (
            len(current) == 1
            and current[0].kind == "heading"
            and index + 1 < len(adjacent)
        ):
            bundled.append((*current, *adjacent[index + 1]))
            index += 2
            continue
        bundled.append(current)
        index += 1
    return tuple(bundled)


def pack_blocks(
    blocks: Sequence[CanonicalBlock],
    *,
    paper_id: str,
    target_weight: int = 4200,
    two_column_rate: float = 0.35,
) -> tuple[CanonicalPage, ...]:
    """Pack source-ordered blocks into deterministic canonical page candidates."""

    if target_weight < 200:
        raise CanonicalReflowError("target_weight is too small")
    if not 0.0 <= two_column_rate <= 1.0:
        raise CanonicalReflowError("two_column_rate must be in [0, 1]")
    groups: list[list[CanonicalBlock]] = []
    current: list[CanonicalBlock] = []
    weight = 0
    for bundle in bundle_blocks(blocks):
        bundle_weight = sum(block.weight for block in bundle)
        if current and weight + bundle_weight > target_weight:
            groups.append(current)
            current = []
            weight = 0
        current.extend(bundle)
        weight += bundle_weight
    if current:
        groups.append(current)

    pages: list[CanonicalPage] = []
    threshold = int(two_column_rate * 10_000)
    for ordinal, group in enumerate(groups, start=1):
        digest = int(
            hashlib.sha256(f"{paper_id}:{ordinal}".encode()).hexdigest()[:8], 16
        )
        # Canonical table HTML remains valid in either layout, but a table's
        # long scientific tokens cannot always break safely inside a narrow
        # minipage.  Keep every table page full-width and use two columns for
        # prose/formula pages only.
        has_wide_table = any(block.has_table for block in group)
        group_weight = sum(block.weight for block in group)
        two_column_eligible = (
            len(group) >= 4
            and group_weight >= int(target_weight * 0.75)
            and not has_wide_table
        )
        layout = (
            "two_column"
            if two_column_eligible and digest % 10_000 < threshold
            else "one_column"
        )
        page_id = f"{paper_id}_reflow_{ordinal:04d}"
        pages.append(
            CanonicalPage(
                page_id=page_id,
                paper_id=paper_id,
                ordinal=ordinal,
                layout=layout,
                blocks=tuple(group),
            )
        )
    return tuple(pages)


def _split_two_columns(
    blocks: Sequence[CanonicalBlock],
) -> tuple[Sequence[CanonicalBlock], Sequence[CanonicalBlock]]:
    total = sum(block.weight for block in blocks)
    running = 0
    candidates: list[tuple[float, int]] = []
    for index, block in enumerate(blocks[:-1], start=1):
        running += block.weight
        following = blocks[index]
        # Keep headings with their following content and captions with the
        # table immediately before them.
        if block.kind == "heading" or following.kind == "caption":
            continue
        candidates.append((abs(running - total / 2), index))
    split = min(candidates)[1] if candidates else max(1, len(blocks) // 2)
    return blocks[:split], blocks[split:]


def _blocks_latex(blocks: Iterable[CanonicalBlock]) -> str:
    return "\n\n".join(block.latex for block in blocks)


def build_page_tex(page: CanonicalPage) -> str:
    """Build one standalone canonical XeLaTeX document."""

    if not page.blocks:
        raise CanonicalReflowError("page has no blocks")
    if page.layout not in {"one_column", "two_column"}:
        raise CanonicalReflowError(f"unsupported page layout: {page.layout}")
    if page.layout == "two_column" and len(page.blocks) > 1:
        left, right = _split_two_columns(page.blocks)
        body = "\n".join(
            [
                r"\noindent\begin{minipage}[t]{0.485\textwidth}",
                r"\vspace{0pt}",
                _blocks_latex(left),
                r"\end{minipage}\hfill%",
                r"\begin{minipage}[t]{0.485\textwidth}",
                r"\vspace{0pt}",
                _blocks_latex(right),
                r"\end{minipage}",
            ]
        )
    else:
        body = _blocks_latex(page.blocks)
    return "\n".join(  # noqa: FLY002 - raw TeX is clearer as a line list
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[letterpaper,margin=0.72in]{geometry}",
            r"\usepackage{fontspec}",
            r"\usepackage{amsmath,amssymb,booktabs,array,tabularx,multirow,seqsplit}",
            r"\usepackage{microtype}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.58em}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.16}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            body,
            r"\end{document}",
            "",
        ]
    )


_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def _tokens(value: str) -> list[str]:
    normalized = (
        html.unescape(value)
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("–", "-")
        .replace("—", "-")
    )
    return [token.casefold() for token in _WORD.findall(normalized)]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for value in right:
        current = [0]
        for index, candidate in enumerate(left, start=1):
            if candidate == value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def verify_rendered_text(expected: str, observed: str) -> tuple[bool, float, float]:
    """Reject-only token-order check; observed PDF text never enters GT."""

    expected_tokens = _tokens(expected)
    observed_tokens = _tokens(observed)
    if not expected_tokens:
        # Formula-only blocks intentionally have no PDF-text verifier runs.
        # Their exact TeX body is shared by GT and the canonical compiler, so
        # successful one-page compilation is the applicable proof.
        return True, 1.0, 1.0
    match = _lcs_length(expected_tokens, observed_tokens)
    recall = match / len(expected_tokens)
    precision = match / max(1, len(observed_tokens))
    # Canonical pages have no source-provided preamble, header, footer, float,
    # or automatic numbering.  Extra PDF tokens here are overwhelmingly math
    # extraction artefacts, so recall is the safety gate and precision remains
    # an audit metric rather than a false rejection source.
    return recall >= 0.975, recall, precision
