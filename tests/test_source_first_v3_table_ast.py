from __future__ import annotations

import pytest

from arxiv_source_first_v3.ast_ir import MathMacroDefinition
from arxiv_source_first_v3.table_ast import (
    TableAstError,
    load_table_verifier_fragments,
    parse_strict_table,
)


def _parse(source: str):
    return parse_strict_table(
        source,
        start=0,
        end=len(source),
        source_id="table.tex",
    )


def test_simple_table_is_source_serialized_as_attribute_free_html() -> None:
    source = r"""\begin{tabular}{cc}
A & B \\
1 & 2
\end{tabular}"""
    table = _parse(source)

    assert table.column_count == 2
    assert table.visible_text == "A B 1 2"
    assert table.html == """<table>
  <tbody>
    <tr>
      <td>A</td>
      <td>B</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>"""
    assert "data-" not in table.html


def test_booktabs_proves_header_and_inline_styles() -> None:
    source = r"""\begin{tabular}{cc}
\toprule
\textbf{Method} & $F_1$ \\
\midrule
Ours & \emph{91.2} \\
\bottomrule
\end{tabular}"""
    table = _parse(source)

    assert [row.header for row in table.rows] == [True, False]
    assert "<th><strong>Method</strong></th>" in table.html
    assert "<th>$F_1$</th>" in table.html
    assert "<td><em>91.2</em></td>" in table.html
    fragments = load_table_verifier_fragments(dict(table.to_metadata()))
    assert ("F_1", False) in fragments
    assert ("Method", True) in fragments
    assert ("91.2", True) in fragments


def test_booktabs_cmidrule_parenthesized_trim_and_body_midrules_are_supported() -> None:
    source = r"""\begin{tabular}{lcc}
\toprule
Model & A & B \\
\cmidrule(lr){2-3}
One & 1 & 2 \\
\midrule
Two & 3 & 4 \\
\bottomrule
\end{tabular}"""
    table = _parse(source)

    assert table.column_count == 3
    assert len(table.rows) == 3
    assert "<td>Two</td>" in table.html


def test_long_text_cell_preserves_paragraphs_and_nested_lists_as_html() -> None:
    source = r"""\begin{tabular}{|p{0.9\linewidth}|}
\hline
Intro text. \par
\begin{enumerate}[label=\alph*.]
\item First rule.
\item Second rule:
  \begin{itemize}
  \item Nested detail.
  \end{itemize}
\end{enumerate}
Closing text. \\ \hline
\end{tabular}"""
    table = _parse(source)

    assert table.column_count == 1
    assert "Intro text.\n<br><br>" in table.html
    assert (
        '<ol type="a"><li>First rule.</li><li>Second rule:\n'
        "<ul><li>Nested detail.</li></ul></li></ol>"
    ) in table.html
    assert "Closing text." in table.html
    assert table.visible_text == (
        "Intro text. First rule. Second rule: Nested detail. Closing text."
    )


def test_literal_multirow_and_multicolumn_are_structural_attributes_only() -> None:
    source = r"""\begin{tabular}{ccc}
\multirow{2}{*}{Ours} & A & B \\
 & \multicolumn{2}{c}{Stable} \\
\end{tabular}"""
    table = _parse(source)

    assert table.column_count == 3
    assert table.html == """<table>
  <tbody>
    <tr>
      <td rowspan="2">Ours</td>
      <td>A</td>
      <td>B</td>
    </tr>
    <tr>
      <td colspan="2">Stable</td>
    </tr>
  </tbody>
</table>"""
    assert "data-source" not in table.html


def test_multirow_middle_placeholders_are_omitted() -> None:
    source = r"""\begin{tabular}{ccc}
\multirow{2}{*}{Left} & Top & \multirow{2}{*}{Right} \\
 & Middle & \\
\end{tabular}"""

    table = _parse(source)

    assert table.html == """<table>
  <tbody>
    <tr>
      <td rowspan="2">Left</td>
      <td>Top</td>
      <td rowspan="2">Right</td>
    </tr>
    <tr>
      <td>Middle</td>
    </tr>
  </tbody>
</table>"""


def test_multirow_missing_placeholder_is_rejected() -> None:
    source = r"""\begin{tabular}{cc}
\multirow{2}{*}{A} & B \\
C & D
\end{tabular}"""

    with pytest.raises(TableAstError, match="rowspan placeholder"):
        _parse(source)


@pytest.mark.parametrize(
    "source",
    [
        r"\begin{tabular}{c}\multirow{2}{*}{A}\end{tabular}",
        r"\begin{tabular}{cc}A & B \\\multirow{2}{*}{C} & D\end{tabular}",
    ],
)
def test_rowspan_must_end_within_source_rows(source: str) -> None:
    with pytest.raises(TableAstError, match="final table row"):
        _parse(source)


def test_rowspan_must_not_cross_thead_tbody_boundary() -> None:
    source = r"""\begin{tabular}{cc}
\toprule
\multirow{2}{*}{Head} & H2 \\
\midrule
 & Body \\
\bottomrule
\end{tabular}"""

    with pytest.raises(TableAstError, match="section boundary"):
        _parse(source)


def _table_with_spec(spec: str, cell_count: int) -> str:
    cells = " & ".join(f"Cell{index}" for index in range(cell_count))
    return "\\begin{tabular}{" + spec + "}" + cells + "\\end{tabular}"


@pytest.mark.parametrize(
    ("spec", "column_count"),
    [
        ("lcrX", 4),
        ("|l|c||r|", 3),
        (r"p{0.9\linewidth}m{2cm}b{.4\textwidth}", 3),
        ("*{2}{c}", 2),
        ("*{2}{lcr}", 6),
        (r"*{2}{|p{1em}|}", 2),
        (r">{\raggedright\arraybackslash}p{2cm}|", 1),
        (r"@{\hspace{1em}}c!{\hspace{0pt}}", 1),
        (r"@{\extracolsep{\fill}}c", 1),
    ],
)
def test_common_literal_column_specs_are_counted(
    spec: str, column_count: int
) -> None:
    table = _parse(_table_with_spec(spec, column_count))

    assert table.column_count == column_count


@pytest.mark.parametrize(
    ("spec", "cell_count"),
    [
        ("c", 2),
        ("ccc", 2),
        ("cc", 3),
        ("q", 1),
        ("*{2}{q}", 2),
        (r"@{visible}c", 1),
        (r">{\bfseries}c", 1),
        (r"p{\textcolor{red}}", 1),
    ],
)
def test_unknown_or_mismatched_column_specs_fail_closed(
    spec: str, cell_count: int
) -> None:
    with pytest.raises(TableAstError):
        _parse(_table_with_spec(spec, cell_count))


def test_source_macro_formula_is_preserved_but_not_literal_pdf_verifier_text() -> None:
    source = r"""\begin{tabular}{c}
\method
\end{tabular}"""
    table = parse_strict_table(
        source,
        start=0,
        end=len(source),
        source_id="table-macro.tex",
        math_macros={
            "method": MathMacroDefinition(
                "method", 0, r"\textsc{LinuxFL}\ensuremath{^{+}}\xspace"
            )
        },
    )

    assert "LINUXFL$^{+}$" in table.html
    assert load_table_verifier_fragments(dict(table.to_metadata())) == (
        ("LINUXFL", True),
        ("^{+}", False),
    )


@pytest.mark.parametrize(
    "source",
    [
        r"\begin{longtable}{c}A\\\end{longtable}",
        r"\begin{tabular}{c}\begin{tabular}{c}A\end{tabular}\end{tabular}",
        r"\begin{tabular}{c}\unknown{A}\end{tabular}",
        r"\begin{tabular}{cc}A & B \\ C\end{tabular}",
    ],
)
def test_ambiguous_or_unsupported_tables_fail_closed(source: str) -> None:
    with pytest.raises(TableAstError):
        _parse(source)
