from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_arxiv_page_markdown_gt.py"
SPEC = importlib.util.spec_from_file_location("arxiv_page_gt", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VERIFY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_arxiv_page_markdown_strict_v2.py"
)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_arxiv_page_gt_strict_v2", VERIFY_SCRIPT
)
assert VERIFY_SPEC and VERIFY_SPEC.loader
VERIFY_MODULE = importlib.util.module_from_spec(VERIFY_SPEC)
sys.modules[VERIFY_SPEC.name] = VERIFY_MODULE
VERIFY_SPEC.loader.exec_module(VERIFY_MODULE)


class ArxivPageMarkdownGtTest(unittest.TestCase):
    @staticmethod
    def strict_line(
        text: str,
        line_id: str,
        index: int,
        bbox: list[float] | None = None,
    ) -> object:
        return MODULE.PageNode(
            "text",
            text,
            bbox or [50, 50 + index * 12, 550, 60 + index * 12],
            10,
            line_id=line_id,
            origin_page=1,
            origin_order=index,
            claimed_line_ids=[line_id],
        )

    def test_table_is_structural_html_with_spans(self) -> None:
        raw = r"""
\begin{table}
\caption{Ablation $F_1$ results}
\begin{tabular}{lcc}
\toprule
Method & P & R \\
\midrule
\multirow{2}{*}{Ours} & 91.2 & 90.1 \\
 & \multicolumn{2}{c}{stable} \\
\bottomrule
\end{tabular}
\end{table}
"""
        rendered = MODULE.render_table(raw, "t1")
        value = rendered.table_html
        self.assertTrue(value.startswith("<table>"))
        self.assertNotIn("data-table-id", value)
        self.assertNotIn("data-source", value)
        self.assertNotIn("<caption", value)
        self.assertEqual(rendered.caption_markdown, "Ablation $F_1$ results")
        self.assertEqual(rendered.parse_status, "parsed")
        self.assertIn("<thead>", value)
        self.assertIn("<tbody>", value)
        self.assertIn('rowspan="2"', value)
        self.assertIn('colspan="2"', value)
        self.assertNotIn("|---", value)

    def test_complete_paper_checkpoint_fast_resume_requires_every_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper = {"stem": "2608.00001v1"}
            pages_dir = root / "pages"
            pages_dir.mkdir(parents=True)
            for name in (
                "source_blocks.jsonl",
                "source_paragraphs.jsonl",
                "inline_source_blocks.jsonl",
                "inline_source_rejections.jsonl",
                "author_superscript_plans.jsonl",
            ):
                (root / name).write_text("", encoding="utf-8")
            (root / "paper_summary.json").write_text(
                json.dumps(
                    {
                        "stem": paper["stem"],
                        "pages_emitted": 2,
                        "compile": {"status": "passed"},
                    }
                ),
                encoding="utf-8",
            )
            for page_number in (1, 2):
                stem = f"page_{page_number:04d}"
                markdown = f"page {page_number}\n"
                (pages_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
                (pages_dir / f"{stem}.png").write_bytes(b"png")
                relative = Path("papers") / paper["stem"] / "pages" / stem
                (pages_dir / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "data_id": f"{paper['stem']}_page_{page_number:04d}",
                            "strict_text_contract_version": MODULE.STRICT_TEXT_CONTRACT_VERSION,
                            "source_paragraph_contract_version": MODULE.SOURCE_PARAGRAPH_CONTRACT_VERSION,
                            "footnote_representation": MODULE.FOOTNOTE_REPRESENTATION,
                            "author_superscript_contract_version": MODULE.AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
                            "image": relative.with_suffix(".png").as_posix(),
                            "markdown": relative.with_suffix(".md").as_posix(),
                            "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                        }
                    ),
                    encoding="utf-8",
                )
            loaded = MODULE.load_complete_paper_checkpoint(
                root,
                paper=paper,
                page_limit=2,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded[0]), 2)
            (pages_dir / "page_0002.md").unlink()
            self.assertIsNone(
                MODULE.load_complete_paper_checkpoint(
                    root,
                    paper=paper,
                    page_limit=2,
                )
            )

    def test_author_affiliation_markers_are_source_confirmed_html_sup(self) -> None:
        plan = MODULE.author_plan_from_raw(
            r"\textbf{Ada Fang}$^{1,2\dagger}$ Robert Lee$^2$",
            source_file=Path("authors.tex"),
            start_line=10,
            end_line=10,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        nodes = [
            MODULE.PageNode(
                "text",
                "Ada Fang1,2† Robert Lee2",
                [100, 100, 500, 115],
                10,
                line_id="paper:p0001:l0001",
                origin_page=1,
                origin_order=0,
                claimed_line_ids=["paper:p0001:l0001"],
            )
        ]
        updated, audit = MODULE.apply_author_superscript_plans(
            nodes, [plan], page_number=1, page_width=612
        )
        self.assertEqual(
            updated[0].text,
            "Ada Fang<sup>1,2†</sup> Robert Lee<sup>2</sup>",
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["markers"], ["1,2†", "2"])
        self.assertEqual(audit["superscripts_emitted"], 2)

    def test_visible_author_plan_is_applied_when_affiliation_plan_is_absent(self) -> None:
        author = MODULE.AuthorSuperscriptPlan(
            source_file=Path("authors.tex"),
            start_line=1,
            end_line=1,
            raw_latex=r"\author{Aditya Thimmaiah}",
            pieces=[("literal", "Aditya Thimmaiah"), ("sup", "1")],
        )
        hidden_affiliation = MODULE.AuthorSuperscriptPlan(
            source_file=Path("authors.tex"),
            start_line=2,
            end_line=2,
            raw_latex=r"\affiliation{Hidden Institute}",
            pieces=[("sup", "1"), ("literal", "Hidden Institute")],
        )
        nodes = [
            MODULE.PageNode(
                "text",
                "Aditya Thimmaiah 1",
                [100, 100, 500, 115],
                10,
                line_id="paper:p0001:l0001",
                origin_page=1,
                origin_order=0,
                claimed_line_ids=["paper:p0001:l0001"],
            )
        ]
        updated, audit = MODULE.apply_author_superscript_plans(
            nodes, [author, hidden_affiliation], page_number=1, page_width=612
        )
        self.assertEqual(updated[0].text, "Aditya Thimmaiah <sup>1</sup>")
        self.assertEqual(audit["status"], "partial")
        self.assertEqual(audit["matched_plans"], 1)
        self.assertEqual(audit["unmatched_plans"], 1)

    def test_author_textsuperscript_and_optional_author_are_parsed(self) -> None:
        direct = MODULE.author_plan_from_raw(
            r"Jane Doe\textsuperscript{1,*}",
            source_file=Path("authors.tex"),
            start_line=1,
            end_line=1,
        )
        self.assertIsNotNone(direct)
        assert direct is not None
        self.assertEqual(direct.markdown_text, "Jane Doe<sup>1,*</sup>")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "main.tex"
            source.write_text(
                (
                    r"\documentclass{article}"
                    r"\author[1,2]{Luca Schaufelberger}"
                    r"\author*[1,2,3]{Artur Sidorenko}"
                    r"\affil[1]{Institute of Examples}"
                    r"\begin{document}x\end{document}"
                ),
                encoding="utf-8",
            )
            plans = MODULE.parse_author_superscript_plans([source])
        self.assertEqual(len(plans), 3)
        self.assertEqual(
            {plan.markdown_text for plan in plans},
            {
                "Luca Schaufelberger<sup>1,2</sup>",
                "Artur Sidorenko<sup>1,2,3*</sup>",
                "<sup>1</sup>Institute of Examples",
            },
        )

    def test_explicit_inst_author_markers_are_html_sup(self) -> None:
        plan = MODULE.author_plan_from_raw(
            r"Alessandro Abate\inst{1}\orcidlink{0000-0002-5627-9093}",
            source_file=Path("authors.tex"),
            start_line=1,
            end_line=1,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.markdown_text, "Alessandro Abate<sup>1</sup>")

    def test_revtex_author_affiliation_numbers_follow_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "main.tex"
            source.write_text(
                "\n".join(
                    [
                        r"\documentclass[aps]{revtex4-2}",
                        r"\begin{document}",
                        r"\author{Vahid Nateghi}\affiliation{Institute A}",
                        r"\author{Lara Neureither}\affiliation{Institute B}",
                        r"\author{Simon Olsson}\affiliation{Institute B}",
                        r"\author{Feliks Nüske}\email{x@example.org}\affiliation{Institute A}",
                        r"\end{document}",
                    ]
                ),
                encoding="utf-8",
            )
            plans = MODULE.parse_author_superscript_plans([source])
        self.assertEqual(
            {plan.markdown_text for plan in plans},
            {
                "Vahid Nateghi<sup>1</sup>",
                "Lara Neureither<sup>2</sup>",
                "Simon Olsson<sup>2</sup>",
                "Feliks Nüske<sup>1</sup>",
                "<sup>1</sup>Institute A",
                "<sup>2</sup>Institute B",
            },
        )

    def test_icml_author_affiliation_numbers_follow_first_key_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "main.tex"
            source.write_text(
                "\n".join(
                    [
                        r"\documentclass{article}",
                        r"\icmlauthor{Jonas Elsborg}{dtu,capex}",
                        r"\icmlauthor{Felix Ærtebjerg}{equal,dtu}",
                        r"\icmlauthor{Luca Thiede}{equal,toronto,vector}",
                        r"\icmlaffiliation{dtu}{Department One}",
                        r"\icmlaffiliation{toronto}{Department Three}",
                        r"\icmlaffiliation{capex}{Department Two}",
                        r"\icmlaffiliation{vector}{Department Four}",
                        r"\begin{document}x\end{document}",
                    ]
                ),
                encoding="utf-8",
            )
            plans = MODULE.parse_author_superscript_plans([source])
        self.assertEqual(
            {plan.markdown_text for plan in plans},
            {
                "Jonas Elsborg<sup>1 2</sup>",
                "Felix Ærtebjerg<sup>* 1</sup>",
                "Luca Thiede<sup>* 3 4</sup>",
                "<sup>1</sup>Department One",
                "<sup>2</sup>Department Two",
                "<sup>3</sup>Department Three",
                "<sup>4</sup>Department Four",
            },
        )

    def test_icml_interleaved_affiliation_footnote_is_rebuilt_from_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "main.tex"
            source.write_text(
                "\n".join(
                    [
                        r"\documentclass{article}",
                        r"\icmlauthor{Aditya Thimmaiah}{ut}",
                        r"\icmlauthor{Jayanth Srinivasa}{cisco}",
                        r"\icmlaffiliation{ut}{The University of Texas at Austin}",
                        r"\icmlaffiliation{cisco}{Cisco Research}",
                        r"\icmlcorrespondingauthor{Aditya Thimmaiah}{\texttt{auditt@utexas.edu}}",
                        r"\begin{document}x\end{document}",
                    ]
                ),
                encoding="utf-8",
            )
            plans = MODULE.parse_author_superscript_plans([source])
        line_ids = [f"paper:p0001:l{index:04d}" for index in range(1, 6)]
        texts = [
            "1The",
            "2Cisco",
            "University of Texas at Austin",
            "Research. Correspon-",
            "dence to: Aditya Thimmaiah <auditt@utexas.edu>.",
        ]
        nodes = [
            MODULE.PageNode(
                "text",
                text,
                [50 + index * 20, 650 + (index // 4) * 10, 200 + index * 20, 660 + (index // 4) * 10],
                8,
                lane="left",
                line_id=line_ids[index],
                origin_page=1,
                origin_order=index,
                claimed_line_ids=[line_ids[index]],
            )
            for index, text in enumerate(texts)
        ]
        updated, audit = MODULE.apply_author_superscript_plans(
            nodes, plans, page_number=1, page_width=612
        )
        self.assertEqual(len(updated), 1)
        self.assertEqual(
            updated[0].text,
            "<sup>1</sup>The University of Texas at Austin "
            "<sup>2</sup>Cisco Research "
            "Correspondence to: Aditya Thimmaiah <auditt@utexas.edu>.",
        )
        self.assertEqual(updated[0].claimed_line_ids, line_ids)
        self.assertTrue(audit["icml_affiliation_group_repaired"])
        self.assertEqual(audit["matched_plans"], 2)
        self.assertEqual(audit["unmatched_plans"], 2)

    def test_inline_parser_excludes_complete_multiline_author_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            source = source_root / "main.tex"
            source.write_text(
                "\n".join(
                    [
                        r"\documentclass{article}",
                        r"\begin{document}",
                        r"\author{Xuejun Sun$^{1}$,\\",
                        r"Yiran Song$^{2}$}",
                        r"Visible prose with $x+y$ inline.",
                        r"\end{document}",
                    ]
                ),
                encoding="utf-8",
            )
            blocks, rejections = MODULE.parse_inline_source_blocks(source_root, [])
        self.assertEqual(rejections, [])
        self.assertEqual(len(blocks), 1)
        self.assertIn("Visible prose", blocks[0].raw_latex)
        self.assertNotIn("Xuejun", blocks[0].raw_latex)

    def test_author_thanks_body_does_not_leak_and_orcid_is_not_guessed(self) -> None:
        plan = MODULE.author_plan_from_raw(
            r"Jane Doe$^1$\thanks{Private source-only body}",
            source_file=Path("authors.tex"),
            start_line=1,
            end_line=1,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.markdown_text, "Jane Doe<sup>1</sup>")
        self.assertNotIn("Private source-only body", plan.plain_text)
        self.assertIsNone(
            MODULE.author_plan_from_raw(
                r"Jane Doe\orcidlink{0000-0001-2345-6789}",
                source_file=Path("authors.tex"),
                start_line=1,
                end_line=1,
            )
        )

    def test_strict_contract_counts_author_sup_separately_from_footnotes(self) -> None:
        node = self.strict_line("Ada Fang1", "paper:p0001:l0001", 0)
        inventory = MODULE.freeze_line_inventory([node], 792, 612)
        node.text = "Ada Fang<sup>1</sup>"
        contract = MODULE.build_strict_text_v2_contract(
            markdown="Ada Fang<sup>1</sup>\n",
            ordered_nodes=[node],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
            author_superscript_audit={
                "contract_version": MODULE.AUTHOR_SUPERSCRIPT_CONTRACT_VERSION,
                "status": "passed",
                "plans": 1,
                "matched_plans": 1,
                "superscripts_emitted": 1,
                "markers": ["1"],
                "matched_line_ids": ["paper:p0001:l0001"],
                "marked_lines": ["Ada Fang<sup>1</sup>"],
            },
        )
        self.assertNotIn("footnote_page_sup_count_mismatch", contract["failure_reasons"])
        self.assertNotIn("page_sup_count_mismatch", contract["failure_reasons"])
        self.assertEqual(contract["author_superscripts"]["superscripts_emitted"], 1)

    def test_independent_verifier_allows_source_captionless_table(self) -> None:
        markdown = """<table>
  <thead><tr><th>A</th></tr></thead>
  <tbody><tr><td>1</td></tr></tbody>
</table>"""
        context = VERIFY_MODULE.CheckContext("captionless")
        VERIFY_MODULE.verify_html_tables(
            markdown,
            [{
                "kind": "table",
                "raw_latex": r"\begin{tabular}{c}1\end{tabular}",
                "table_html": markdown,
                "markdown": markdown,
                "table_parse_status": "parsed",
            }],
            context,
        )
        self.assertEqual(context.errors, [])

        required_context = VERIFY_MODULE.CheckContext("caption-required")
        VERIFY_MODULE.verify_html_tables(
            markdown,
            [
                {
                    "kind": "table",
                    "raw_latex": r"\begin{table}\caption{Scores}\end{table}",
                    "table_html": markdown,
                    "markdown": markdown,
                    "table_parse_status": "parsed",
                }
            ],
            required_context,
        )
        self.assertIn("table 1: missing separate caption", required_context.errors)

    def test_display_math_keeps_latex(self) -> None:
        raw = r"""\begin{equation}\label{loss}
\mathcal{L}=\sum_i x_i^2
\end{equation}"""
        value = MODULE.clean_display_math(raw, "equation")
        self.assertTrue(value.startswith("$$\n"))
        self.assertIn(r"\mathcal{L}=\sum_i x_i^2", value)
        self.assertNotIn(r"\label", value)
        self.assertNotIn(r"\begin{equation}", value)

    def test_parse_source_blocks_finds_heading_formula_and_table(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
\section{Results}
Text.
\begin{equation}
x=y
\end{equation}
\begin{table}
\caption{Scores}
\begin{tabular}{lc}
Method & Score \\
\midrule
Ours & 9 \\
\end{tabular}
\end{table}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text(source, encoding="utf-8")
            blocks = MODULE.parse_source_blocks(Path(directory))
        self.assertEqual([block.kind for block in blocks], ["heading", "display_math", "table"])
        self.assertIn("## Results", blocks[0].markdown)
        self.assertEqual(blocks[0].heading_command, "section")
        self.assertFalse(blocks[0].heading_starred)
        self.assertEqual(blocks[0].heading_source_title, "Results")
        self.assertIn("<table", blocks[2].markdown)
        self.assertTrue(blocks[2].markdown.startswith("Scores\n\n<table>"))
        self.assertEqual(blocks[2].caption_markdown, "Scores")
        self.assertTrue(blocks[2].table_html.startswith("<table>"))
        self.assertEqual(blocks[2].table_parse_status, "parsed")
        self.assertNotIn("<caption", blocks[2].markdown)

    def test_parse_source_blocks_records_starred_heading_metadata(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
\subsection*{Acknowledgements}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text(source, encoding="utf-8")
            blocks = MODULE.parse_source_blocks(Path(directory))
            serialized = blocks[0].as_json(Path(directory))
        self.assertEqual(blocks[0].heading_command, "subsection")
        self.assertTrue(blocks[0].heading_starred)
        self.assertEqual(blocks[0].heading_source_title, "Acknowledgements")
        self.assertEqual(serialized["heading_number_status"], "pending")

    def test_parse_source_blocks_finds_standalone_tabular(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
\begin{tabular}{lc}
Method & Score \\
Ours & 9 \\
\end{tabular}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text(source, encoding="utf-8")
            blocks = MODULE.parse_source_blocks(Path(directory))
        self.assertEqual([block.kind for block in blocks], ["table"])
        self.assertIn("<thead>", blocks[0].markdown)
        self.assertIn("<tbody>", blocks[0].markdown)
        self.assertNotIn("data-parse-status", blocks[0].markdown)
        self.assertIsNone(blocks[0].caption_markdown)
        self.assertEqual(blocks[0].table_parse_status, "parsed")

    def test_parse_source_blocks_ignores_content_after_end_document(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
\section{Visible}
\begin{equation}
x=y
\end{equation}
\end{document}
\section{Draft only}
\begin{equation}
z=w
\end{equation}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text(source, encoding="utf-8")
            blocks = MODULE.parse_source_blocks(Path(directory))
        self.assertEqual([block.kind for block in blocks], ["heading", "display_math"])
        self.assertIn("Visible", blocks[0].markdown)
        self.assertNotIn("Draft only", "\n".join(block.markdown for block in blocks))
        self.assertNotIn("z=w", "\n".join(block.markdown for block in blocks))

    def test_parse_source_paragraphs_treats_each_list_item_as_one_paragraph(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
Ordinary first line
continues here.

\begin{itemize}
\item First item begins here
      and continues on another source line.
\item Second item remains separate.
\end{itemize}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text(source, encoding="utf-8")
            paragraphs = MODULE.parse_source_paragraphs(
                Path(directory), [], [source_path.resolve()]
            )
        self.assertEqual(len(paragraphs), 3)
        self.assertEqual(paragraphs[0].kind, "paragraph")
        self.assertEqual(paragraphs[1].kind, "itemize_item")
        self.assertIn("continues on another source line", paragraphs[1].raw_latex)
        self.assertEqual(paragraphs[1].item_ordinal, 1)
        self.assertEqual(paragraphs[2].item_ordinal, 2)
        self.assertNotEqual(paragraphs[1].paragraph_id, paragraphs[2].paragraph_id)

    def test_run_in_heading_tail_is_source_paragraph_and_inline_math(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
\paragraph{Agents' incentives.} We compare welfare $u_i[t]$ across rounds.
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text(source, encoding="utf-8")
            structural = MODULE.parse_source_blocks(Path(directory))
            paragraphs = MODULE.parse_source_paragraphs(
                Path(directory), structural, [source_path.resolve()]
            )
            inline, rejections = MODULE.parse_inline_source_blocks(
                Path(directory), structural
            )
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(
            paragraphs[0].raw_latex,
            r"We compare welfare $u_i[t]$ across rounds.",
        )
        self.assertEqual(rejections, [])
        self.assertEqual(len(inline), 1)
        self.assertEqual(inline[0].target_feature_counts, {"math": 1})

    def test_synctex_inputs_declared_after_content_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = (Path(directory) / "proofs.tex").resolve()
            synctex_path = Path(directory) / "main.synctex.gz"
            with MODULE.gzip.open(synctex_path, "wt", encoding="utf-8") as stream:
                stream.write("SyncTeX Version:1\nContent:\nInput:257:")
                stream.write(str(source_path))
                stream.write("\n")
            inputs = MODULE.synctex_inputs(synctex_path)
        self.assertEqual(inputs[source_path], str(source_path))

    def test_simple_math_macros_expand_in_display_and_inline_math(self) -> None:
        source = r"""
\documentclass{article}
\newcommand{\E}{\mathbb{E}}
\DeclareMathOperator{\val}{Val}
\begin{document}
The value is $\val[X]=\E[X]$.
\begin{equation}
\val[X]=\E[X]
\end{equation}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "main.tex"
            source_path.write_text(source, encoding="utf-8")
            macros = MODULE.collect_simple_math_macros(Path(directory))
            structural = MODULE.parse_source_blocks(Path(directory), macros)
            inline, rejections = MODULE.parse_inline_source_blocks(
                Path(directory), structural, macros
            )
        self.assertEqual(rejections, [])
        self.assertIn(r"\operatorname{Val}[X]=\mathbb{E}[X]", structural[0].markdown)
        math_values = [
            node.value
            for node in MODULE.iter_inline_nodes(inline[0].plan.root)
            if node.kind == "math"
        ]
        self.assertEqual(math_values, [r"\operatorname{Val}[X]=\mathbb{E}[X]"])

    def test_multiline_display_bbox_unions_all_synctex_rows(self) -> None:
        bbox = MODULE.select_display_math_bbox(
            [
                {"bbox": [100, 200, 500, 215], "W": 400, "H": 15},
                {"bbox": [110, 225, 490, 240], "W": 380, "H": 15},
                {"bbox": [120, 250, 480, 265], "W": 360, "H": 15},
            ]
        )
        self.assertEqual(bbox, [100, 200, 500, 265])

    def test_source_paragraph_id_overrides_false_lane_split(self) -> None:
        first = MODULE.PageNode(
            "text",
            "The snub cube is the only",
            [101.265, 687.58, 540.004, 699.535],
            11.955,
            lane="full",
            source_paragraph_id="sp-example",
            source_paragraph_slice_id="sp-example@p0012",
        )
        second = MODULE.PageNode(
            "text",
            "non-hexagonal solid in the non-zero group.",
            [101.265, 702.026, 320.165, 713.981],
            11.955,
            lane="left",
            source_paragraph_id="sp-example",
            source_paragraph_slice_id="sp-example@p0012",
        )
        markdown = MODULE.nodes_to_markdown([first, second], 12)
        self.assertEqual(
            markdown,
            "The snub cube is the only non-hexagonal solid in the non-zero group.\n",
        )

    def test_single_column_short_caption_line_does_not_split_paragraph(self) -> None:
        first = MODULE.PageNode(
            "text",
            "Figure 3: Comparison of different strategies and our 12-band filter",
            [108.0, 274.294, 504.172, 284.257],
            9.963,
            lane="full",
        )
        second = MODULE.PageNode(
            "text",
            "bank (described in Section 3.1).",
            [108.0, 285.253, 234.724, 295.216],
            9.963,
            lane="left",
        )

        markdown = MODULE.nodes_to_markdown(
            [first, second],
            8,
            layout="single_column",
        )

        self.assertEqual(
            markdown,
            "Figure 3: Comparison of different strategies and our 12-band filter "
            "bank (described in Section 3.1).\n",
        )

    def test_different_source_paragraph_ids_force_markdown_boundary(self) -> None:
        first = MODULE.PageNode(
            "text",
            "First paragraph ends without a large visual gap.",
            [72, 100, 540, 112],
            10,
            lane="full",
            source_paragraph_id="sp-one",
            source_paragraph_slice_id="sp-one@p0001",
        )
        second = MODULE.PageNode(
            "text",
            "Second paragraph starts immediately below.",
            [72, 114, 540, 126],
            10,
            lane="full",
            source_paragraph_id="sp-two",
            source_paragraph_slice_id="sp-two@p0001",
        )
        markdown = MODULE.nodes_to_markdown([first, second], 2)
        self.assertEqual(
            markdown,
            "First paragraph ends without a large visual gap.\n\n"
            "Second paragraph starts immediately below.\n",
        )

    def test_same_source_paragraph_uses_distinct_page_slices(self) -> None:
        first_page = MODULE.PageNode(
            "text",
            "Visible text at the end of page one-",
            [72, 700, 540, 712],
            10,
            lane="full",
            source_paragraph_id="sp-cross-page",
            source_paragraph_slice_id="sp-cross-page@p0001",
        )
        second_page = MODULE.PageNode(
            "text",
            "continues at the top of page two.",
            [72, 72, 540, 84],
            10,
            lane="full",
            source_paragraph_id="sp-cross-page",
            source_paragraph_slice_id="sp-cross-page@p0002",
        )
        self.assertEqual(
            MODULE.nodes_to_markdown([first_page], 1),
            "Visible text at the end of page one-\n",
        )
        self.assertEqual(
            MODULE.nodes_to_markdown([second_page], 2),
            "continues at the top of page two.\n",
        )

    def test_synctex_glyph_majority_assigns_source_paragraph(self) -> None:
        paragraph = MODULE.SourceParagraph(
            paragraph_id="sp-example",
            kind="itemize_item",
            source_file=Path("main.tex"),
            source_lines=[10, 11],
            raw_latex="visible prose",
        )
        node = MODULE.PageNode(
            "text",
            "visible prose",
            [100, 700, 320, 714],
            11,
            line_id="line-1",
            origin_page=12,
            claimed_line_ids=["line-1"],
        )
        points = [
            MODULE.SourceParagraphPoint(
                12, 120 + index, 712, "sp-example", Path("main.tex"), 10
            )
            for index in range(4)
        ]
        audit = MODULE.annotate_source_paragraph_ids(
            [node], points, {paragraph.paragraph_id: paragraph}
        )
        self.assertEqual(node.source_paragraph_id, "sp-example")
        self.assertEqual(node.source_paragraph_slice_id, "sp-example@p0012")
        self.assertEqual(audit["lines_mapped"], 1)

    def test_unique_page_start_source_suffix_restores_paragraph_boundary(self) -> None:
        previous = MODULE.SourceParagraph(
            paragraph_id="sp-previous",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[10],
            raw_latex="A long paragraph ends with incomplete environmental data availability.",
        )
        following = MODULE.SourceParagraph(
            paragraph_id="sp-following",
            kind="paragraph",
            source_file=Path("main.tex"),
            source_lines=[12],
            raw_latex="The methodological contribution starts a new paragraph.",
        )
        orphan = self.strict_line(
            "availability.", "orphan", 0, [38, 84, 95, 94]
        )
        orphan.origin_page = 20
        first_following = self.strict_line(
            "The methodological contribution starts", "following", 1, [38, 96, 288, 106]
        )
        first_following.origin_page = 20
        points = [
            MODULE.SourceParagraphPoint(
                20,
                50 + index,
                101,
                "sp-following",
                Path("main.tex"),
                12,
            )
            for index in range(4)
        ]
        audit = MODULE.annotate_source_paragraph_ids(
            [orphan, first_following],
            points,
            {
                previous.paragraph_id: previous,
                following.paragraph_id: following,
            },
        )
        self.assertEqual(orphan.source_paragraph_id, "sp-previous")
        self.assertEqual(first_following.source_paragraph_id, "sp-following")
        self.assertEqual(audit["page_start_source_suffix_lines_mapped"], 1)
        orphan.lane = first_following.lane = "left"
        self.assertEqual(
            MODULE.nodes_to_markdown([orphan, first_following], 20),
            "availability.\n\nThe methodological contribution starts\n",
        )

    def test_inline_prose_is_split_per_sentence_and_footnote_body_is_opaque(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
The value $x$ is stable\footnote{The hidden value $y$ is discussed elsewhere.}.
Next, \textbf{our method} uses $z$.
\begin{equation}
q=r
\end{equation}
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text(source, encoding="utf-8")
            structural = MODULE.parse_source_blocks(Path(directory))
            blocks, rejections = MODULE.parse_inline_source_blocks(
                Path(directory), structural
            )
        self.assertEqual(rejections, [])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(
            blocks[0].target_feature_counts,
            {"math": 1, "footnote": 1},
        )
        self.assertEqual(len(blocks[0].footnotes), 1)
        self.assertIn(r"The hidden value $y$", blocks[0].footnotes[0].body_raw)
        self.assertEqual(
            blocks[1].target_feature_counts,
            {"math": 1, "strong": 1},
        )
        self.assertNotIn("$y$", blocks[0].plan.anchors)

    def test_footnote_only_sentence_is_a_target_and_keeps_full_plan(self) -> None:
        source = r"""
\documentclass{article}
\begin{document}
Several literal anchor words precede the note\footnote{A hidden body with $x$.}.
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text(source, encoding="utf-8")
            blocks, rejections = MODULE.parse_inline_source_blocks(
                Path(directory), []
            )
        self.assertEqual(rejections, [])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].target_feature_counts, {"footnote": 1})
        self.assertEqual(len(blocks[0].footnotes), 1)
        self.assertIn(r"\footnote", blocks[0].plan.raw)
        self.assertTrue(blocks[0].plan.raw.endswith("."))

    def test_three_pdf_footnotes_are_structured_with_exact_claims(self) -> None:
        raw_blocks = [
            r"For several initial values use the recursion\footnote{Note that the $k=2$ case uses \cite{hidden-key}, Theorem \ref{result}, and {\em floor function}.}",
            r"This appeared in a widely available journal.\footnote{The $3$-bonacci numbers appeared in \cite{another-key}.}",
            r"The displayed expression is given by\footnote{We use $\lfloor x \rfloor$ for the {\em floor function}.}",
        ]
        callout_texts = [
            "For several initial values use the recursion1",
            "This appeared in a widely available journal.2",
            "The displayed expression is given by3",
        ]
        blocks = []
        for index, raw in enumerate(raw_blocks, start=1):
            plan = MODULE.parse_inline_plan(raw)
            block_id = f"i{index}"
            blocks.append(
                MODULE.InlineSourceBlock(
                    block_id=block_id,
                    source_file=Path("main.tex"),
                    start_line=index,
                    end_line=index,
                    raw_latex=raw,
                    plan=plan,
                    query_lines=[index],
                    page=1,
                    bbox=[50, 240 + index * 25, 550, 252 + index * 25],
                    mapping_status="mapped",
                    footnotes=MODULE.footnote_specs_from_plan(plan, block_id),
                )
            )
        raw_nodes = [
            self.strict_line(text, f"callout-{index}", index, [50, 240 + index * 25, 550, 252 + index * 25])
            for index, text in enumerate(callout_texts, start=1)
        ]
        raw_nodes.extend(
            [
                self.strict_line("1Note that the k = 2 case uses", "def-1a", 4, [50, 610, 550, 622]),
                self.strict_line("[2], Theorem 2.1, and", "def-1b", 5, [50, 624, 550, 636]),
                self.strict_line("floor", "def-1c", 6, [50, 638, 550, 650]),
                self.strict_line("function.", "def-1d", 7, [50, 652, 550, 664]),
                self.strict_line("2The 3-bonacci numbers appeared in [4].", "def-2", 8, [50, 670, 550, 682]),
                self.strict_line("3We use floor x for the floor function.", "def-3", 9, [50, 688, 550, 700]),
                self.strict_line("1", "page-number", 10, [300, 780, 312, 792]),
            ]
        )
        canonical, layout = MODULE.order_page_nodes(raw_nodes, 612, "single_column")
        inventory = MODULE.freeze_line_inventory(canonical, 800)
        enriched, integration = MODULE.apply_inline_source_blocks(
            raw_nodes, blocks, 612, "single_column"
        )
        self.assertEqual(integration["features_unresolved_total"], 0)
        ordered, _ = MODULE.order_page_nodes(enriched, 612, layout)
        integrated, footnote_integration = MODULE.integrate_footnote_definitions(
            ordered, blocks, inventory, 1
        )
        self.assertEqual(footnote_integration["status"], "passed")
        definition_nodes = [
            node for node in integrated if node.kind == "footnote_definitions"
        ]
        self.assertEqual(len(definition_nodes), 1)
        definition = definition_nodes[0]
        self.assertEqual(
            definition.claimed_line_ids,
            ["def-1a", "def-1b", "def-1c", "def-1d", "def-2", "def-3"],
        )
        self.assertIn(r"<sup>1</sup> Note that the $k=2$ case uses [2]", definition.text)
        self.assertIn("Theorem 2.1", definition.text)
        self.assertIn("*floor function*", definition.text)
        self.assertIn(r"<sup>2</sup> The $3$-bonacci", definition.text)
        self.assertIn(r"<sup>3</sup> We use $\lfloor x \rfloor$", definition.text)
        self.assertNotIn("hidden-key", definition.text)
        self.assertNotIn("another-key", definition.text)
        self.assertEqual(integrated[-1].text, "1")
        markdown = MODULE.nodes_to_markdown(integrated, 1)
        audit = MODULE.finalize_footnote_audit(
            markdown, blocks, footnote_integration
        )
        self.assertEqual((audit["total"], audit["structured"], audit["fallback"]), (3, 3, 0))
        for note in audit["notes"]:
            self.assertEqual(note["callout_markdown_occurrences"], 1)
            self.assertEqual(note["definition_markdown_occurrences"], 1)
            self.assertEqual(note["total_sup_occurrences"], 2)
            self.assertEqual(note["representation"], "html_sup")
            self.assertEqual(note["content_validation_status"], "passed")
            self.assertEqual(markdown.count(f"<sup>{note['marker']}</sup>"), 2)
        self.assertNotRegex(markdown, r"\[\^[0-9]+\]")
        claims = MODULE.audit_line_claims(inventory, integrated, 1)
        self.assertEqual(claims["status"], "passed")
        contract = MODULE.build_strict_text_v2_contract(
            markdown=markdown,
            ordered_nodes=integrated,
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=blocks,
            footnote_audit=audit,
        )
        self.assertEqual(contract["strict_text_v2_status"], "passed")
        self.assertEqual(contract["failure_reasons"], [])

    def test_footnote_definition_failure_is_atomic_and_restores_marker(self) -> None:
        raw = r"Several literal anchor words precede the note\footnote{Expected body text.}"
        plan = MODULE.parse_inline_plan(raw)
        block = MODULE.InlineSourceBlock(
            block_id="i1",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=1,
            raw_latex=raw,
            plan=plan,
            query_lines=[1],
            page=1,
            bbox=[50, 250, 550, 262],
            mapping_status="mapped",
            footnotes=MODULE.footnote_specs_from_plan(plan, "i1"),
        )
        raw_nodes = [
            self.strict_line(
                "Several literal anchor words precede the note1",
                "callout",
                0,
                [50, 250, 550, 262],
            ),
            self.strict_line("1Different body text.", "definition", 1, [50, 650, 550, 662]),
            self.strict_line("1", "page-number", 2, [300, 780, 312, 792]),
        ]
        canonical, _ = MODULE.order_page_nodes(raw_nodes, 612, "single_column")
        inventory = MODULE.freeze_line_inventory(canonical, 800)
        enriched, _ = MODULE.apply_inline_source_blocks(
            raw_nodes, [block], 612, "single_column"
        )
        self.assertTrue(any("<sup>1</sup>" in node.text for node in enriched))
        ordered, _ = MODULE.order_page_nodes(enriched, 612, "single_column")
        integrated, audit = MODULE.integrate_footnote_definitions(
            ordered, [block], inventory, 1
        )
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["fallback"], 1)
        self.assertFalse(any(node.kind == "footnote_definitions" for node in integrated))
        self.assertIn("note1", " ".join(node.text for node in integrated))
        self.assertIn("1Different body text.", " ".join(node.text for node in integrated))
        self.assertNotIn("<sup>1</sup>", " ".join(node.text for node in integrated))

    def test_duplicate_footnote_marker_falls_back_atomically(self) -> None:
        blocks = []
        raw_nodes = []
        for index, word in enumerate(("alpha", "beta"), start=1):
            raw = rf"Several anchor words end with {word}\footnote{{Body {word}.}}"
            plan = MODULE.parse_inline_plan(raw)
            block_id = f"i{index}"
            blocks.append(
                MODULE.InlineSourceBlock(
                    block_id=block_id,
                    source_file=Path("main.tex"),
                    start_line=index,
                    end_line=index,
                    raw_latex=raw,
                    plan=plan,
                    query_lines=[index],
                    page=1,
                    bbox=[50, 240 + index * 25, 550, 252 + index * 25],
                    mapping_status="mapped",
                    footnotes=MODULE.footnote_specs_from_plan(plan, block_id),
                )
            )
            raw_nodes.append(
                self.strict_line(
                    f"Several anchor words end with {word}1",
                    f"callout-{index}",
                    index,
                    [50, 240 + index * 25, 550, 252 + index * 25],
                )
            )
        raw_nodes.extend(
            [
                self.strict_line("1Body alpha.", "def-a", 3, [50, 650, 550, 662]),
                self.strict_line("1Body beta.", "def-b", 4, [50, 665, 550, 677]),
            ]
        )
        canonical, _ = MODULE.order_page_nodes(raw_nodes, 612, "single_column")
        inventory = MODULE.freeze_line_inventory(canonical, 800)
        enriched, _ = MODULE.apply_inline_source_blocks(
            raw_nodes, blocks, 612, "single_column"
        )
        ordered, _ = MODULE.order_page_nodes(enriched, 612, "single_column")
        integrated, audit = MODULE.integrate_footnote_definitions(
            ordered, blocks, inventory, 1
        )
        self.assertEqual(audit["failure_reason"], "duplicate_footnote_marker")
        self.assertEqual(audit["fallback"], 2)
        self.assertNotIn("<sup>1</sup>", " ".join(node.text for node in integrated))

    def test_inline_enrichment_keeps_pdf_citation_and_restores_markup(self) -> None:
        raw = r"A \textbf{robust model} uses $x_i$ in \cite{hidden-key}."
        block = MODULE.InlineSourceBlock(
            block_id="i1",
            source_file=Path("main.tex"),
            start_line=10,
            end_line=10,
            raw_latex=raw,
            plan=MODULE.parse_inline_plan(raw),
            query_lines=[10],
            page=1,
            bbox=[50, 100, 500, 112],
            mapping_status="mapped",
        )
        nodes = [
            MODULE.PageNode(
                "text",
                "A robust model uses xi in [3].",
                [50, 100, 500, 112],
                10,
            )
        ]
        enriched, audit = MODULE.apply_inline_source_blocks(nodes, [block], 612)
        self.assertEqual(audit["features_unresolved_total"], 0)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(
            enriched[0].text,
            "A **robust model** uses $x_i$ in [3].",
        )
        self.assertNotIn("hidden-key", enriched[0].text)

    def test_inline_math_absorbs_delayed_pdf_subscript_satellites(self) -> None:
        raw = r"Scoring tests assign $s_{DM}(o),s_{AD}(o)$ for both agents."
        block = MODULE.InlineSourceBlock(
            block_id="i-subscripts",
            source_file=Path("main.tex"),
            start_line=10,
            end_line=10,
            raw_latex=raw,
            plan=MODULE.parse_inline_plan(raw),
            query_lines=[10],
            page=1,
            bbox=[72, 100, 540, 125],
            mapping_status="mapped",
        )
        baseline = self.strict_line(
            "Scoring tests assign s (o), s (o) for both agents.",
            "baseline",
            0,
            [72, 100, 540, 115],
        )
        delayed_dm = self.strict_line("DM", "sub-dm", 1, [270, 106, 285, 113])
        delayed_ad = self.strict_line("AD", "sub-ad", 2, [330, 106, 345, 113])
        delayed_dm.font_size = 7
        delayed_ad.font_size = 7
        enriched, audit = MODULE.apply_inline_source_blocks(
            [baseline, delayed_dm, delayed_ad],
            [block],
            612,
            "single_column",
        )
        self.assertEqual(audit["blocks_matched"], 1)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(
            enriched[0].text,
            r"Scoring tests assign $s_{DM}(o),s_{AD}(o)$ for both agents.",
        )
        self.assertEqual(block.absorbed_pdf_line_ids, ["sub-dm", "sub-ad"])
        self.assertEqual(
            enriched[0].claimed_line_ids,
            ["baseline", "sub-dm", "sub-ad"],
        )

    def test_inline_enrichment_propagates_exact_line_claims(self) -> None:
        raw = r"Alpha context before $x$ remains stable near omega."
        block = MODULE.InlineSourceBlock(
            block_id="ip1",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=1,
            raw_latex=raw,
            plan=MODULE.parse_inline_plan(raw),
            query_lines=[1],
            page=1,
            bbox=[50, 100, 500, 124],
            mapping_status="mapped",
        )
        lines = [
            self.strict_line("Alpha context before x remains", "p0001-l0001", 0, [50, 100, 300, 111]),
            self.strict_line("stable near omega.", "p0001-l0002", 1, [50, 112, 240, 123]),
        ]
        enriched, audit = MODULE.apply_inline_source_blocks(lines, [block], 612)
        self.assertEqual(audit["blocks_matched"], 1)
        self.assertEqual(enriched[0].claimed_line_ids, ["p0001-l0001", "p0001-l0002"])
        self.assertEqual(enriched[0].origin_order, 0)
        self.assertIn("ip1", enriched[0].inline_source_ids)

    def test_inline_alignment_uses_virtual_whitespace_not_adjacent_paragraphs(self) -> None:
        raw = (
            "\\begin{center}\n"
            r"\emph{If one source paragraph spans two visible lines.}"
            "\n\\end{center}"
        )
        block = MODULE.InlineSourceBlock(
            block_id="ip-boundary",
            source_file=Path("main.tex"),
            start_line=10,
            end_line=12,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[11],
            page=1,
            bbox=[70, 120, 540, 150],
            mapping_status="mapped",
        )
        previous = self.strict_line(
            "Previous paragraph remains separate.",
            "p0001-l0001",
            0,
            [70, 100, 540, 111],
        )
        first = self.strict_line(
            "If one source paragraph spans two",
            "p0001-l0002",
            1,
            [74, 120, 538, 131],
        )
        second = self.strict_line(
            "visible lines.",
            "p0001-l0003",
            2,
            [195, 134, 415, 145],
        )
        following = self.strict_line(
            "Following paragraph also remains separate.",
            "p0001-l0004",
            3,
            [70, 160, 540, 171],
        )
        for node in (first, second):
            node.source_paragraph_id = "sp-target"
            node.source_paragraph_slice_id = "sp-target@p0001"
        previous.source_paragraph_id = "sp-before"
        previous.source_paragraph_slice_id = "sp-before@p0001"
        following.source_paragraph_id = "sp-after"
        following.source_paragraph_slice_id = "sp-after@p0001"

        enriched, audit = MODULE.apply_inline_source_blocks(
            [previous, first, second, following],
            [block],
            612,
            "single_column",
        )

        self.assertEqual(audit["blocks_matched"], 1)
        replacement = next(node for node in enriched if "ip-boundary" in node.inline_source_ids)
        self.assertEqual(
            replacement.claimed_line_ids,
            ["p0001-l0002", "p0001-l0003"],
        )
        self.assertEqual(replacement.source_paragraph_id, "sp-target")
        self.assertEqual(
            replacement.text,
            "*If one source paragraph spans two visible lines.*",
        )
        self.assertTrue(any(node.text == previous.text for node in enriched))
        self.assertTrue(any(node.text == following.text for node in enriched))

    def test_unique_markdown_fallback_only_changes_plain_prose(self) -> None:
        raw = r"Set $q=\lfloor n/2\rfloor$."
        block = MODULE.InlineSourceBlock(
            block_id="i2",
            source_file=Path("main.tex"),
            start_line=20,
            end_line=20,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[20],
            page=1,
            bbox=[50, 200, 300, 212],
            mapping_status="mapped",
            match_status="fallback_pdf",
        )
        markdown = "Prose.\n\nSet q = floor(n/2).\n\n$$\nq=0\n$$\n"
        enriched, audit = MODULE.apply_inline_blocks_to_markdown(markdown, [block])
        self.assertEqual(audit["matched_by_unique_markdown_alignment"], 1)
        self.assertIn(r"Set $q=\lfloor n/2\rfloor$.", enriched)
        self.assertIn("$$\nq=0\n$$", enriched)
        self.assertEqual(block.match_reason, "unique_markdown_prose_alignment")

    def test_markdown_fallback_rejects_ambiguous_matches(self) -> None:
        raw = r"Set $q$."
        block = MODULE.InlineSourceBlock(
            block_id="i3",
            source_file=Path("main.tex"),
            start_line=30,
            end_line=30,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[30],
            page=1,
            bbox=[50, 300, 300, 312],
            mapping_status="mapped",
            match_status="fallback_pdf",
        )
        markdown = "Set q.\n\nSet r.\n"
        enriched, audit = MODULE.apply_inline_blocks_to_markdown(markdown, [block])
        self.assertEqual(enriched, markdown)
        self.assertEqual(audit["ambiguous_markdown_alignments"], 1)
        self.assertEqual(block.match_reason, "ambiguous_markdown_alignment")

    def test_source_first_fallback_repairs_corrupt_math_glyph_order(self) -> None:
        raw = r"POVM is non-negative and complete: $A\succeq0$, where $I$ is the identity."
        block = MODULE.InlineSourceBlock(
            block_id="i4",
            source_file=Path("main.tex"),
            start_line=40,
            end_line=40,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[40],
            page=1,
            bbox=[50, 400, 500, 430],
            mapping_status="mapped",
            match_status="fallback_pdf",
        )
        markdown = "Before.\n\nPOVM is non-negative and com- A I plete: A >= 0, where I is the identity.\n\nAfter.\n"
        enriched, audit = MODULE.apply_source_first_inline_blocks_to_markdown(
            markdown, [block]
        )
        self.assertEqual(audit["matched_by_unique_source_first_alignment"], 1)
        self.assertIn(
            r"POVM is non-negative and complete: $A\succeq0$, where $I$ is the identity.",
            enriched,
        )
        self.assertNotIn("com- A I plete", enriched)

    def test_structural_integration_demotes_non_emitted_line_match(self) -> None:
        raw = r"The score is $x$."
        block = MODULE.InlineSourceBlock(
            block_id="i5",
            source_file=Path("main.tex"),
            start_line=50,
            end_line=50,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[50],
            page=1,
            bbox=[50, 500, 300, 512],
            mapping_status="mapped",
            match_status="matched",
            enriched_markdown=r"The score is $x$.",
        )
        audit = MODULE.reconcile_emitted_inline_blocks([block], set())
        self.assertEqual(audit["line_matches_removed_by_structural_integration"], 1)
        self.assertEqual(block.match_status, "fallback_pdf")
        self.assertEqual(
            block.match_reason,
            "line_match_removed_by_structural_integration",
        )

    def test_final_markdown_reconciliation_checks_real_markup(self) -> None:
        raw = r"The score is $x$."
        present = MODULE.InlineSourceBlock(
            block_id="i6",
            source_file=Path("main.tex"),
            start_line=60,
            end_line=60,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[60],
            match_status="matched",
            enriched_markdown=r"The score is $x$.",
        )
        missing = MODULE.InlineSourceBlock(
            block_id="i7",
            source_file=Path("main.tex"),
            start_line=61,
            end_line=61,
            raw_latex=r"The score is $y$.",
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(r"The score is $y$.")),
            query_lines=[61],
            match_status="matched",
            enriched_markdown=r"The score is $y$.",
        )
        audit = MODULE.reconcile_final_inline_markup(
            "Before.\n\nThe score is $x$.\n",
            [present, missing],
        )
        self.assertEqual(audit["final_markup_claims_present"], 1)
        self.assertEqual(audit["final_markup_claims_missing"], 1)
        self.assertEqual(present.match_status, "matched")
        self.assertEqual(missing.match_status, "fallback_pdf")

    def test_source_first_fallback_never_overwrites_existing_inline_markup(self) -> None:
        raw = r"Alpha protected uses $x$ near omega."
        block = MODULE.InlineSourceBlock(
            block_id="i8",
            source_file=Path("main.tex"),
            start_line=70,
            end_line=70,
            raw_latex=raw,
            plan=MODULE.focus_inline_plan(MODULE.parse_inline_plan(raw)),
            query_lines=[70],
            page=1,
            bbox=[50, 600, 500, 612],
            mapping_status="mapped",
            match_status="fallback_pdf",
        )
        markdown = "Alpha **protected** uses x near omega.\n"
        enriched, audit = MODULE.apply_source_first_inline_blocks_to_markdown(
            markdown, [block]
        )
        self.assertEqual(audit["matched_by_unique_source_first_alignment"], 0)
        self.assertEqual(enriched, markdown)
        self.assertEqual(block.match_status, "fallback_pdf")

    def test_inline_markdown_syntax_detects_escaped_closing_dollar(self) -> None:
        self.assertEqual(
            MODULE.markdown_inline_syntax_issues(r"Valid $\ldots\ $ math."),
            [],
        )
        issues = MODULE.markdown_inline_syntax_issues(r"Broken $\ldots\$ math.")
        self.assertEqual(len(issues), 1)
        self.assertTrue(issues[0].startswith("unclosed_inline_math_at_offset_"))

    def test_ordered_anchor_probe_tolerates_discretionary_hyphen(self) -> None:
        self.assertTrue(
            MODULE.ordered_anchor_probe_matches(
                ["The ORGANIZATION", "label is retained"],
                "The ORGA- NIZATION value label is retained.",
            )
        )
        self.assertFalse(
            MODULE.ordered_anchor_probe_matches(
                ["label is retained", "The ORGANIZATION"],
                "The ORGA- NIZATION value label is retained.",
            )
        )

    def test_synctex_output_to_top_left_bbox(self) -> None:
        raw = """SyncTeX result begin
Output:/tmp/paper.pdf
Page:3
x:20.0
y:30.0
h:72.0
v:212.0
W:468.0
H:12.0
SyncTeX result end
"""
        records = MODULE.parse_synctex_output(raw)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["Page"], 3)
        self.assertEqual(records[0]["bbox"], [72.0, 200.0, 540.0, 212.0])

    def test_natbib_retry_is_limited_to_complete_natbib_aux_failure(self) -> None:
        allowed = """Package natbib Error: Bibliograph
y not compatible with author-year citations.
Output written on /tmp/paper.xdv (14 pages).
SyncTeX written on /tmp/paper.synctex.gz.
"""
        self.assertEqual(
            MODULE.natbib_retry_reason(allowed),
            "natbib_author_year_aux_error_after_complete_xdv",
        )
        self.assertIsNone(MODULE.natbib_retry_reason("Undefined control sequence. Output written on /tmp/paper.xdv"))
        self.assertIsNone(
            MODULE.natbib_retry_reason(
                "Package natbib Error: Bibliography not compatible with author-year citations."
            )
        )

    def test_two_column_order_is_left_then_right(self) -> None:
        node = MODULE.PageNode
        nodes = [
            node("text", "Title", [100, 10, 500, 20], 16),
            *[node("text", f"L{i}", [50, 50 + i * 12, 280, 60 + i * 12], 10) for i in range(6)],
            *[node("text", f"R{i}", [330, 50 + i * 12, 560, 60 + i * 12], 10) for i in range(6)],
        ]
        ordered, layout = MODULE.order_page_nodes(nodes, 612)
        self.assertEqual(layout, "two_column")
        texts = [item.text for item in ordered]
        self.assertEqual(texts[0], "Title")
        self.assertLess(texts.index("L5"), texts.index("R0"))

    def test_two_column_centered_footer_folio_is_emitted_after_right_column(self) -> None:
        nodes = [
            *[
                self.strict_line(
                    f"L{i}", f"left-{i}", i, [38, 84 + i * 12, 289, 94 + i * 12]
                )
                for i in range(6)
            ],
            *[
                self.strict_line(
                    f"R{i}", f"right-{i}", i + 6, [307, 84 + i * 12, 558, 94 + i * 12]
                )
                for i in range(6)
            ],
            self.strict_line("18", "folio", 12, [292.7, 770.8, 302.6, 780.8]),
        ]
        ordered, layout = MODULE.order_page_nodes(nodes, 595.276)
        self.assertEqual(layout, "two_column")
        self.assertEqual(ordered[-1].text, "18")
        self.assertEqual(ordered[-1].lane, "full")
        texts = [item.text for item in ordered]
        self.assertLess(texts.index("R5"), texts.index("18"))
        inventory = MODULE.freeze_line_inventory(ordered, 841.89, 595.276)
        folio = next(entry for entry in inventory["lines"] if entry["text"] == "18")
        self.assertTrue(folio["must_preserve"]["page_number"])

    def test_strict_ordered_validator_rejects_swapped_blocks(self) -> None:
        alpha = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        omega = "lambda mu nu xi omicron pi rho sigma tau upsilon"
        metrics = MODULE.ordered_text_metrics(alpha + " " + omega, omega + " " + alpha)
        self.assertEqual(metrics["status"], "failed")
        self.assertLess(metrics["anchor_monotonicity"], 1.0)

    def test_strict_ordered_validator_detects_missing_identical_line(self) -> None:
        repeated = "same repeated visible line with enough unique context tokens"
        metrics = MODULE.ordered_text_metrics(
            repeated + " " + repeated + " final anchor words remain here",
            repeated + " final anchor words remain here",
        )
        self.assertEqual(metrics["status"], "failed")
        self.assertGreater(metrics["token_missing"], 0)
        self.assertLess(metrics["fivegram_recall"], 0.99)

    def test_line_claim_audit_detects_duplicate_missing_and_order(self) -> None:
        original = [
            self.strict_line("first line", "p0001-l0001", 0),
            self.strict_line("second line", "p0001-l0002", 1),
            self.strict_line("third line", "p0001-l0003", 2),
        ]
        ordered, _ = MODULE.order_page_nodes(original, 612, layout_hint="single_column")
        inventory = MODULE.freeze_line_inventory(ordered, 792)
        emitted = [
            self.strict_line("third line", "p0001-l0003", 2),
            MODULE.PageNode(
                "text",
                "first twice",
                [50, 50, 550, 60],
                10,
                line_id="p0001-l0001",
                origin_page=1,
                origin_order=0,
                claimed_line_ids=["p0001-l0001", "p0001-l0001"],
            ),
        ]
        audit = MODULE.audit_line_claims(inventory, emitted, 1)
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["missing_line_ids"], ["p0001-l0002"])
        self.assertEqual(audit["duplicate_line_ids"], ["p0001-l0001"])
        self.assertTrue(audit["order_inversions"])

    def test_line_claim_audit_rejects_reversed_ids_inside_one_node(self) -> None:
        original = [
            self.strict_line("first line", "l1", 0),
            self.strict_line("second line", "l2", 1),
        ]
        inventory = MODULE.freeze_line_inventory(original, 792)
        merged = MODULE.PageNode(
            "text",
            "second line first line",
            [50, 50, 550, 72],
            10,
            origin_page=1,
            claimed_line_ids=["l2", "l1"],
        )
        audit = MODULE.audit_line_claims(inventory, [merged], 1)
        self.assertEqual(audit["status"], "failed")
        self.assertFalse(audit["canonical_order_match"])

    def test_structural_claim_must_be_contiguous_and_exact_once(self) -> None:
        original = [
            self.strict_line("one", "l1", 0),
            self.strict_line("two", "l2", 1),
            self.strict_line("three", "l3", 2),
        ]
        inventory = MODULE.freeze_line_inventory(original, 792)
        structure = MODULE.PageNode(
            "display_math",
            "$$x=y$$",
            [50, 50, 550, 84],
            0,
            source_block_id="b1",
            origin_page=1,
            claimed_line_ids=["l1", "l3"],
        )
        retained = self.strict_line("two", "l2", 1)
        audit = MODULE.audit_line_claims(inventory, [structure, retained], 1)
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["noncontiguous_structural_claims"], ["b1"])

    def test_words_to_lines_assigns_stable_unique_ids_to_repeated_text(self) -> None:
        class FakePage:
            width = 612
            height = 792
            page_number = 3
            chars: list[dict[str, object]] = []

            @staticmethod
            def extract_words(**_kwargs: object) -> list[dict[str, object]]:
                return [
                    {"text": "Repeated", "x0": 50.0, "x1": 110.0, "top": 100.0, "bottom": 112.0},
                    {"text": "Repeated", "x0": 50.0, "x1": 110.0, "top": 200.0, "bottom": 212.0},
                ]

        first = MODULE.words_to_line_nodes(FakePage())
        second = MODULE.words_to_line_nodes(FakePage())
        self.assertEqual([node.line_id for node in first], [node.line_id for node in second])
        self.assertEqual(len({node.line_id for node in first}), 2)
        self.assertTrue(all(node.claimed_line_ids == [node.line_id] for node in first))

    def test_line_inventory_preserves_two_column_canonical_order_and_hash(self) -> None:
        nodes = [
            self.strict_line("Title", "p0001-title", 0, [100, 10, 500, 20]),
            *[
                self.strict_line(f"L{i}", f"p0001-left-{i}", i + 1, [50, 50 + i * 12, 280, 60 + i * 12])
                for i in range(6)
            ],
            *[
                self.strict_line(f"R{i}", f"p0001-right-{i}", i + 7, [330, 50 + i * 12, 560, 60 + i * 12])
                for i in range(6)
            ],
        ]
        ordered, layout = MODULE.order_page_nodes(nodes, 612)
        inventory = MODULE.freeze_line_inventory(ordered, 792)
        self.assertEqual(layout, "two_column")
        self.assertLess(
            inventory["canonical_line_ids"].index("p0001-left-5"),
            inventory["canonical_line_ids"].index("p0001-right-0"),
        )
        self.assertEqual(inventory["sha256"], MODULE.line_inventory_hash(inventory["lines"]))
        self.assertEqual(len(inventory["sha256"]), 64)

    def test_must_preserve_inventory_marks_edges_caption_number_and_hyphen(self) -> None:
        nodes = [
            self.strict_line("Running header", "header", 0, [50, 10, 550, 20]),
            self.strict_line("Table 4: Results", "caption", 1, [50, 200, 550, 212]),
            self.strict_line("17", "page-number", 2, [290, 760, 310, 772]),
            self.strict_line("continued-", "hyphen", 3, [50, 775, 200, 787]),
        ]
        inventory = MODULE.freeze_line_inventory(nodes, 792)
        by_id = {entry["line_id"]: entry for entry in inventory["lines"]}
        self.assertTrue(by_id["header"]["must_preserve"]["header"])
        self.assertTrue(by_id["caption"]["must_preserve"]["caption"])
        self.assertTrue(by_id["page-number"]["must_preserve"]["page_number"])
        self.assertTrue(by_id["hyphen"]["must_preserve"]["page_edge_hyphen"])

    def test_table_caption_number_is_preserved_and_required(self) -> None:
        block = MODULE.SourceBlock(
            block_id="t-number",
            kind="table",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=5,
            raw_latex="",
            markdown="Results\n\n<table><tbody><tr><td>1</td></tr></tbody></table>",
            query_lines=[1],
            caption_markdown="Results",
            table_html="<table><tbody><tr><td>1</td></tr></tbody></table>",
            table_parse_status="parsed",
        )
        MODULE.preserve_visible_table_number(block, "Table 7: Results")
        self.assertTrue(block.markdown.startswith("Table 7: Results\n\n<table>"))
        self.assertNotIn("<caption", block.markdown)
        self.assertEqual(block.visible_caption_prefix, "Table 7")
        self.assertEqual(block.caption_number_status, "preserved")

    def test_table_number_is_not_synthesized_when_page_caption_is_unnumbered(self) -> None:
        block = MODULE.SourceBlock(
            block_id="t-unnumbered",
            kind="table",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=5,
            raw_latex="",
            markdown="Results\n\n<table><tbody><tr><td>1</td></tr></tbody></table>",
            query_lines=[1],
            caption_markdown="Results",
            table_html="<table><tbody><tr><td>1</td></tr></tbody></table>",
            table_parse_status="parsed",
        )
        MODULE.preserve_visible_table_number(block, "Results")
        self.assertEqual(block.caption_markdown, "Results")
        self.assertIsNone(block.visible_caption_prefix)
        self.assertEqual(block.caption_number_status, "visible_unnumbered")
        self.assertNotIn("Table 1", block.markdown)

    def test_display_formula_number_is_preserved_as_safe_tag(self) -> None:
        block = MODULE.SourceBlock(
            block_id="eq-number",
            kind="display_math",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=3,
            raw_latex="",
            markdown="$$\nx=y\n$$",
            query_lines=[3],
        )
        line = self.strict_line("x = y (3.2)", "equation-line", 0)
        MODULE.preserve_visible_formula_number(block, [line])
        self.assertIn(r"\tag{3.2}", block.markdown)
        self.assertEqual(block.pdf_visible_formula_number, "3.2")
        self.assertEqual(block.formula_number_status, "preserved")

    def test_symbolic_parenthesis_is_not_mistaken_for_equation_number(self) -> None:
        block = MODULE.SourceBlock(
            block_id="eq-symbol",
            kind="display_math",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=3,
            raw_latex="",
            markdown="$$\nf(x)\n$$",
            query_lines=[3],
        )
        line = self.strict_line("f (x)", "equation-line", 0)
        MODULE.preserve_visible_formula_number(block, [line])
        self.assertNotIn(r"\tag", block.markdown)
        self.assertEqual(block.formula_number_status, "absent")

    def test_caption_cannot_be_ignored_by_strict_contract(self) -> None:
        caption = self.strict_line("Figure 2: Required caption", "caption-line", 0)
        inventory = MODULE.freeze_line_inventory([caption], 792)
        contract = MODULE.build_strict_text_v2_contract(
            markdown="unrelated output\n",
            ordered_nodes=[],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertIn("required_caption_unclaimed", contract["failure_reasons"])
        self.assertEqual(contract["contract"]["ignored_graphic"], 0)

    def test_structural_and_inline_sentinels_must_appear_exactly_once(self) -> None:
        line = self.strict_line("Visible equation (1)", "eq-line", 0)
        inventory = MODULE.freeze_line_inventory([line], 792)
        structural = MODULE.PageNode(
            "display_math",
            "$$\nx=y\n$$",
            [50, 50, 550, 70],
            0,
            source_block_id="eq1",
            line_id="eq-line",
            origin_page=1,
            claimed_line_ids=["eq-line"],
        )
        contract = MODULE.build_strict_text_v2_contract(
            markdown="$$\nx=y\n$$\n\n$$\nx=y\n$$\n",
            ordered_nodes=[structural],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertIn("structural_sentinel_not_exact_once:eq1", contract["failure_reasons"])

    def test_strict_contract_passes_complete_ordered_claims_and_exposes_hash(self) -> None:
        nodes = [
            self.strict_line(
                "Alpha beta gamma delta epsilon zeta eta theta.", "line-a", 0
            ),
            self.strict_line(
                "Iota kappa lambda mu nu xi omicron pi rho.", "line-b", 1
            ),
        ]
        inventory = MODULE.freeze_line_inventory(nodes, 792)
        markdown = MODULE.nodes_to_markdown(nodes, 2)
        contract = MODULE.build_strict_text_v2_contract(
            markdown=markdown,
            ordered_nodes=nodes,
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
        )
        self.assertEqual(contract["strict_text_v2_status"], "passed")
        self.assertEqual(contract["failure_reasons"], [])
        self.assertEqual(contract["line_inventory"]["count"], 2)
        self.assertEqual(len(contract["line_inventory"]["sha256"]), 64)
        self.assertEqual(contract["claims"]["claimed_unique_count"], 2)
        self.assertEqual(len(contract["markdown_sha256"]), 64)

    def test_strict_contract_rejects_unstructured_visible_math(self) -> None:
        node = self.strict_line(
            "The unstructured relation x = y remains in PDF text.",
            "math-line",
            0,
        )
        inventory = MODULE.freeze_line_inventory([node], 792)
        markdown = MODULE.nodes_to_markdown([node], 1)
        contract = MODULE.build_strict_text_v2_contract(
            markdown=markdown,
            ordered_nodes=[node],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertEqual(
            contract["unstructured_math_fragment_issues"],
            ["unstructured_math_text:math-line"],
        )

    def test_strict_contract_couples_claimed_id_to_inventory_text(self) -> None:
        original = self.strict_line(
            "Alpha beta gamma delta epsilon zeta eta theta.", "line-a", 0
        )
        inventory = MODULE.freeze_line_inventory([original], 792)
        wrong = self.strict_line(
            "Completely unrelated words occupy this claimed output node.",
            "line-a",
            0,
        )
        markdown = MODULE.nodes_to_markdown([wrong], 2)
        contract = MODULE.build_strict_text_v2_contract(
            markdown=markdown,
            ordered_nodes=[wrong],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertIn(
            "claimed_line_text_metrics_failed", contract["failure_reasons"]
        )

    def test_strict_contract_rejects_unresolved_target_inline_feature(self) -> None:
        line = self.strict_line(
            "The visible value x remains in ordinary prose here.", "line-a", 0
        )
        inventory = MODULE.freeze_line_inventory([line], 792)
        block = MODULE.InlineSourceBlock(
            block_id="inline-math",
            source_file=Path("main.tex"),
            start_line=1,
            end_line=1,
            raw_latex=r"The visible value $x$ remains in ordinary prose here.",
            plan=MODULE.parse_inline_plan(
                r"The visible value $x$ remains in ordinary prose here."
            ),
            query_lines=[1],
            page=1,
            bbox=[50, 50, 550, 60],
            mapping_status="mapped",
            match_status="fallback_pdf",
        )
        markdown = MODULE.nodes_to_markdown([line], 2)
        contract = MODULE.build_strict_text_v2_contract(
            markdown=markdown,
            ordered_nodes=[line],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[block],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertIn(
            "inline_enrichment_unresolved:inline-math",
            contract["failure_reasons"],
        )

    def test_strict_contract_rejects_atomic_footnote_fallback(self) -> None:
        line = self.strict_line(
            "Several visible words end with a note1.", "line-a", 0
        )
        inventory = MODULE.freeze_line_inventory([line], 792)
        markdown = MODULE.nodes_to_markdown([line], 2)
        footnote_audit = {
            "status": "failed",
            "representation": "html_sup",
            "definition_node_id": "footnotes-page-0001",
            "total": 1,
            "structured": 0,
            "fallback": 1,
            "failure_reason": "definition_body_alignment_failed",
            "notes": [
                {
                    "note_id": "i1-fn01",
                    "marker": "1",
                    "status": "fallback",
                    "failure_reason": "definition_body_alignment_failed",
                    "content_validation_status": "failed",
                    "content_validation_issues": ["definition_body_alignment_failed"],
                    "callout_line_ids": ["line-a"],
                    "definition_line_ids": [],
                    "callout_markdown_occurrences": 0,
                    "definition_markdown_occurrences": 0,
                    "total_sup_occurrences": 0,
                    "representation": "html_sup",
                }
            ],
        }
        contract = MODULE.build_strict_text_v2_contract(
            markdown=markdown,
            ordered_nodes=[line],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
            footnote_audit=footnote_audit,
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertIn("footnote_structuring_unresolved", contract["failure_reasons"])
        self.assertTrue(
            contract["contract"]["strict_footnote_structure_hard_gate"]
        )

    def test_strict_contract_rejects_legacy_markdown_footnote_syntax(self) -> None:
        line = self.strict_line(
            "Several complete visible words remain in this sentence.", "line-a", 0
        )
        inventory = MODULE.freeze_line_inventory([line], 792)
        contract = MODULE.build_strict_text_v2_contract(
            markdown="Several complete visible words remain in this sentence.[^1]\n",
            ordered_nodes=[line],
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertIn(
            "legacy_markdown_footnote_syntax_present",
            contract["failure_reasons"],
        )

    def test_heading_integration_preserves_compiled_number_prefixes(self) -> None:
        cases = [
            ("3.2.4.", "Inter-Annotator Agreement", 4),
            ("IV.", "Asymptotics", 2),
            ("A.", "Prerequisites", 3),
        ]
        for prefix, title, level in cases:
            with self.subTest(prefix=prefix):
                block = MODULE.SourceBlock(
                    block_id="h1",
                    kind="heading",
                    source_file=Path("main.tex"),
                    start_line=1,
                    end_line=1,
                    raw_latex=title,
                    markdown="#" * level + " " + title,
                    query_lines=[1],
                    heading_level=level,
                    heading_command="section",
                    heading_source_title=title,
                )
                node = MODULE.PageNode(
                    "text", f"{prefix} {title}", [50, 100, 300, 115], 12
                )
                retained, structured, audit = MODULE.integrate_source_blocks(
                    [node], [block], 612
                )
                self.assertEqual(retained, [])
                self.assertEqual(structured[0].text, "#" * level + f" {prefix} {title}")
                self.assertEqual(block.pdf_visible_heading, f"{prefix} {title}")
                self.assertEqual(block.visible_number_prefix, prefix)
                self.assertEqual(block.heading_number_status, "preserved")
                self.assertTrue(audit["heading_numbering"]["strict"])

    def test_wrapped_two_column_heading_matches_with_interleaved_other_lane(self) -> None:
        title = "Existing Low Resource Language NER Dataset"
        block = MODULE.SourceBlock(
            block_id="h2",
            kind="heading",
            source_file=Path("main.tex"),
            start_line=2,
            end_line=2,
            raw_latex=title,
            markdown="### " + title,
            query_lines=[2],
            heading_level=3,
            heading_command="subsection",
            heading_source_title=title,
        )
        nodes = [
            MODULE.PageNode("text", "2.1. Existing Low Resource Language", [330, 100, 560, 111], 11),
            MODULE.PageNode("text", "Unrelated left-column prose", [50, 105, 280, 116], 10),
            MODULE.PageNode("text", "NER Dataset", [330, 112, 430, 123], 11),
        ]
        retained, structured, audit = MODULE.integrate_source_blocks(nodes, [block], 612)
        self.assertEqual([node.text for node in retained], ["Unrelated left-column prose"])
        self.assertEqual(structured[0].text, "### 2.1. " + title)
        self.assertEqual(block.heading_matched_line_count, 2)
        self.assertEqual(audit["heading_numbering"]["preserved"], 1)

    def test_starred_heading_never_invents_or_strips_number(self) -> None:
        block = MODULE.SourceBlock(
            block_id="h3",
            kind="heading",
            source_file=Path("main.tex"),
            start_line=3,
            end_line=3,
            raw_latex="Acknowledgements",
            markdown="## Acknowledgements",
            query_lines=[3],
            heading_level=2,
            heading_command="section",
            heading_starred=True,
            heading_source_title="Acknowledgements",
        )
        node = MODULE.PageNode("text", "Acknowledgements", [50, 100, 250, 115], 12)
        retained, structured, audit = MODULE.integrate_source_blocks([node], [block], 612)
        self.assertEqual(retained, [])
        self.assertEqual(structured[0].text, "## Acknowledgements")
        self.assertIsNone(block.visible_number_prefix)
        self.assertEqual(block.heading_number_status, "unnumbered")
        self.assertTrue(audit["heading_numbering"]["strict"])

        suspicious = MODULE.SourceBlock(
            block_id="h4",
            kind="heading",
            source_file=Path("main.tex"),
            start_line=4,
            end_line=4,
            raw_latex="Acknowledgements",
            markdown="## Acknowledgements",
            query_lines=[4],
            heading_level=2,
            heading_command="section",
            heading_starred=True,
            heading_source_title="Acknowledgements",
        )
        numbered_node = MODULE.PageNode(
            "text", "A. Acknowledgements", [50, 100, 250, 115], 12
        )
        retained, structured, audit = MODULE.integrate_source_blocks(
            [numbered_node], [suspicious], 612
        )
        self.assertEqual([node.text for node in retained], ["A. Acknowledgements"])
        self.assertEqual(structured, [])
        self.assertEqual(suspicious.heading_number_status, "ambiguous")
        self.assertFalse(audit["heading_numbering"]["strict"])

    def test_ambiguous_duplicate_heading_falls_back_without_removal(self) -> None:
        block = MODULE.SourceBlock(
            block_id="h5",
            kind="heading",
            source_file=Path("main.tex"),
            start_line=5,
            end_line=5,
            raw_latex="Introduction",
            markdown="## Introduction",
            query_lines=[5],
            heading_level=2,
            heading_command="section",
            heading_source_title="Introduction",
        )
        nodes = [
            MODULE.PageNode("text", "1. Introduction", [50, 100, 250, 115], 12),
            MODULE.PageNode("text", "1. Introduction", [50, 300, 250, 315], 12),
        ]
        retained, structured, audit = MODULE.integrate_source_blocks(nodes, [block], 612)
        self.assertEqual(retained, nodes)
        self.assertEqual(structured, [])
        self.assertEqual(audit["removed_pdf_lines"], 0)
        self.assertEqual(block.heading_structure_status, "fallback_unmatched")
        self.assertEqual(block.heading_number_status, "ambiguous")

    def test_run_in_heading_keeps_trailing_pdf_prose(self) -> None:
        block = MODULE.SourceBlock(
            block_id="h6",
            kind="heading",
            source_file=Path("main.tex"),
            start_line=6,
            end_line=6,
            raw_latex="Dataset Statistics",
            markdown="##### Dataset Statistics",
            query_lines=[6],
            heading_level=5,
            heading_command="paragraph",
            heading_source_title="Dataset Statistics",
        )
        node = MODULE.PageNode(
            "text", "Dataset Statistics Figure 4 shows the distribution.", [50, 100, 500, 115], 10
        )
        retained, structured, audit = MODULE.integrate_source_blocks([node], [block], 612)
        self.assertEqual(retained, [])
        self.assertEqual([item.text for item in structured], [
            "##### Dataset Statistics",
            "Figure 4 shows the distribution.",
        ])
        self.assertEqual(audit["heading_numbering"]["unnumbered"], 1)

    def test_run_in_heading_accepts_long_trailing_prose_in_one_inline_node(self) -> None:
        title = "Agents' incentives and tests' induced welfare."
        block = MODULE.SourceBlock(
            block_id="h-long-runin",
            kind="heading",
            source_file=Path("main.tex"),
            start_line=9,
            end_line=9,
            raw_latex=title,
            markdown="##### " + title,
            query_lines=[9],
            heading_level=5,
            heading_command="paragraph",
            heading_source_title=title,
        )
        node = self.strict_line(
            "Agents’ incentives and tests’ induced welfare. We adopt an extreme "
            "perspective and assume that the sole objective of both agents is to "
            "be selected by the test, with additional trailing prose.",
            "runin-line",
            0,
            [72, 100, 540, 140],
        )
        node.inline_source_ids = ["i-runin"]
        node.source_paragraph_id = "sp-runin"
        node.source_paragraph_slice_id = "sp-runin@p0001"
        retained, structured, audit = MODULE.integrate_source_blocks(
            [node], [block], 612
        )
        self.assertEqual(retained, [])
        self.assertEqual(structured[0].text, "##### " + title)
        self.assertTrue(structured[1].text.startswith("We adopt an extreme perspective"))
        self.assertEqual(structured[1].source_paragraph_id, "sp-runin")
        self.assertEqual(structured[1].inline_source_ids, ["i-runin"])
        self.assertEqual(audit["heading_numbering"]["unnumbered"], 1)

    def test_page_title_lines_merge_and_plain_hash_is_escaped(self) -> None:
        nodes = [
            MODULE.PageNode("text", "A Long Paper", [100, 40, 500, 60], 18),
            MODULE.PageNode("text", "Title Across Lines", [100, 65, 500, 85], 18),
            MODULE.PageNode("text", "Author Name", [200, 100, 400, 112], 11),
            MODULE.PageNode("text", "# U j1 intersects U j2", [50, 220, 500, 232], 10),
        ]
        markdown = MODULE.nodes_to_markdown(nodes, 1)
        heading_lines = [line for line in markdown.splitlines() if line.startswith("# ")]
        self.assertEqual(heading_lines, ["# A Long Paper Title Across Lines"])
        self.assertIn(r"\# U j1 intersects U j2", markdown)

    def test_extract_source_hyphenated_terms_is_deterministic_and_ignores_commands(self) -> None:
        source = r"""
single-parameter and \textit{QCRB-achieving}; a two-
stage method and Ｆｕｌｌ-width input.
\singleparameter-command must not contribute a command name.
$x-y$ and $alpha-beta$ are mathematical. % hidden-source-term
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.tex").write_text(source, encoding="utf-8")
            (root / "a.tex").write_text("single-parameter", encoding="utf-8")
            terms = MODULE.extract_source_hyphenated_terms(root)
        self.assertEqual(
            terms,
            [
                "full-width",
                "qcrb-achieving",
                "single-parameter",
                "two-stage",
            ],
        )

    def test_extract_source_hyphenated_terms_handles_unmatched_math_delimiter_linearly(self) -> None:
        # Real submissions occasionally contain a dollar delimiter whose mate
        # is supplied by a macro or is otherwise absent from the raw source.
        # A regex with overlapping escaped-character/plain-character branches
        # used to backtrack exponentially on this shape.
        source = "$" + (r"\command{value}" * 20_000) + " visible-term"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.tex").write_text(source, encoding="utf-8")
            terms = MODULE.extract_source_hyphenated_terms(root)
        self.assertEqual(terms, ["visible-term"])

    def test_synctex_compile_uses_nested_main_tex_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            main_dir = source_root / "Arxiv"
            main_dir.mkdir(parents=True)
            main_tex = main_dir / "main.tex"
            main_tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
            build_dir = root / "build"
            observed: dict[str, object] = {}

            def fake_run(command, *, cwd, env, log_path, timeout_seconds, label):
                observed["cwd"] = cwd
                build_dir.mkdir(parents=True, exist_ok=True)
                (build_dir / "main.pdf").write_bytes(b"%PDF-probe")
                (build_dir / "main.synctex.gz").write_bytes(b"synctex-probe")
                return 0, False, 0.1

            paper = {
                "stem": "nested-main",
                "source_dir": str(source_root),
                "main_tex": str(main_tex),
                "compile": {"engine": "pdflatex"},
            }
            with mock.patch.object(MODULE, "run_with_heartbeat", side_effect=fake_run):
                MODULE.compile_with_synctex(paper, build_dir, resume=False)
            self.assertEqual(observed["cwd"], main_dir.resolve())

    def test_synctex_resume_rebuilds_nonempty_cache_without_success_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            main_tex = source_root / "main.tex"
            main_tex.write_text(r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
            build_dir = root / "build"
            build_dir.mkdir()
            # This is the exact failure shape left by a TeX process that dies
            # during final PDF/SyncTeX writes: both files are non-empty but no
            # successful compile_info was committed.
            (build_dir / "main.pdf").write_bytes(b"truncated-pdf")
            (build_dir / "main.synctex.gz").write_bytes(b"truncated-synctex")
            calls = 0

            def fake_run(command, *, cwd, env, log_path, timeout_seconds, label):
                nonlocal calls
                calls += 1
                (build_dir / "main.pdf").write_bytes(b"%PDF-rebuilt")
                (build_dir / "main.synctex.gz").write_bytes(b"synctex-rebuilt")
                return 0, False, 0.1

            paper = {
                "stem": "invalid-cache",
                "source_dir": str(source_root),
                "main_tex": str(main_tex),
                "compile": {"engine": "pdflatex"},
            }
            with mock.patch.object(MODULE, "run_with_heartbeat", side_effect=fake_run):
                MODULE.compile_with_synctex(paper, build_dir, resume=True)
            self.assertEqual(calls, 1)

    def test_strict_punctuation_rejects_spaced_and_newline_hyphenation(self) -> None:
        issues = MODULE.strict_punctuation_issues(
            "The ORGA- NIZATION uses a Low-\nResource corpus.", []
        )
        self.assertEqual(
            issues,
            [
                "intra_page_hyphenation_residue:low-resource",
                "intra_page_hyphenation_residue:orga-nization",
            ],
        )
        self.assertEqual(
            VERIFY_MODULE.independent_hyphenation_residue_issues(
                "The ORGA- NIZATION uses a Low-\nResource corpus."
            ),
            issues,
        )

    def test_strict_punctuation_rejects_uncovered_collapsed_source_term(self) -> None:
        self.assertEqual(
            MODULE.strict_punctuation_issues(
                "A singleparameter estimator is used.", ["single-parameter"]
            ),
            ["source_hyphen_collapsed:single-parameter"],
        )
        self.assertEqual(
            MODULE.strict_punctuation_issues(
                "singleparameter and single-parameter are paired.",
                ["single-parameter"],
            ),
            [],
        )

    def test_strict_punctuation_accepts_normal_hyphen_and_ignores_math_or_single_letter(self) -> None:
        markdown = (
            "A single-parameter method uses $alpha- beta$ and "
            "$singleparameter$ with k- bonacci notation."
        )
        self.assertEqual(
            MODULE.strict_punctuation_issues(markdown, ["single-parameter"]),
            [],
        )
        self.assertEqual(
            VERIFY_MODULE.independent_hyphenation_residue_issues(markdown), []
        )

    def test_strict_punctuation_issue_report_aggregation(self) -> None:
        rows = [
            {
                "strict_punctuation_issues": [
                    "intra_page_hyphenation_residue:low-resource",
                    "source_hyphen_collapsed:single-parameter",
                ]
            },
            {"strict_punctuation_issues": []},
            {
                "strict_punctuation_issues": [
                    "source_hyphen_collapsed:single-parameter"
                ]
            },
        ]
        expected = {
            "pages_with_issues": 2,
            "total_issues": 3,
            "by_type": {
                "intra_page_hyphenation_residue": 1,
                "source_hyphen_collapsed": 2,
            },
            "by_issue": {
                "intra_page_hyphenation_residue:low-resource": 1,
                "source_hyphen_collapsed:single-parameter": 2,
            },
        }
        self.assertEqual(MODULE.aggregate_strict_punctuation_issues(rows), expected)
        self.assertEqual(
            VERIFY_MODULE.aggregate_serialized_punctuation_issues(rows), expected
        )

    def test_strict_punctuation_issue_is_a_contract_hard_failure(self) -> None:
        nodes = [
            self.strict_line(
                "The Low- Resource corpus remains visible.", "line-p", 0
            )
        ]
        inventory = MODULE.freeze_line_inventory(nodes, 792)
        issue = "intra_page_hyphenation_residue:low-resource"
        contract = MODULE.build_strict_text_v2_contract(
            markdown="The Low- Resource corpus remains visible.\n",
            ordered_nodes=nodes,
            inventory=inventory,
            page_number=1,
            layout="single_column",
            page_blocks=[],
            inline_blocks=[],
            punctuation_issues=[issue],
        )
        self.assertEqual(contract["strict_text_v2_status"], "failed")
        self.assertEqual(contract["punctuation_issues"], [issue])
        self.assertIn(issue, contract["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
