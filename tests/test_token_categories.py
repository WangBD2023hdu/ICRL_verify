from __future__ import annotations

import math

import pytest

from qwen_mm_token_probe.token_categories import (
    annotate_token_categories,
    summarize_token_categories,
)


def _rows(pieces: list[str]) -> list[dict]:
    return [
        {
            "index": index,
            "token_id": index,
            "raw_token": piece,
            "mutation_ids": "",
            "p_original": 0.5,
            "p_teacher": 0.25,
            "delta_logp_teacher_minus_original": math.log(0.5),
        }
        for index, piece in enumerate(pieces)
    ]


def test_markdown_and_html_classification_across_token_boundaries() -> None:
    pieces = [
        "```",
        "markdown",
        "\n",
        "##",
        " ",
        "Heading",
        "\n\n",
        "<",
        "table",
        "><tr><t",
        "d>",
        "cell",
        "</td></tr></table>",
        "\n",
        "```",
    ]
    rows = _rows(pieces)
    annotate_token_categories(rows, response_text="".join(pieces))
    assert [row["index"] for row in rows if row["token_category"] == "body"] == [5, 11]
    assert all(
        row["token_category"] == "formatting"
        for row in rows
        if row["index"] not in (5, 11)
    )


def test_mixed_token_assigned_once_with_mutation_priority() -> None:
    rows = _rows(["<td>word", "</td>", " ", "<td>mut", "ated", "</td>"])
    rows[3]["mutation_ids"] = "m001"
    rows[4]["mutation_ids"] = "m001"
    annotate_token_categories(
        rows, response_text="".join(row["raw_token"] for row in rows)
    )
    assert [row["token_category"] for row in rows] == [
        "body",
        "formatting",
        "formatting",
        "mutation",
        "mutation",
        "formatting",
    ]
    assert rows[0]["mixed_format_content"] is True
    assert rows[3]["mixed_format_content"] is True
    assert rows[0]["format_char_count"] == 4
    assert rows[0]["body_char_count"] == 4
    assert sum(row["token_count"] for row in summarize_token_categories(rows)) == len(
        rows
    )


def test_math_and_inline_code_are_content_not_markdown_markers() -> None:
    text = "$x_1 > y$ and `# literal` and C#"
    rows = _rows(list(text))
    annotate_token_categories(rows, response_text=text)
    for index in (text.index("_"), text.index(">"), text.index("#"), text.rindex("#")):
        assert rows[index]["token_category"] == "body"
    assert rows[text.index("`")]["token_category"] == "formatting"


def test_lists_emphasis_links_and_pipe_table_syntax() -> None:
    text = "1. **bold** and _italic_ [link](url)\n| A | B |\n|---|---|"
    rows = _rows(list(text))
    annotate_token_categories(rows, response_text=text)
    for symbol in ("1", ".", "*", "_", "[", "]", "(", ")", "|", "-"):
        assert rows[text.index(symbol)]["token_category"] == "formatting"
    for word in ("bold", "italic", "link", "url", "A", "B"):
        assert rows[text.index(word)]["token_category"] == "body"


def test_probability_summary_micro_average_and_empty_category() -> None:
    rows = _rows(["a", "b", "c", "\n"])
    for row, original, teacher in zip(rows, [0.1, 0.7, 0.4, 0.8], [0.2, 0.5, 0.4, 0.1]):
        row.update(
            p_original=original,
            p_teacher=teacher,
            delta_logp_teacher_minus_original=math.log(teacher / original),
        )
    before = [(row["p_original"], row["p_teacher"]) for row in rows]
    annotate_token_categories(rows, response_text="abc\n")
    summary = {row["token_category"]: row for row in summarize_token_categories(rows)}
    assert summary["body"]["token_count"] == 3
    assert summary["body"]["mean_p_original"] == pytest.approx(0.4)
    assert summary["body"]["mean_p_teacher"] == pytest.approx(1.1 / 3)
    assert summary["body"]["teacher_lower_probability_rate"] == pytest.approx(1 / 3)
    assert summary["body"]["mean_delta_p_teacher_minus_original"] == pytest.approx(
        -0.1 / 3
    )
    assert summary["mutation"]["token_count"] == 0
    assert summary["mutation"]["mean_p_teacher"] is None
    assert summary["mutation"]["teacher_lower_probability_rate"] is None
    assert before == [(row["p_original"], row["p_teacher"]) for row in rows]


def test_decode_mismatch_is_exposed_not_silently_indexed_into_other_text() -> None:
    rows = _rows(["#", " ", "heading"])
    annotate_token_categories(rows, response_text="different text")
    assert all(row["category_surface_matches_response"] is False for row in rows)
    assert [row["token_category"] for row in rows] == [
        "formatting",
        "formatting",
        "body",
    ]
