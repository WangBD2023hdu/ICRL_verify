from __future__ import annotations

from qwen_mm_token_probe.mutation_spans import resolve_mutation_spans


def test_provided_span_disambiguates_repeated_word() -> None:
    ground_truth = "repeat here; repeat there"
    changes = [{"ocr_ans": "repeat", "markdown_span": [13, 19], "bbox": [1, 2]}]

    resolved = resolve_mutation_spans(ground_truth, changes)

    assert resolved == [
        {
            "ocr_ans": "repeat",
            "markdown_span": [13, 19],
            "bbox": [1, 2],
            "span_resolution": "provided",
            "span_match_count": 2,
        }
    ]
    assert (
        ground_truth[resolved[0]["markdown_span"][0] : resolved[0]["markdown_span"][1]]
        == "repeat"
    )


def test_unique_match_uses_whole_word_not_substring() -> None:
    ground_truth = "scatter cat catalog"

    resolved = resolve_mutation_spans(ground_truth, [{"ocr_ans": "cat"}])

    assert resolved[0]["markdown_span"] == [8, 11]
    assert resolved[0]["span_resolution"] == "unique_gt_match"
    assert resolved[0]["span_match_count"] == 1


def test_repeated_whole_word_is_left_unresolved() -> None:
    ground_truth = "cat scatter cat"

    resolved = resolve_mutation_spans(ground_truth, [{"ocr_ans": "cat"}])

    assert resolved[0]["markdown_span"] == []
    assert resolved[0]["span_resolution"] == "ambiguous_gt_match"
    assert resolved[0]["span_match_count"] == 2


def test_absent_word_and_missing_word_do_not_run_empty_regex() -> None:
    changes = [{"ocr_ans": "missing"}, {"origin_ans": "old"}]

    resolved = resolve_mutation_spans("a document", changes)

    assert resolved[0]["markdown_span"] == []
    assert resolved[0]["span_resolution"] == "not_found"
    assert resolved[0]["span_match_count"] == 0
    assert resolved[1]["markdown_span"] == []
    assert resolved[1]["span_resolution"] == "not_found"
    assert resolved[1]["span_match_count"] == 0


def test_input_dictionaries_are_not_modified() -> None:
    changes = [{"ocr_ans": "word", "markdown_span": [0, 1], "extra": "kept"}]
    original = [dict(change) for change in changes]

    resolved = resolve_mutation_spans("word", changes)

    assert changes == original
    assert resolved[0] is not changes[0]
    assert resolved[0]["markdown_span"] == [0, 4]


def test_char_only_span_expands_to_unique_full_word() -> None:
    ground_truth = "prefix edited suffix"
    changes = [{"ocr_ans": "edited", "markdown_span": [8, 9]}]

    resolved = resolve_mutation_spans(ground_truth, changes)

    assert resolved[0]["markdown_span"] == [7, 13]
    assert resolved[0]["span_resolution"] == "unique_gt_match"
    assert resolved[0]["span_match_count"] == 1


def test_html_table_offsets_are_raw_string_offsets_and_bbox_is_ignored() -> None:
    ground_truth = "<table><tr><td>edited</td><td>other</td></tr></table>"
    changes = [{"ocr_ans": "edited", "bbox": [900, 800, 950, 820]}]

    resolved = resolve_mutation_spans(ground_truth, changes)
    start, end = resolved[0]["markdown_span"]

    assert (start, end) == (
        ground_truth.index("edited"),
        ground_truth.index("edited") + len("edited"),
    )
    assert ground_truth[start:end] == "edited"
    assert resolved[0]["span_resolution"] == "unique_gt_match"
