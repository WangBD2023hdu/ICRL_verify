#!/usr/bin/env python3
"""Independently verify a source-recompiled confusable OCR pilot."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any

import pdfplumber
from PIL import Image, ImageChops


WORD_RE = re.compile(r"[A-Za-z]{4,}")
CONFUSABLE_PAIRS = {
    ("a", "o"),
    ("c", "e"),
    ("c", "o"),
    ("e", "c"),
    ("g", "q"),
    ("h", "n"),
    ("i", "l"),
    ("l", "i"),
    ("n", "h"),
    ("o", "a"),
    ("o", "c"),
    ("q", "g"),
    ("s", "z"),
    ("u", "v"),
    ("v", "u"),
    ("z", "s"),
}
MUTATION_POLICY_VERSION = "chaos_visual_v2"
SELECTION_POLICY_VERSION = (
    "page_exact_source_paragraph_v6_rendered_line_spread_current_gt_no_bibliography"
)
BIBLIOGRAPHY_POLICY_VERSION = "exclude_bibliography_tail_v1"
STRICT_INPUT_FILTER_POLICY_VERSION = "strict_gt_current_contract_v1"
STRICT_INPUT_STRICT_TEXT_CONTRACT_VERSION = 2
STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION = 5
SOURCE_FIRST_INPUT_POLICY_VERSION = "source_first_color_v6_literal_markdown_v5"
SOURCE_FIRST_SCHEMA_VERSION = 6
SOURCE_FIRST_CONTRACT = "source_first_color_v6"
SOURCE_FIRST_VERIFIER_CONTRACT_VERSION = 4
SOURCE_FIRST_PROBE_POLICY_VERSION = (
    "paragraph_list_payload_then_paragraph_then_whole_v2"
)
SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION = "exact_page_character_sequence_v1"
SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION = "aux_number_unique_titleformat_label_v1"
BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?:(?:appendix\s+)?[A-Z0-9]+(?:\.[A-Z0-9]+)*[.)]?\s+)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited)\s*:?[\s*_]*$",
    re.IGNORECASE,
)
VERL_PROMPT = (
    "<image>\nPlease transcribe all text in this page image faithfully, "
    "exactly as printed (including any typos)."
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve_clean_artifact(clean_gt_root: Path, relative: str) -> Path:
    direct = clean_gt_root / relative
    if direct.is_file():
        return direct
    matches = sorted(clean_gt_root.glob(f"shard_*/{relative}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one clean artifact for {relative!r}; found {len(matches)}"
        )
    return matches[0]


def markdown_has_bibliography_heading(markdown: str) -> bool:
    return any(
        BIBLIOGRAPHY_HEADING_RE.fullmatch(line)
        for line in markdown.splitlines()
        if line.strip()
    )


def markdown_is_table_of_contents_page(markdown: str) -> bool:
    nonblank = [line.strip() for line in markdown.splitlines() if line.strip()]
    if any(
        re.fullmatch(r"(?:#{1,6}\s*)?(?:table\s+of\s+)?contents\s*:?[\s*_]*", line, re.I)
        for line in nonblank
    ):
        return True
    isolated_page_numbers = sum(
        bool(re.fullmatch(r"(?:[ivxlcdm]+|\d{1,4})", line, re.I))
        for line in nonblank
    )
    return isolated_page_numbers >= 2


def bibliography_start_page_for_clean_row(
    clean_gt_root: Path, clean_row: dict[str, Any]
) -> int | None:
    page_dir = resolve_clean_artifact(
        clean_gt_root, str(clean_row["markdown"])
    ).parent
    starts: list[int] = []
    for markdown_path in sorted(page_dir.glob("page_*.md")):
        match = re.fullmatch(r"page_(\d+)", markdown_path.stem)
        if match is None:
            continue
        markdown = markdown_path.read_text(encoding="utf-8")
        if (
            markdown_has_bibliography_heading(markdown)
            and not markdown_is_table_of_contents_page(markdown)
        ):
            starts.append(int(match.group(1)))
    return min(starts) if starts else None


def resolve_clean_pdf(clean_gt_root: Path, paper_id: str, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path.resolve()
    filename = path.name
    matches = list(
        clean_gt_root.glob(f"shard_*/papers/{paper_id}/synctex_build/{filename}")
    )
    direct = clean_gt_root / "papers" / paper_id / "synctex_build" / filename
    if direct.is_file():
        matches.append(direct)
    matches = sorted(set(item.resolve() for item in matches if item.is_file()))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one rebased clean PDF for {paper_id}; found {len(matches)} "
            f"matching {filename!r} under {clean_gt_root}"
        )
    return matches[0]


def load_clean_page_index(
    clean_gt_root: Path, pairs: list[dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if pairs is not None:
        for pair in pairs:
            paper_id = str(pair["paper_id"])
            page_number = int(pair["page_number"])
            relative = Path("papers") / paper_id / "pages" / f"page_{page_number:04d}.json"
            matches = []
            direct = clean_gt_root / relative
            if direct.is_file():
                matches.append(direct)
            matches.extend(clean_gt_root.glob(f"shard_*/{relative}"))
            unique_matches = sorted(set(matches))
            if len(unique_matches) != 1:
                raise FileNotFoundError(
                    f"expected one clean sidecar for {paper_id} page {page_number}; "
                    f"found {len(unique_matches)}"
                )
            row = read_json(unique_matches[0])
            data_id = str(row["data_id"])
            if data_id in index and index[data_id] != row:
                raise ValueError(f"duplicate clean data_id: {data_id}")
            index[data_id] = row
        return index
    paths = list(clean_gt_root.glob("papers/*/pages/page_*.json"))
    paths.extend(clean_gt_root.glob("shard_*/papers/*/pages/page_*.json"))
    for path in sorted(paths):
        row = read_json(path)
        data_id = str(row["data_id"])
        if data_id in index:
            raise ValueError(f"duplicate clean data_id: {data_id}")
        index[data_id] = row
    return index


def load_source_first_clean_page_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("source-first case manifest must be a JSON array")
    index: dict[str, dict[str, Any]] = {}
    for case in cases:
        markdown_path = Path(str(case["markdown_path"]))
        image_path = Path(str(case["image"]))
        if not markdown_path.is_absolute():
            markdown_path = manifest_path.parent / markdown_path
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        markdown_path = markdown_path.resolve()
        image_path = image_path.resolve()
        sidecar = read_json(markdown_path.with_suffix(".json"))
        result_root = markdown_path.parent.parent
        report = read_json(result_root / "validation_report.json")
        verifier_value = sidecar.get("verifier", {})
        verifier = verifier_value if isinstance(verifier_value, dict) else {}
        source_probes_path = result_root / "source_probes.jsonl"
        source_probes = read_jsonl(source_probes_path) if source_probes_path.is_file() else []
        known_probe_ids = [str(row.get("probe_id") or "") for row in source_probes]
        page_probe_ids = sidecar.get("source_probe_ids")
        shadow_invariant = sidecar.get("shadow_invariant")
        figure_policy = report.get("figure_policy")
        figure_status = (report.get("figure_removal") or {}).get("status")
        valid_figure_policy = (
            (figure_policy == "drop_figures" and figure_status == "passed")
            or (figure_policy == "keep_figures" and figure_status == "disabled")
        )
        ordered_content_match = (
            verifier.get("exact_ordered_character_stream_match") is True
        )
        required = {
            "page_schema": sidecar.get("schema_version") == SOURCE_FIRST_SCHEMA_VERSION,
            "page_contract": sidecar.get("contract") == SOURCE_FIRST_CONTRACT,
            "page_probe_policy": sidecar.get("probe_policy_version")
            == SOURCE_FIRST_PROBE_POLICY_VERSION,
            "page_shadow_invariant_policy": sidecar.get(
                "shadow_invariant_policy_version"
            )
            == SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION,
            "page_heading_label_policy": sidecar.get(
                "heading_label_policy_version"
            )
            == SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION,
            "page_shadow_text_identity": isinstance(shadow_invariant, dict)
            and shadow_invariant.get("character_count_equal") is True
            and shadow_invariant.get("character_text_equal") is True,
            "page_shadow_geometry_role": isinstance(shadow_invariant, dict)
            and shadow_invariant.get("geometry_role") == "diagnostic_only",
            "page_figure_policy": sidecar.get("figure_policy") == figure_policy,
            "page_status": sidecar.get("status") == "passed",
            "generation_source": sidecar.get("generation_source") == "latex_source",
            "page_provenance": sidecar.get("page_provenance") == "compiled_vector_color",
            "pdf_role": sidecar.get("pdf_role") == "independent_verifier_only",
            "verifier_status": verifier.get("status") == "passed",
            "verifier_contract_version": verifier.get("contract_version")
            == SOURCE_FIRST_VERIFIER_CONTRACT_VERSION,
            "ordered_content_match": ordered_content_match,
            "report_schema": report.get("schema_version") == SOURCE_FIRST_SCHEMA_VERSION,
            "report_contract": report.get("contract") == SOURCE_FIRST_CONTRACT,
            "report_probe_policy": report.get("probe_policy_version")
            == SOURCE_FIRST_PROBE_POLICY_VERSION,
            "report_shadow_invariant_policy": report.get(
                "shadow_invariant_policy_version"
            )
            == SOURCE_FIRST_SHADOW_INVARIANT_POLICY_VERSION,
            "report_heading_label_policy": report.get(
                "heading_label_policy_version"
            )
            == SOURCE_FIRST_HEADING_LABEL_POLICY_VERSION,
            "report_figure_policy": valid_figure_policy,
            "report_reference_removal": (report.get("reference_removal") or {}).get("status")
            == "passed",
            "report_status": report.get("status") == "passed",
            "pdf_not_generator": report.get("pdf_used_for_generation") is False,
            "pdf_is_verifier": report.get("pdf_used_for_verification") is True,
            "markdown_exists": markdown_path.is_file(),
            "image_exists": image_path.is_file(),
            "source_probes_exists": source_probes_path.is_file(),
            "source_probe_inventory": bool(known_probe_ids)
            and all(known_probe_ids)
            and len(known_probe_ids) == len(set(known_probe_ids)),
            "source_probe_ids": isinstance(page_probe_ids, list)
            and bool(page_probe_ids)
            and set(map(str, page_probe_ids)) <= set(known_probe_ids),
            "manifest_id": str(case.get("pair_id")) == str(sidecar.get("data_id")),
        }
        failures = sorted(key for key, passed in required.items() if not passed)
        if failures:
            raise ValueError(
                f"source-first clean reference failed for {case.get('pair_id')}: {failures}"
            )
        normalized = dict(sidecar)
        normalized.update(
            {
                "markdown": str(markdown_path),
                "image": str(image_path),
                "source_pdf": str(Path(str(report["clean_pdf"])).resolve()),
                "source_first_input": True,
            }
        )
        data_id = str(normalized["data_id"])
        if data_id in index:
            raise ValueError(f"duplicate source-first data_id: {data_id}")
        index[data_id] = normalized
    return index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pdf_words(page: Any) -> list[dict[str, Any]]:
    return [
        {
            "text": str(word.get("text", "")),
            "x0": float(word["x0"]),
            "top": float(word["top"]),
            "x1": float(word["x1"]),
            "bottom": float(word["bottom"]),
        }
        for word in page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
            split_at_punctuation=True,
        )
    ]


def char_differences(clean: str, edited: str) -> list[tuple[int, str, str]]:
    if len(clean) != len(edited):
        return []
    return [
        (index, left, right)
        for index, (left, right) in enumerate(zip(clean, edited))
        if left != right
    ]


def expected_pixel_bbox(
    word: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> list[int]:
    return [
        max(0, min(image_width, math.floor(word["x0"] * image_width / page_width))),
        max(0, min(image_height, math.floor(word["top"] * image_height / page_height))),
        max(0, min(image_width, math.ceil(word["x1"] * image_width / page_width))),
        max(0, min(image_height, math.ceil(word["bottom"] * image_height / page_height))),
    ]


def one_char_confusion(origin: str, edited: str) -> tuple[str, str] | None:
    if len(origin) != len(edited) or not WORD_RE.fullmatch(origin) or not WORD_RE.fullmatch(edited):
        return None
    differences = [(left, right) for left, right in zip(origin, edited) if left != right]
    if len(differences) != 1 or differences[0] not in CONFUSABLE_PAIRS:
        return None
    return differences[0]


def verify_pair(
    *,
    root: Path,
    clean_gt_root: Path,
    clean_row: dict[str, Any],
    pair: dict[str, Any],
    paper_result: dict[str, Any],
    expected_input_policy: str = STRICT_INPUT_FILTER_POLICY_VERSION,
) -> list[str]:
    errors: list[str] = []
    pair_id = str(pair["pair_id"])
    metadata = read_json(root / pair["metadata"])
    if metadata["pair_id"] != pair_id:
        errors.append("metadata_pair_id_mismatch")
    if metadata.get("mutation_policy_version") != MUTATION_POLICY_VERSION:
        errors.append("mutation_policy_version_mismatch")
    if metadata.get("selection_policy_version") != SELECTION_POLICY_VERSION:
        errors.append("selection_policy_version_mismatch")
    if (
        metadata.get("strict_input_filter_policy_version")
        != expected_input_policy
    ):
        errors.append("strict_input_filter_policy_version_mismatch")
    if (
        pair.get("strict_input_filter_policy_version")
        != expected_input_policy
    ):
        errors.append("pair_strict_input_filter_policy_version_mismatch")
    if metadata.get("bibliography_policy_version") != BIBLIOGRAPHY_POLICY_VERSION:
        errors.append("bibliography_policy_version_mismatch")
    if metadata.get("bibliography_content_present") is not False:
        errors.append("bibliography_content_contract_missing")
    changes = metadata.get("changes", [])
    if len(changes) not in (3, 4) or pair.get("mutation_count") != len(changes):
        errors.append("mutation_count_not_3_or_4")
    labels = []
    for change in changes:
        confusion = one_char_confusion(change["origin_ans"], change["ocr_ans"])
        if confusion is None:
            errors.append(f"invalid_one_char_confusion:{change['origin_ans']}->{change['ocr_ans']}")
        if any(character.isdigit() for character in change["origin_ans"] + change["ocr_ans"]):
            errors.append("digit_mutation")
        labels.append((change["origin_ans"], change["ocr_ans"]))
    if len(labels) != len(set(labels)):
        errors.append("duplicate_change")

    clean_md = resolve_clean_artifact(
        clean_gt_root, str(clean_row["markdown"])
    ).read_text(encoding="utf-8")
    if clean_row.get("source_first_input"):
        if clean_row.get("status") != "passed":
            errors.append("source_first_clean_status_not_passed")
        if clean_row.get("generation_source") != "latex_source":
            errors.append("source_first_generation_not_latex")
        if clean_row.get("page_provenance") != "compiled_vector_color":
            errors.append("source_first_page_provenance_mismatch")
        if clean_row.get("pdf_role") != "independent_verifier_only":
            errors.append("source_first_pdf_role_mismatch")
        verifier = clean_row.get("verifier", {})
        if (
            not isinstance(verifier, dict)
            or verifier.get("status") != "passed"
            or verifier.get("contract_version")
            != SOURCE_FIRST_VERIFIER_CONTRACT_VERSION
            or verifier.get("exact_ordered_character_stream_match") is not True
        ):
            errors.append("source_first_clean_verifier_not_exact")
        source_probe_ids = clean_row.get("source_probe_ids")
        if not isinstance(source_probe_ids, list) or not source_probe_ids:
            errors.append("source_first_clean_source_probe_ids_missing")
        markdown_path = Path(str(clean_row.get("markdown", "")))
        if not (markdown_path.parent.parent / "source_probes.jsonl").is_file():
            errors.append("source_first_clean_source_probes_missing")
    else:
        if clean_row.get("validation_status") != "passed":
            errors.append("clean_validation_status_not_passed")
        if (
            clean_row.get("strict_text_contract_version")
            != STRICT_INPUT_STRICT_TEXT_CONTRACT_VERSION
        ):
            errors.append("clean_strict_text_contract_version_mismatch")
        if clean_row.get("strict_text_v2_status") != "passed":
            errors.append("clean_strict_text_v2_not_passed")
        if clean_row.get("strict_text_v2_failure_reasons") != []:
            errors.append("clean_strict_text_failure_reasons_not_empty")
        if (
            clean_row.get("author_superscript_contract_version")
            != STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION
        ):
            errors.append("clean_author_superscript_contract_version_mismatch")
        clean_contract = clean_row.get("strict_text_contract", {})
        if not isinstance(clean_contract, dict):
            errors.append("clean_strict_text_contract_missing")
        else:
            if clean_contract.get("strict_author_superscript_hard_gate") is not True:
                errors.append("clean_author_superscript_hard_gate_missing")
            if (
                clean_contract.get("author_superscript_contract_version")
                != STRICT_INPUT_AUTHOR_SUPERSCRIPT_CONTRACT_VERSION
            ):
                errors.append("clean_author_superscript_contract_mismatch")
        clean_claims = clean_row.get("strict_text_claims", {})
        if not isinstance(clean_claims, dict) or clean_claims.get("status") != "passed":
            errors.append("clean_line_claims_not_passed")
        elif clean_claims.get("canonical_order_match") is not True:
            errors.append("clean_canonical_order_not_exact")
        for metric_name in ("strict_text_ordered_metrics", "strict_text_claimed_line_metrics"):
            metric = clean_row.get(metric_name, {})
            if not isinstance(metric, dict) or metric.get("status") != "passed":
                errors.append(f"clean_{metric_name}_not_passed")
    clean_md_sha256 = hashlib.sha256(clean_md.encode("utf-8")).hexdigest()
    if clean_row.get("markdown_sha256") != clean_md_sha256:
        errors.append("clean_markdown_sha256_mismatch")
    edited_md = (root / pair["edited_markdown"]).read_text(encoding="utf-8")
    md_differences = char_differences(clean_md, edited_md)
    if len(clean_md) != len(edited_md):
        errors.append("markdown_length_changed")
    if len(md_differences) != len(changes):
        errors.append("markdown_diff_count_mismatch")
    for change in changes:
        start, end = change["markdown_span"]
        if clean_md[start:end] != change["origin_ans"]:
            errors.append(f"clean_markdown_span_mismatch:{change['origin_ans']}")
        if edited_md[start:end] != change["ocr_ans"]:
            errors.append(f"edited_markdown_span_mismatch:{change['ocr_ans']}")

    clean_image_path = resolve_clean_artifact(clean_gt_root, str(clean_row["image"]))
    edited_image_path = root / pair["edited_image"]
    with Image.open(clean_image_path) as clean_image, Image.open(edited_image_path) as edited_image:
        clean_image.load()
        edited_image.load()
        if clean_image.size != edited_image.size:
            errors.append("image_size_changed")
        if ImageChops.difference(clean_image.convert("RGB"), edited_image.convert("RGB")).getbbox() is None:
            errors.append("images_pixel_identical")
        image_width, image_height = edited_image.size

    clean_pdf = resolve_clean_pdf(
        clean_gt_root, str(pair["paper_id"]), str(clean_row["source_pdf"])
    )
    local_edited_pdf = root / "papers" / str(pair["paper_id"]) / "paper_edited.pdf"
    edited_pdf = (
        local_edited_pdf
        if local_edited_pdf.is_file()
        else Path(str(paper_result["edited_pdf"]))
    )
    page_number = int(pair["page_number"])
    with pdfplumber.open(clean_pdf) as clean_document, pdfplumber.open(edited_pdf) as edited_document:
        if len(clean_document.pages) != len(edited_document.pages):
            errors.append("pdf_page_count_changed")
            return errors
        clean_page = clean_document.pages[page_number - 1]
        edited_page = edited_document.pages[page_number - 1]
        clean_words = pdf_words(clean_page)
        edited_words = pdf_words(edited_page)
        if len(clean_words) != len(edited_words):
            errors.append("pdf_word_count_changed")
            return errors
        differing_indices = [
            index
            for index, (left, right) in enumerate(zip(clean_words, edited_words))
            if left["text"] != right["text"]
        ]
        observed = collections.Counter(
            (clean_words[index]["text"], edited_words[index]["text"])
            for index in differing_indices
        )
        declared = collections.Counter(labels)
        if observed != declared:
            errors.append(f"pdf_word_differences_mismatch:{dict(observed)}")
        max_shift = max(
            (abs(left["top"] - right["top"]) for left, right in zip(clean_words, edited_words)),
            default=0.0,
        )
        if max_shift > 1.25:
            errors.append(f"pdf_vertical_reflow:{max_shift:.4f}")
        unused_indices = set(differing_indices)
        for change in changes:
            matching = [
                index
                for index in unused_indices
                if clean_words[index]["text"] == change["origin_ans"]
                and edited_words[index]["text"] == change["ocr_ans"]
            ]
            if len(matching) != 1:
                errors.append(f"pdf_change_not_unique:{change['origin_ans']}")
                continue
            index = matching[0]
            unused_indices.remove(index)
            expected_bbox = expected_pixel_bbox(
                edited_words[index],
                page_width=float(edited_page.width),
                page_height=float(edited_page.height),
                image_width=image_width,
                image_height=image_height,
            )
            if change["bbox"] != expected_bbox:
                errors.append(f"bbox_mismatch:{change['ocr_ans']}")
    return errors


def verify_exports(root: Path, pairs: list[dict[str, Any]], report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    by_pair = {row["pair_id"]: row for row in pairs}
    sft_path = root / report["exports"]["sft"]
    sft_rows = read_jsonl(sft_path)
    if len(sft_rows) != len(pairs):
        errors.append("sft_count_mismatch")
    for row in sft_rows:
        if set(row) != {"images", "conversations"}:
            errors.append("sft_top_level_shape")
        if "changes" in row:
            errors.append("sft_contains_changes")

    verl_rows = read_jsonl(root / "verl_grpo" / "train.jsonl") + read_jsonl(
        root / "verl_grpo" / "val.jsonl"
    )
    if len(verl_rows) != len(pairs):
        errors.append("verl_count_mismatch")
    expected_top = {"data_source", "prompt", "images", "reward_model", "extra_info", "ability"}
    for row in verl_rows:
        if set(row) != expected_top:
            errors.append("verl_top_level_shape")
            continue
        if row["data_source"] != "chaos_document_ocr" or row["ability"] != "document_ocr":
            errors.append("verl_dataset_identity")
        if row["prompt"] != [{"role": "user", "content": VERL_PROMPT}]:
            errors.append("verl_prompt_mismatch")
        pair_id = row["extra_info"].get("pair_id")
        if pair_id not in by_pair:
            errors.append("verl_unknown_pair")
            continue
        pair = by_pair[pair_id]
        edited_md = (root / pair["edited_markdown"]).read_text(encoding="utf-8")
        if row["reward_model"] != {"style": "rule", "ground_truth": edited_md}:
            errors.append("verl_ground_truth_mismatch")
        for change in row["extra_info"].get("changes", []):
            if set(change) != {"ocr_ans", "origin_ans", "bbox"}:
                errors.append("verl_change_shape")
        expected_image = str(PurePosixPath(report["server_root"]) / pair["edited_image"])
        if row["images"] != [expected_image]:
            errors.append("verl_image_path_mismatch")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        errors.append("pyarrow_missing")
    else:
        for split in ("train", "val"):
            parquet_rows = pq.read_table(root / "verl_grpo" / f"{split}.parquet").num_rows
            jsonl_rows = len(read_jsonl(root / "verl_grpo" / f"{split}.jsonl"))
            if parquet_rows != jsonl_rows:
                errors.append(f"parquet_count_mismatch:{split}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--clean-gt-root", type=Path)
    inputs.add_argument("--source-first-case-manifest", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    source_first_mode = args.source_first_case_manifest is not None
    clean_gt_root = (
        args.source_first_case_manifest.resolve().parent
        if source_first_mode
        else args.clean_gt_root.resolve()
    )
    started = time.monotonic()
    pairs = read_jsonl(root / "pairs.jsonl")
    report = read_json(root / "validation_report.json")
    print(
        f"[start] verifier pairs={len(pairs)} root={root} "
        f"clean_reference_root={clean_gt_root}",
        flush=True,
    )
    clean_page_index = (
        load_source_first_clean_page_index(args.source_first_case_manifest)
        if source_first_mode
        else load_clean_page_index(clean_gt_root, pairs)
    )
    print(
        f"[clean_reference_index] pages={len(clean_page_index)}",
        flush=True,
    )
    errors: list[dict[str, str]] = []
    if report.get("output_mode") != "edited_only" or report.get("clean_assets_copied") is not False:
        errors.append({"pair_id": "dataset", "error": "dataset_not_edited_only"})
    if report.get("mutation_policy_version") != MUTATION_POLICY_VERSION:
        errors.append({"pair_id": "dataset", "error": "dataset_mutation_policy_mismatch"})
    if report.get("selection_policy_version") != SELECTION_POLICY_VERSION:
        errors.append({"pair_id": "dataset", "error": "dataset_selection_policy_mismatch"})
    strict_input_audit_relative = report.get("strict_input_filter_audit")
    strict_input_audit_path = (
        root / str(strict_input_audit_relative)
        if strict_input_audit_relative
        else None
    )
    if strict_input_audit_path is None or not strict_input_audit_path.is_file():
        errors.append({"pair_id": "dataset", "error": "strict_input_filter_audit_missing"})
        expected_input_policy = (
            SOURCE_FIRST_INPUT_POLICY_VERSION
            if source_first_mode
            else STRICT_INPUT_FILTER_POLICY_VERSION
        )
    else:
        strict_input_audit = read_json(strict_input_audit_path)
        expected_input_policy = (
            SOURCE_FIRST_INPUT_POLICY_VERSION
            if source_first_mode
            else STRICT_INPUT_FILTER_POLICY_VERSION
        )
        if strict_input_audit.get("policy_version") != expected_input_policy:
            errors.append(
                {"pair_id": "dataset", "error": "strict_input_filter_audit_policy_mismatch"}
            )
        for report_key, audit_key in (
            ("strict_input_pages_scanned", "scanned_pages"),
            ("strict_input_pages_accepted", "accepted_pages"),
            ("strict_input_pages_rejected", "rejected_pages"),
        ):
            if report.get(report_key) != strict_input_audit.get(audit_key):
                errors.append(
                    {"pair_id": "dataset", "error": f"{report_key}_mismatch"}
                )
    if report.get("strict_input_filter_policy_version") != expected_input_policy:
        errors.append(
            {"pair_id": "dataset", "error": "dataset_strict_input_filter_policy_mismatch"}
        )
    if report.get("bibliography_policy_version") != BIBLIOGRAPHY_POLICY_VERSION:
        errors.append({"pair_id": "dataset", "error": "dataset_bibliography_policy_mismatch"})
    paper_results = {
        paper_dir.name: read_json(paper_dir / "paper_result.json")
        for paper_dir in sorted((root / "papers").iterdir())
        if (paper_dir / "paper_result.json").is_file()
    }
    seen_ids: set[str] = set()
    bibliography_start_by_paper: dict[str, int | None] = {}
    for index, pair in enumerate(pairs, start=1):
        pair_id = str(pair["pair_id"])
        if pair_id in seen_ids:
            errors.append({"pair_id": pair_id, "error": "duplicate_pair_id"})
        seen_ids.add(pair_id)
        if any(key.startswith("clean_") for key in pair):
            errors.append({"pair_id": pair_id, "error": "pair_contains_clean_field"})
        clean_row = clean_page_index.get(str(pair.get("data_id")))
        if clean_row is None:
            errors.append({"pair_id": pair_id, "error": "clean_reference_missing"})
            continue
        paper_id = str(pair["paper_id"])
        if paper_id not in bibliography_start_by_paper:
            bibliography_start_by_paper[paper_id] = bibliography_start_page_for_clean_row(
                clean_gt_root, clean_row
            )
        bibliography_start = bibliography_start_by_paper[paper_id]
        if (
            bibliography_start is not None
            and int(pair["page_number"]) >= bibliography_start
        ):
            errors.append(
                {
                    "pair_id": pair_id,
                    "error": (
                        "bibliography_tail_not_excluded:"
                        f"start={bibliography_start}:page={pair['page_number']}"
                    ),
                }
            )
        for error in verify_pair(
            root=root,
            clean_gt_root=clean_gt_root,
            clean_row=clean_row,
            pair=pair,
            paper_result=paper_results[pair["paper_id"]],
            expected_input_policy=expected_input_policy,
        ):
            errors.append({"pair_id": pair_id, "error": error})
        if index == 1 or index % 5 == 0 or index == len(pairs):
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[progress] verified={index}/{len(pairs)} pct={100*index/max(1,len(pairs)):.1f}% "
                f"throughput={index/elapsed:.2f} pairs/s errors={len(errors)} "
                f"current={pair_id}",
                flush=True,
            )
    for error in verify_exports(root, pairs, report):
        errors.append({"pair_id": "dataset", "error": error})
    allowed_empty = {
        f"verl_grpo/{split}.jsonl"
        for split in ("train", "val")
        if int(report.get("exports", {}).get(split, -1)) == 0
    }
    zero_byte = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.stat().st_size == 0
        and str(path.relative_to(root)) not in allowed_empty
        and not any(part in {"build", "source_edited"} for part in path.relative_to(root).parts)
    ]
    if zero_byte:
        errors.append({"pair_id": "dataset", "error": f"zero_byte_files:{zero_byte[:5]}"})
    clean_assets = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith("_clean.png") or path.name.endswith("_clean.md"))
    ]
    if clean_assets:
        errors.append(
            {"pair_id": "dataset", "error": f"clean_assets_present:{clean_assets[:5]}"}
        )
    verification = {
        "status": "passed" if not errors else "failed",
        "pairs": len(pairs),
        "unique_pair_ids": len(seen_ids),
        "mutation_distribution": dict(sorted(collections.Counter(pair["mutation_count"] for pair in pairs).items())),
        "errors": errors,
        "zero_byte_files": len(zero_byte),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(root / "independent_verifier_report.json", verification)
    print(
        f"[final] status={verification['status']} pairs={len(pairs)} errors={len(errors)} "
        f"zero_byte={len(zero_byte)} elapsed={verification['elapsed_seconds']}s",
        flush=True,
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
