# PDF Hallucination Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent batch PDF hallucination evaluator that compares PDF-parser text against OpenAI-compatible chat model transcription and produces metrics plus visual HTML review pages.

**Architecture:** Create a standalone Python package under `pdf_hallu_eval/`. The pipeline discovers PDFs, extracts page text, renders page images, calls an OpenAI-compatible chat endpoint, normalizes and aligns text, computes page/PDF/dataset metrics, and writes static review artifacts.

**Tech Stack:** Python 3.10+, standard-library CLI/config/concurrency, optional `pypdf`/`pdfplumber` for text extraction, Poppler `pdftoppm` for rendering, optional `pymupdf`, no required dependency on the existing `qwen_mm_token_probe` package.

---

### File Structure

- Create `pdf_hallu_eval/pyproject.toml`: standalone package metadata, dependencies, CLI entrypoint.
- Create `pdf_hallu_eval/README.md`: installation, configuration, run examples, output structure.
- Create `pdf_hallu_eval/configs/default.yaml`: example batch/model/report config.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/__init__.py`: package version.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/normalize.py`: text normalization.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/align.py`: character-level Levenshtein alignment.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/metrics.py`: page/PDF/dataset metric aggregation.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/pdf_parser.py`: parser text extraction with `pypdf`, `pdfplumber`, optional `pymupdf`.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/pdf_render.py`: page rendering through optional `pymupdf` or `pdftoppm`.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/chat_client.py`: OpenAI-compatible Chat Completions client using `urllib`.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/storage.py`: deterministic output paths and JSON/JSONL helpers.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/report_html.py`: static index and per-PDF review pages.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/batch.py`: resumable batch orchestration.
- Create `pdf_hallu_eval/src/pdf_hallu_eval/cli.py`: `pdf-hallu-eval run` command.
- Create `pdf_hallu_eval/tests/*.py`: standard-library `unittest` tests for core behavior.

### Task 1: Core Text Comparison

- [ ] Write failing tests for normalization, alignment, and metric definitions in `pdf_hallu_eval/tests/test_core.py`.
- [ ] Run `python3 -m unittest discover -s pdf_hallu_eval/tests -v` and verify failures are missing imports/modules.
- [ ] Implement `normalize.py`, `align.py`, and `metrics.py`.
- [ ] Run the same unittest command and verify the core tests pass.

### Task 2: Storage, Parser, Renderer, and Client Interfaces

- [ ] Write failing tests for JSONL helpers, output path stability, mocked chat payload shape, and renderer/parser dependency errors.
- [ ] Run unittest and verify failures.
- [ ] Implement `storage.py`, `chat_client.py`, `pdf_parser.py`, and `pdf_render.py`.
- [ ] Run unittest and verify tests pass.

### Task 3: HTML Review

- [ ] Write failing tests that build tiny page records and assert `index.html` and per-PDF review files include metrics, image references, parser text, model text, and diff classes.
- [ ] Run unittest and verify failures.
- [ ] Implement `report_html.py`.
- [ ] Run unittest and verify tests pass.

### Task 4: Batch Runner and CLI

- [ ] Write failing tests for dry-run/mock-client batch execution over fake PDF jobs and resume behavior.
- [ ] Run unittest and verify failures.
- [ ] Implement `batch.py` and `cli.py`.
- [ ] Run unittest and verify tests pass.

### Task 5: Documentation and Smoke Verification

- [ ] Write `README.md`, `pyproject.toml`, and `configs/default.yaml`.
- [ ] Run `python3 -m unittest discover -s pdf_hallu_eval/tests -v`.
- [ ] Run `python3 -m compileall pdf_hallu_eval/src pdf_hallu_eval/tests`.
- [ ] Run `python3 pdf_hallu_eval/src/pdf_hallu_eval/cli.py --help` or equivalent module invocation with `PYTHONPATH`.
- [ ] Inspect `git diff --stat` and `git status --short` before final response.

