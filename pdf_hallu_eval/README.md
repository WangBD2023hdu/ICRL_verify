# PDF Hallucination Evaluator

`pdf_hallu_eval` is an independent batch evaluator for PDF transcription hallucination. It compares text extracted by a PDF parser with text produced by an OpenAI-compatible chat model from rendered PDF page images, then writes metrics and a static HTML review site for human inspection.

## What It Measures

For each page, the tool aligns normalized parser text `R` and model output `P` at character level:

```text
C = matched characters
S = substitutions
D = reference characters omitted by the model
I = model characters unsupported by the reference
```

Metrics:

```text
CER = (S + D + I) / len(R)
hallucination_rate = (S + I) / len(P)
pure_insertion_rate = I / len(P)
omission_rate = (S + D) / len(R)
coverage = C / len(R)
```

`hallucination_rate` is broad and counts both wrong characters and extra unsupported output. `pure_insertion_rate` is stricter and only counts extra model output.

## Install

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Rendering uses either optional `pymupdf` or Poppler `pdftoppm`.

For the optional PyMuPDF backend:

```bash
pip install -e '.[pymupdf]'
```

For Poppler on macOS:

```bash
brew install poppler
```

## Run

```bash
pdf-hallu-eval run \
  --pdf-dir /path/to/pdfs \
  --output-dir outputs/pdf_hallu_eval \
  --base-url http://localhost:8000/v1 \
  --model your-vision-model \
  --api-key-env OPENAI_API_KEY \
  --workers 8 \
  --resume
```

Model access uses the official OpenAI Python client with a configurable `base_url`, so it works with OpenAI and OpenAI-compatible local or third-party services. The chat endpoint must follow the OpenAI Chat Completions shape:

```text
POST /v1/chat/completions
messages[0].content = [
  {"type": "text", ...},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]
```

### INF Bearer Authentication

Set the API key in the shell. Do not put the key itself in the command, config file, or output directory:

```bash
export INF_API_KEY='your-api-key'
```

The equivalent authentication headers can be checked with curl:

```bash
curl -k https://eopjcpemcecbc8p8hgcec9ocgbpqqhep.openapi-hw.infly.cn \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $INF_API_KEY"
```

`-k` is a curl-only option that disables TLS certificate verification. The evaluator keeps normal TLS verification enabled.

Pass the environment variable name to the evaluator with `--api-key-env`:

```bash
pdf-hallu-eval run \
  --pdf-dir /path/to/pdfs \
  --output-dir outputs/inf_eval \
  --base-url https://eopjcpemcecbc8p8hgcec9ocgbpqqhep.openapi-hw.infly.cn \
  --model your-vision-model \
  --api-key-env INF_API_KEY \
  --workers 8 \
  --resume
```

The OpenAI client automatically adds `Content-Type: application/json` and `Authorization: Bearer <INF_API_KEY>`. It treats `--base-url` as an API root and sends Chat Completions requests below that root at `chat/completions`. If the INF URL above is already the complete inference endpoint rather than an OpenAI-compatible API root, use the provider's OpenAI-compatible base URL instead.

Use `--dry-run` to exercise parser/render/report generation without model calls.

## Important Options

```text
--parser auto|pymupdf|pdfplumber|pypdf
--render-backend auto|pymupdf|pdftoppm
--workers 8
--resume / --no-resume
--force
--limit 100
--preserve-newlines
--dpi 144
```

By default, newlines are flattened before alignment because parser and model line wrapping often differ. Raw parser/model text is still preserved in review pages.

## Outputs

```text
outputs/pdf_hallu_eval/
  pages.jsonl
  pdf_summary.csv
  dataset_summary.json
  images/
  parser_text/
  model_text/
  alignments/
  review/
    index.html
    <pdf-id>.html
```

Open `review/index.html` to review the worst PDFs first. Each PDF page shows:

```text
left: rendered PDF page image
middle: PDF parser text
right: model output text
bottom: character-level diff
```

Diff colors:

```text
red: model insertion, unsupported by parser reference
orange: substitution
gray: parser reference omitted by model
plain: matched model output
```

## Notes

- Born-digital PDFs are the best fit for the first version because parser text can serve as a strong pseudo-reference.
- Scanned PDFs with no text layer may be marked `parser_empty`; those pages are useful for triage but not reliable hallucination-rate ground truth.
- `--resume` skips pages that already have an alignment JSON file, so interrupted large runs can continue without repeating model calls.
