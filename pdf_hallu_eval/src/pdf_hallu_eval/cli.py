from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from pdf_hallu_eval.batch import BatchConfig, run_batch
from pdf_hallu_eval.chat_client import ChatConfig, OpenAIChatClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-hallu-eval",
        description="Batch-evaluate PDF transcription hallucination with visual review reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run batch PDF hallucination evaluation.")
    run.add_argument("--pdf-dir", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--base-url", default="http://localhost:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the Bearer API key (for example, INF_API_KEY).",
    )
    run.add_argument("--prompt", default=None)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--timeout-s", type=float, default=120.0)
    run.add_argument("--retries", type=int, default=3)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--parser", choices=["auto", "pymupdf", "pdfplumber", "pypdf"], default="auto")
    run.add_argument("--render-backend", choices=["auto", "pymupdf", "pdftoppm"], default="auto")
    run.add_argument("--dpi", type=int, default=144)
    run.add_argument("--pdftoppm-path", default=None)
    run.add_argument("--preserve-newlines", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Skip model calls and generate empty predictions.")
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--force", action="store_true", help="Ignore cached page alignments.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _run(args: argparse.Namespace) -> int:
    batch_config = BatchConfig(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        parser=args.parser,
        render_backend=args.render_backend,
        dpi=args.dpi,
        workers=args.workers,
        resume=args.resume,
        force=args.force,
        limit=args.limit,
        preserve_newlines=args.preserve_newlines,
        dry_run=args.dry_run,
        pdftoppm_path=args.pdftoppm_path,
    )

    chat_client = None
    if not args.dry_run:
        chat_config = ChatConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env, ""),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            retries=args.retries,
            prompt=args.prompt or ChatConfig(model=args.model, base_url=args.base_url).prompt,
        )
        chat_client = OpenAIChatClient(chat_config)

    result = run_batch(batch_config, chat_client=chat_client)
    print(f"Processed {result.total_pages} pages from {result.total_pdfs} PDFs.")
    print(f"Review index: {result.output_dir / 'review' / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
