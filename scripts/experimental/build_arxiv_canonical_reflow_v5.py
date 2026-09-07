#!/usr/bin/env python3
"""Multimodal arXiv synthesis with the user's weighted 1012-letter policy.

Reuses V4 parsing, parallel edited compilation, image sharding, and streaming
SFT/VERL export. Uses the confirmed DOC2MD prompt with existing image paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from arxiv_canonical_reflow_v4.prompts import DOC2MD_PROMPT_STYLE
from arxiv_canonical_reflow_v4.weighted_mutation import POLICY_NAME
from scripts.experimental import build_arxiv_canonical_reflow_v4 as pipeline


def main(argv: list[str] | None = None) -> int:
    return pipeline.main(
        argv,
        default_mutation_policy=POLICY_NAME,
        default_prompt_style=DOC2MD_PROMPT_STYLE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
