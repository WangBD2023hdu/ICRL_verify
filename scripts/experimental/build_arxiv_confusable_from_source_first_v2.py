#!/usr/bin/env python3
"""Build v1-format edited SFT/VERL data from strict source-first v2 pages."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arxiv_source_first_v2.mutation_adapter import main  # noqa: I001


if __name__ == "__main__":
    raise SystemExit(main())
