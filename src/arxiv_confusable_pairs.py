"""Shared empirical letter-pair weights for multimodal V5 and text rewrite."""

from __future__ import annotations

import hashlib
import json

# Integer counts, not rounded percentages. The requested 0 -> e is excluded.
PAIR_COUNTS: dict[str, dict[str, int]] = {
    "e": {"c": 214, "o": 55, "a": 50, "f": 1},
    "o": {"c": 128, "e": 52, "a": 35},
    "c": {"e": 53, "o": 29, "a": 9},
    "a": {"e": 31, "c": 28, "o": 20, "n": 1},
    "n": {"m": 56, "h": 7, "r": 7, "v": 1},
    "m": {"n": 63},
    "u": {"v": 38, "a": 3, "o": 3, "y": 1},
    "v": {"u": 18, "w": 17},
    "r": {"n": 11, "t": 10, "i": 3, "c": 1},
    "w": {"v": 19, "u": 1},
    "i": {"l": 17, "t": 1},
    "t": {"i": 4, "r": 4},
    "l": {"i": 7},
    "p": {"q": 4},
    "b": {"d": 1, "q": 1, "r": 1},
    "d": {"b": 2, "o": 1},
    "h": {"n": 2},
    "f": {"l": 1},
    "s": {"c": 1},
}
PAIR_WEIGHTS = {
    (source, target): count
    for source, targets in PAIR_COUNTS.items()
    for target, count in targets.items()
}
TOTAL_WEIGHT = sum(PAIR_WEIGHTS.values())
POLICY_FINGERPRINT = hashlib.sha256(
    json.dumps(PAIR_COUNTS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:12]
