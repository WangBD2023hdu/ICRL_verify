from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class TokenScoreLike(Protocol):
    index: int
    token_id: int
    token: str
    raw_token: str
    p_original: float
    p_masked: float
    logp_original: float
    logp_masked: float


@dataclass(frozen=True)
class WordScore:
    index: int
    unit_type: str
    text: str
    raw_text: str
    token_start: int
    token_end: int
    token_ids: list[int]
    tokens: list[str]
    sum_logp_original: float
    sum_logp_masked: float
    mean_logp_original: float
    mean_logp_masked: float
    first_token_p_original: float
    first_token_p_masked: float
    first_token_logp_original: float
    first_token_logp_masked: float

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def p_original(self) -> float:
        return _safe_exp(self.sum_logp_original)

    @property
    def p_masked(self) -> float:
        return _safe_exp(self.sum_logp_masked)

    @property
    def delta_p(self) -> float:
        return self.p_original - self.p_masked

    @property
    def delta_sum_logp(self) -> float:
        return self.sum_logp_original - self.sum_logp_masked

    @property
    def delta_mean_logp(self) -> float:
        return self.mean_logp_original - self.mean_logp_masked

    @property
    def first_token_delta_p(self) -> float:
        return self.first_token_p_original - self.first_token_p_masked

    @property
    def first_token_delta_logp(self) -> float:
        return self.first_token_logp_original - self.first_token_logp_masked

    @property
    def image_dependency_logp(self) -> float:
        return self.first_token_delta_logp

    @property
    def compact_text(self) -> str:
        text = self.text.replace(" ", "·").replace("\n", "\\n")
        if len(text) > 18:
            return text[:17] + "..."
        return text

    def to_dict(self) -> dict[str, float | int | str | list[int] | list[str]]:
        return {
            "index": self.index,
            "unit_type": self.unit_type,
            "text": self.text,
            "raw_text": self.raw_text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "token_ids": self.token_ids,
            "tokens": self.tokens,
            "p_original": self.p_original,
            "p_masked": self.p_masked,
            "delta_p": self.delta_p,
            "sum_logp_original": self.sum_logp_original,
            "sum_logp_masked": self.sum_logp_masked,
            "delta_sum_logp": self.delta_sum_logp,
            "mean_logp_original": self.mean_logp_original,
            "mean_logp_masked": self.mean_logp_masked,
            "delta_mean_logp": self.delta_mean_logp,
            "first_token_p_original": self.first_token_p_original,
            "first_token_p_masked": self.first_token_p_masked,
            "first_token_delta_p": self.first_token_delta_p,
            "first_token_logp_original": self.first_token_logp_original,
            "first_token_logp_masked": self.first_token_logp_masked,
            "first_token_delta_logp": self.first_token_delta_logp,
            "image_dependency_logp": self.image_dependency_logp,
        }


def group_token_scores(scores: list[TokenScoreLike]) -> list[WordScore]:
    groups: list[list[TokenScoreLike]] = []
    current: list[TokenScoreLike] = []
    current_type: str | None = None

    for score in scores:
        unit_type = _piece_type(score.raw_token)
        if _starts_new_unit(current, current_type, score.raw_token, unit_type):
            groups.append(current)
            current = [score]
            current_type = unit_type
        else:
            current.append(score)
            current_type = _merge_unit_type(current_type, unit_type)

    if current:
        groups.append(current)

    return [_build_word_score(index, group) for index, group in enumerate(groups)]


def _build_word_score(index: int, group: list[TokenScoreLike]) -> WordScore:
    raw_text = "".join(score.raw_token for score in group)
    unit_type = _group_type(group)
    text = raw_text if unit_type == "space" else raw_text.strip()
    if text == "":
        text = _display_space(raw_text)

    sum_original = sum(float(score.logp_original) for score in group)
    sum_masked = sum(float(score.logp_masked) for score in group)
    count = len(group)
    return WordScore(
        index=index,
        unit_type=unit_type,
        text=text,
        raw_text=raw_text,
        token_start=int(group[0].index),
        token_end=int(group[-1].index) + 1,
        token_ids=[int(score.token_id) for score in group],
        tokens=[score.token for score in group],
        sum_logp_original=float(sum_original),
        sum_logp_masked=float(sum_masked),
        mean_logp_original=float(sum_original / count),
        mean_logp_masked=float(sum_masked / count),
        first_token_p_original=float(group[0].p_original),
        first_token_p_masked=float(group[0].p_masked),
        first_token_logp_original=float(group[0].logp_original),
        first_token_logp_masked=float(group[0].logp_masked),
    )


def _starts_new_unit(
    current: list[TokenScoreLike],
    current_type: str | None,
    piece: str,
    piece_type: str,
) -> bool:
    if not current:
        return False
    if current_type == "space" or piece_type == "space":
        return True
    if piece.startswith((" ", "\n", "\t")):
        return True
    if current_type == piece_type == "word":
        return False
    if current_type == piece_type == "punct":
        return False
    return True


def _merge_unit_type(current_type: str | None, piece_type: str) -> str:
    if current_type is None:
        return piece_type
    if current_type == piece_type:
        return current_type
    if "word" in {current_type, piece_type}:
        return "word"
    if "other" in {current_type, piece_type}:
        return "other"
    return piece_type


def _group_type(group: list[TokenScoreLike]) -> str:
    types = [_piece_type(score.raw_token) for score in group]
    if any(unit_type == "word" for unit_type in types):
        return "word"
    if all(unit_type == "space" for unit_type in types):
        return "space"
    if all(unit_type == "punct" for unit_type in types):
        return "punct"
    return "other"


def _piece_type(piece: str) -> str:
    stripped = piece.strip()
    if stripped == "":
        return "space"
    if any(char.isalnum() or char == "_" for char in stripped):
        return "word"
    if all(not (char.isalnum() or char == "_") for char in stripped):
        return "punct"
    return "other"


def _display_space(text: str) -> str:
    return text.replace(" ", "·").replace("\n", "\\n").replace("\t", "\\t")


def _safe_exp(value: float) -> float:
    if value < -745.0:
        return 0.0
    if value > 700.0:
        return math.inf
    return math.exp(value)
