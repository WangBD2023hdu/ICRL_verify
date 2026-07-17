from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TokenScore:
    index: int
    token_id: int
    token: str
    raw_token: str
    p_original: float
    p_masked: float
    logp_original: float
    logp_masked: float
    top_token_id_original: int
    top_token_original: str
    top_raw_token_original: str
    top_p_original: float
    top_logp_original: float
    top_token_id_masked: int
    top_token_masked: str
    top_raw_token_masked: str
    top_p_masked: float
    top_logp_masked: float

    @property
    def delta_p(self) -> float:
        return self.p_original - self.p_masked

    @property
    def delta_logp(self) -> float:
        return self.logp_original - self.logp_masked

    @property
    def top_token_changed(self) -> bool:
        return self.top_token_id_original != self.top_token_id_masked

    @property
    def target_is_top_original(self) -> bool:
        return self.token_id == self.top_token_id_original

    @property
    def target_is_top_masked(self) -> bool:
        return self.token_id == self.top_token_id_masked

    @property
    def compact_token(self) -> str:
        token = self.token.replace(" ", "·")
        if len(token) > 12:
            return token[:11] + "…"
        return token

    def to_dict(self) -> dict[str, bool | float | int | str]:
        data = asdict(self)
        data["delta_p"] = self.delta_p
        data["delta_logp"] = self.delta_logp
        data["top_token_changed"] = self.top_token_changed
        data["target_is_top_original"] = self.target_is_top_original
        data["target_is_top_masked"] = self.target_is_top_masked
        return data
