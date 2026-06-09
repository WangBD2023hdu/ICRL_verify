from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


MaskFill = Literal["mean", "black", "white", "noise"]


@dataclass(frozen=True)
class MaskConfig:
    ratio: float = 0.35
    patch_size: int = 32
    fill: MaskFill = "mean"
    seed: int = 7

    def validate(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"mask ratio must be in [0, 1], got {self.ratio}")
        if self.patch_size <= 0:
            raise ValueError(f"patch size must be positive, got {self.patch_size}")
        if self.fill not in {"mean", "black", "white", "noise"}:
            raise ValueError(f"unsupported mask fill: {self.fill}")

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class MaskMetadata:
    image_width: int
    image_height: int
    grid_width: int
    grid_height: int
    num_patches: int
    num_masked_patches: int
    config: MaskConfig

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["config"] = self.config.to_dict()
        return data


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def save_rgb_image(image: Image.Image, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)
    return out_path


def apply_random_patch_mask(
    image: Image.Image,
    config: MaskConfig,
) -> tuple[Image.Image, MaskMetadata]:
    """Mask random square patches in an RGB image.

    The mask is sampled on a non-overlapping patch grid so the effective mask
    ratio is stable and easy to reproduce with a seed.
    """

    config.validate()
    rgb_image = image.convert("RGB")
    pixels = np.array(rgb_image, dtype=np.uint8).copy()
    height, width = pixels.shape[:2]

    grid_height = int(np.ceil(height / config.patch_size))
    grid_width = int(np.ceil(width / config.patch_size))
    num_patches = grid_height * grid_width
    num_masked = int(round(num_patches * config.ratio))

    rng = np.random.default_rng(config.seed)
    selected = set(rng.choice(num_patches, size=num_masked, replace=False).tolist())
    mean_color = pixels.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)

    for patch_index in selected:
        row = patch_index // grid_width
        col = patch_index % grid_width
        y0 = row * config.patch_size
        y1 = min(y0 + config.patch_size, height)
        x0 = col * config.patch_size
        x1 = min(x0 + config.patch_size, width)

        if config.fill == "mean":
            fill_value = mean_color
        elif config.fill == "black":
            fill_value = np.array([0, 0, 0], dtype=np.uint8)
        elif config.fill == "white":
            fill_value = np.array([255, 255, 255], dtype=np.uint8)
        else:
            fill_value = rng.integers(
                low=0,
                high=256,
                size=(y1 - y0, x1 - x0, 3),
                dtype=np.uint8,
            )

        pixels[y0:y1, x0:x1] = fill_value

    masked = Image.fromarray(pixels, mode="RGB")
    metadata = MaskMetadata(
        image_width=width,
        image_height=height,
        grid_width=grid_width,
        grid_height=grid_height,
        num_patches=num_patches,
        num_masked_patches=num_masked,
        config=config,
    )
    return masked, metadata
