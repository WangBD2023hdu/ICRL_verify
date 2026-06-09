from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageFilter


MaskStrategy = Literal["patch", "word"]
MaskEffect = Literal["replace", "fade", "blur", "noise", "blur_fade"]
MaskFill = Literal["mean", "black", "white", "noise"]


@dataclass(frozen=True)
class Region:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str

    def clipped(self, width: int, height: int, padding: int = 0) -> "Region | None":
        x1 = max(0, min(width, self.x1 - padding))
        y1 = max(0, min(height, self.y1 - padding))
        x2 = max(0, min(width, self.x2 + padding))
        y2 = max(0, min(height, self.y2 + padding))
        if x2 <= x1 or y2 <= y1:
            return None
        return Region(x1=x1, y1=y1, x2=x2, y2=y2, source=self.source)

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class MaskConfig:
    strategy: MaskStrategy = "patch"
    ratio: float = 0.35
    patch_size: int = 32
    fill: MaskFill = "mean"
    effect: MaskEffect = "replace"
    opacity: float = 1.0
    blur_radius: float = 1.2
    noise_std: float = 10.0
    seed: int = 7
    word_boxes_path: str | None = None
    word_padding: int = 2
    word_gap: int = 12
    word_min_width: int = 4
    word_min_height: int = 4
    text_threshold: int | None = None

    def validate(self) -> None:
        if self.strategy not in {"patch", "word"}:
            raise ValueError(f"unsupported mask strategy: {self.strategy}")
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"mask ratio must be in [0, 1], got {self.ratio}")
        if self.patch_size <= 0:
            raise ValueError(f"patch size must be positive, got {self.patch_size}")
        if self.fill not in {"mean", "black", "white", "noise"}:
            raise ValueError(f"unsupported mask fill: {self.fill}")
        if self.effect not in {"replace", "fade", "blur", "noise", "blur_fade"}:
            raise ValueError(f"unsupported mask effect: {self.effect}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"mask opacity must be in [0, 1], got {self.opacity}")
        if self.blur_radius < 0:
            raise ValueError(f"blur radius must be non-negative, got {self.blur_radius}")
        if self.noise_std < 0:
            raise ValueError(f"noise std must be non-negative, got {self.noise_std}")
        if self.word_padding < 0:
            raise ValueError(f"word padding must be non-negative, got {self.word_padding}")
        if self.word_gap < 0:
            raise ValueError(f"word gap must be non-negative, got {self.word_gap}")
        if self.word_min_width <= 0 or self.word_min_height <= 0:
            raise ValueError("word min width and height must be positive")
        if self.text_threshold is not None and not 0 <= self.text_threshold <= 255:
            raise ValueError(f"text threshold must be in [0, 255], got {self.text_threshold}")

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class MaskMetadata:
    image_width: int
    image_height: int
    strategy: str
    effect: str
    num_candidates: int
    num_selected_regions: int
    selected_regions: list[Region]
    config: MaskConfig
    grid_width: int | None = None
    grid_height: int | None = None
    word_box_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "strategy": self.strategy,
            "effect": self.effect,
            "num_candidates": self.num_candidates,
            "num_selected_regions": self.num_selected_regions,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "word_box_source": self.word_box_source,
            "selected_regions": [region.to_dict() for region in self.selected_regions],
            "config": self.config.to_dict(),
        }


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


def apply_image_mask(
    image: Image.Image,
    config: MaskConfig,
) -> tuple[Image.Image, MaskMetadata]:
    config.validate()
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    rng = np.random.default_rng(config.seed)

    grid_width = None
    grid_height = None
    word_box_source = None
    if config.strategy == "patch":
        candidates, grid_width, grid_height = _patch_regions(width, height, config.patch_size)
    else:
        candidates, word_box_source = _word_regions(rgb_image, config)

    selected = _select_regions(candidates, config.ratio, rng)
    masked = _apply_region_effects(rgb_image, selected, config, rng)
    metadata = MaskMetadata(
        image_width=width,
        image_height=height,
        strategy=config.strategy,
        effect=config.effect,
        num_candidates=len(candidates),
        num_selected_regions=len(selected),
        selected_regions=selected,
        config=config,
        grid_width=grid_width,
        grid_height=grid_height,
        word_box_source=word_box_source,
    )
    return masked, metadata


def apply_random_patch_mask(
    image: Image.Image,
    config: MaskConfig,
) -> tuple[Image.Image, MaskMetadata]:
    patch_config = MaskConfig(**{**config.to_dict(), "strategy": "patch"})
    return apply_image_mask(image, patch_config)


def _patch_regions(width: int, height: int, patch_size: int) -> tuple[list[Region], int, int]:
    grid_height = int(np.ceil(height / patch_size))
    grid_width = int(np.ceil(width / patch_size))
    regions = []
    for row in range(grid_height):
        for col in range(grid_width):
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = min(x1 + patch_size, width)
            y2 = min(y1 + patch_size, height)
            regions.append(Region(x1=x1, y1=y1, x2=x2, y2=y2, source="patch"))
    return regions, grid_width, grid_height


def _word_regions(image: Image.Image, config: MaskConfig) -> tuple[list[Region], str]:
    width, height = image.size
    if config.word_boxes_path:
        regions = load_word_boxes(config.word_boxes_path, width=width, height=height)
        source = str(Path(config.word_boxes_path).expanduser())
    else:
        regions = detect_word_like_boxes(image, config)
        source = "auto_text_threshold"

    clipped = []
    for region in regions:
        box = region.clipped(width, height, padding=config.word_padding)
        if box is None:
            continue
        if box.x2 - box.x1 < config.word_min_width:
            continue
        if box.y2 - box.y1 < config.word_min_height:
            continue
        clipped.append(box)
    return clipped, source


def _select_regions(
    candidates: list[Region],
    ratio: float,
    rng: np.random.Generator,
) -> list[Region]:
    if not candidates or ratio <= 0.0:
        return []
    num_selected = min(len(candidates), max(1, int(round(len(candidates) * ratio))))
    selected_indices = rng.choice(len(candidates), size=num_selected, replace=False)
    selected = [candidates[int(i)] for i in selected_indices]
    return sorted(selected, key=lambda region: (region.y1, region.x1, region.y2, region.x2))


def _apply_region_effects(
    image: Image.Image,
    regions: list[Region],
    config: MaskConfig,
    rng: np.random.Generator,
) -> Image.Image:
    if not regions or config.opacity <= 0:
        return image.copy()

    original = np.array(image, dtype=np.float32)
    pixels = original.copy()
    mean_color = original.reshape(-1, 3).mean(axis=0)
    blurred = None
    if config.effect in {"blur", "blur_fade"} and config.blur_radius > 0:
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))
        blurred = np.array(blurred_image, dtype=np.float32)

    for region in regions:
        current = pixels[region.y1 : region.y2, region.x1 : region.x2]
        degraded = _degraded_patch(
            current=current,
            blurred=None if blurred is None else blurred[region.y1 : region.y2, region.x1 : region.x2],
            config=config,
            rng=rng,
            mean_color=mean_color,
        )
        pixels[region.y1 : region.y2, region.x1 : region.x2] = (
            current * (1.0 - config.opacity) + degraded * config.opacity
        )

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def _degraded_patch(
    *,
    current: np.ndarray,
    blurred: np.ndarray | None,
    config: MaskConfig,
    rng: np.random.Generator,
    mean_color: np.ndarray,
) -> np.ndarray:
    if config.effect in {"replace", "fade"}:
        return _fill_patch(current.shape, config.fill, rng, mean_color)
    if config.effect == "blur":
        return current if blurred is None else blurred
    if config.effect == "noise":
        noise = rng.normal(loc=0.0, scale=config.noise_std, size=current.shape)
        return np.clip(current + noise, 0, 255)

    fill = _fill_patch(current.shape, config.fill, rng, mean_color)
    blur_patch = current if blurred is None else blurred
    return blur_patch * 0.65 + fill * 0.35


def _fill_patch(
    shape: tuple[int, ...],
    fill: MaskFill,
    rng: np.random.Generator,
    mean_color: np.ndarray,
) -> np.ndarray:
    if fill == "mean":
        return np.broadcast_to(mean_color, shape).astype(np.float32)
    if fill == "black":
        return np.zeros(shape, dtype=np.float32)
    if fill == "white":
        return np.full(shape, 255.0, dtype=np.float32)
    return rng.integers(low=0, high=256, size=shape).astype(np.float32)


def load_word_boxes(path: str | Path, *, width: int, height: int) -> list[Region]:
    box_path = Path(path).expanduser()
    data = json.loads(box_path.read_text(encoding="utf-8"))
    raw_boxes = list(_iter_boxes(data))
    regions = []
    for box in raw_boxes:
        region = _box_to_region(box, width=width, height=height, source="word_box")
        if region is not None:
            regions.append(region)
    return regions


def detect_word_like_boxes(image: Image.Image, config: MaskConfig) -> list[Region]:
    gray = np.array(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    threshold = config.text_threshold
    if threshold is None:
        threshold = min(220, max(80, int(np.percentile(gray, 20) + 25)))
    dark = gray <= threshold

    row_min_pixels = max(2, int(width * 0.0015))
    row_has_text = dark.sum(axis=1) >= row_min_pixels
    line_groups = _groups_from_bool(row_has_text, max_gap=2)

    regions = []
    for y1, y2 in line_groups:
        if y2 - y1 < config.word_min_height:
            continue
        line_dark = dark[y1:y2]
        col_has_text = line_dark.sum(axis=0) >= 1
        for x1, x2 in _groups_from_bool(col_has_text, max_gap=config.word_gap):
            if x2 - x1 < config.word_min_width:
                continue
            regions.append(Region(x1=x1, y1=y1, x2=x2, y2=y2, source="auto_word"))
    return regions


def _groups_from_bool(values: np.ndarray, *, max_gap: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = None
    last_true = None
    for idx, value in enumerate(values.tolist()):
        if value:
            if start is None:
                start = idx
            last_true = idx
            continue
        if start is not None and last_true is not None and idx - last_true > max_gap:
            groups.append((start, last_true + 1))
            start = None
            last_true = None
    if start is not None and last_true is not None:
        groups.append((start, last_true + 1))
    return groups


def _iter_boxes(data: Any) -> list[list[float]]:
    boxes: list[list[float]] = []
    if isinstance(data, dict):
        if _looks_like_box(data.get("bbox")):
            boxes.append([float(v) for v in data["bbox"]])
        if _looks_like_box(data.get("box")):
            boxes.append([float(v) for v in data["box"]])
        if all(key in data for key in ("x", "y", "w", "h")):
            x = float(data["x"])
            y = float(data["y"])
            boxes.append([x, y, x + float(data["w"]), y + float(data["h"])])
        for key, value in data.items():
            if key in {"bbox", "box"}:
                continue
            if key in {"x", "y", "w", "h"}:
                continue
            boxes.extend(_iter_boxes(value))
    elif isinstance(data, list):
        if _looks_like_box(data):
            boxes.append([float(v) for v in data])
        else:
            for item in data:
                boxes.extend(_iter_boxes(item))
    return boxes


def _looks_like_box(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    )


def _box_to_region(
    box: list[float],
    *,
    width: int,
    height: int,
    source: str,
) -> Region | None:
    if max(box) <= 1.5:
        x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    else:
        x1, y1, x2, y2 = box
    region = Region(
        x1=int(round(min(x1, x2))),
        y1=int(round(min(y1, y2))),
        x2=int(round(max(x1, x2))),
        y2=int(round(max(y1, y2))),
        source=source,
    )
    return region.clipped(width, height)
