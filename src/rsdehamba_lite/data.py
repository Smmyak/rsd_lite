from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HAZY_ALIASES = {"hazy", "haze", "fog", "input", "inputs", "source", "src"}
CLEAR_ALIASES = {"clear", "clean", "gt", "groundtruth", "ground_truth", "target", "targets", "label", "labels"}


@dataclass(frozen=True)
class PairRecord:
    hazy: Path
    clear: Path
    split: str
    haze_level: str


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _role(path: Path) -> str | None:
    parts = {p.lower().replace("-", "_") for p in path.parts}
    if parts & HAZY_ALIASES:
        return "hazy"
    if parts & CLEAR_ALIASES:
        return "clear"
    return None


def _haze_level(path: Path) -> str:
    lower = [p.lower() for p in path.parts]
    for name in ("thin", "moderate", "thick"):
        if any(name in part for part in lower):
            return name
    return "unknown"


def _split(path: Path) -> str:
    lower = [p.lower() for p in path.parts]
    for name in ("train", "val", "valid", "validation", "test"):
        if any(name == part or name in part for part in lower):
            return "val" if name in {"val", "valid", "validation"} else name
    return "unsplit"


def discover_pairs(data_root: str | Path) -> list[PairRecord]:
    root = Path(data_root)
    hazy: dict[str, Path] = {}
    clear: dict[str, Path] = {}
    for image in _images(root):
        role = _role(image)
        rel = image.relative_to(root)
        key = image.stem.lower().replace("_hazy", "").replace("_clear", "").replace("_gt", "")
        key = f"{_haze_level(rel)}::{_split(rel)}::{key}"
        if role == "hazy":
            hazy[key] = image
        elif role == "clear":
            clear[key] = image

    pairs = []
    for key in sorted(hazy.keys() & clear.keys()):
        level, split, _ = key.split("::", 2)
        pairs.append(PairRecord(hazy=hazy[key], clear=clear[key], split=split, haze_level=level))
    return pairs


def load_rgb(path: Path, image_size: int | None = None, random_crop: bool = False) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if image_size:
            w, h = img.size
            if random_crop and w >= image_size and h >= image_size:
                left = random.randint(0, w - image_size)
                top = random.randint(0, h - image_size)
                img = img.crop((left, top, left + image_size, top + image_size))
            else:
                img = img.resize((image_size, image_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PairedDehazeDataset(Dataset):
    def __init__(
        self,
        pairs: Iterable[PairRecord],
        image_size: int | None = 256,
        random_crop: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        return {
            "hazy": load_rgb(pair.hazy, self.image_size, self.random_crop),
            "clear": load_rgb(pair.clear, self.image_size, self.random_crop),
            "hazy_path": str(pair.hazy),
            "clear_path": str(pair.clear),
            "haze_level": pair.haze_level,
        }

