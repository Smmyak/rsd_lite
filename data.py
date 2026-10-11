from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

HAZY_ALIASES = {
    "hazy", "haze", "fog", "input", "inputs", "source", "src"
}

CLEAR_ALIASES = {
    "clear", "clean", "gt", "groundtruth", "ground_truth",
    "target", "targets", "label", "labels"
}


@dataclass(frozen=True)
class PairRecord:
    hazy: Path
    clear: Path
    split: str
    haze_level: str


def _images(root: Path) -> list[Path]:
    """Find supported image files recursively."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _normalise_name(name: str) -> str:
    """Normalize a folder or filename component."""
    name = name.lower().strip()
    return re.sub(r"[\s-]+", "_", name)


def _role(path: Path) -> str | None:
    """
    Identify whether an image represents a hazy input or clear target.
    Checks parent folders and filename markers.
    """
    parts = [_normalise_name(part) for part in path.parts]

    for folder in reversed(parts[:-1]):
        if folder in HAZY_ALIASES:
            return "hazy"
        if folder in CLEAR_ALIASES:
            return "clear"

    stem = _normalise_name(path.stem)
    patterns = [
        (r"(^|_)(hazy|haze|fog|input|source|src)(_|$)", "hazy"),
        (r"(^|_)(clear|clean|gt|groundtruth|ground_truth|target|label)(_|$)", "clear"),
    ]

    for pattern, role in patterns:
        if re.search(pattern, stem):
            return role

    return None


def _pair_stem(path: Path) -> str:
    """Normalize suffixes that identify hazy inputs and clear targets."""
    stem = _normalise_name(path.stem)
    marker = (
        r"(^|_)(hazy|haze|fog|input|inputs|source|src|"
        r"clear|clean|gt|groundtruth|ground_truth|"
        r"target|targets|label|labels)(_|$)"
    )

    previous = None
    while previous != stem:
        previous = stem
        stem = re.sub(marker, "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("_")

    return stem


def _haze_level(path: Path) -> str:
    """Extract thin, moderate, or thick from the path."""
    parts = [_normalise_name(part) for part in path.parts]

    for level in ("thin", "moderate", "thick"):
        if any(
            part == level
            or part.startswith(f"haze1k_{level}")
            or part.startswith(f"{level}_")
            for part in parts
        ):
            return level

    return "unknown"


def _split(path: Path) -> str:
    """Determine the dataset split from directory names."""
    parts = [_normalise_name(part) for part in path.parts[:-1]]

    for part in reversed(parts):
        if part in {"train", "training"}:
            return "train"
        if part in {"val", "valid", "validation"}:
            return "val"
        if part in {"test", "testing"}:
            return "test"

    for part in reversed(parts):
        if re.match(r"^(train|training)(_|[0-9a-z])", part):
            return "train"
        if re.match(r"^(val|valid|validation)(_|[0-9a-z])", part):
            return "val"
        if re.match(r"^(test|testing)(_|[0-9a-z])", part):
            return "test"

    return "unsplit"


def discover_pairs(data_root: str | Path) -> list[PairRecord]:
    """
    Discover paired hazy and clear images.

    Pair keys include haze level and split so images from different
    subsets or haze levels are not accidentally matched.
    """
    root = Path(data_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    hazy: dict[tuple[str, str, str], Path] = {}
    clear: dict[tuple[str, str, str], Path] = {}

    for image in _images(root):
        role = _role(image)
        if role is None:
            continue

        relative_path = image.relative_to(root)
        key = (_haze_level(relative_path), _split(relative_path), _pair_stem(image))

        if not key[2]:
            continue

        destination = hazy if role == "hazy" else clear
        if key in destination:
            raise ValueError(
                f"Duplicate {role} image for pair key {key}:\n"
                f"  {destination[key]}\n"
                f"  {image}"
            )
        destination[key] = image

    pairs: list[PairRecord] = []
    for key in sorted(hazy.keys() & clear.keys()):
        level, split, _ = key
        pairs.append(
            PairRecord(
                hazy=hazy[key],
                clear=clear[key],
                split=split,
                haze_level=level,
            )
        )

    if not pairs:
        raise RuntimeError(
            f"No image pairs found under {root}. Inspect the dataset "
            "folders and filenames to verify hazy/clear naming conventions."
        )

    return pairs


def load_rgb(
    path: Path,
    image_size: int | None = None,
    random_crop: bool = False,
) -> torch.Tensor:
    """
    Load an RGB image as a float tensor in [0, 1].

    For paired training crops, use PairedDehazeDataset so that the
    same crop coordinates are applied to both images.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")

        if image_size is not None:
            if image_size <= 0:
                raise ValueError("image_size must be positive")

            width, height = img.size
            if random_crop and width >= image_size and height >= image_size:
                left = random.randint(0, width - image_size)
                top = random.randint(0, height - image_size)
                img = img.crop(
                    (left, top, left + image_size, top + image_size)
                )
            else:
                img = img.resize(
                    (image_size, image_size), Image.Resampling.BICUBIC
                )

        array = np.asarray(img, dtype=np.float32) / 255.0

    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PairedDehazeDataset(Dataset):
    """
    Dataset returning aligned hazy/clear image pairs.

    When random cropping is enabled, both images use the same crop
    coordinates, which is essential for pixel-level reconstruction loss.
    """

    def __init__(
        self,
        pairs: Iterable[PairRecord],
        image_size: int | None = 256,
        random_crop: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.random_crop = random_crop

        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]

        with Image.open(pair.hazy) as hazy_file, Image.open(pair.clear) as clear_file:
            hazy_img = hazy_file.convert("RGB")
            clear_img = clear_file.convert("RGB")

            if hazy_img.size != clear_img.size:
                raise ValueError(
                    f"Paired image sizes differ:\n"
                    f"  Hazy:  {pair.hazy} ({hazy_img.size})\n"
                    f"  Clear: {pair.clear} ({clear_img.size})"
                )

            if self.image_size is not None:
                width, height = hazy_img.size

                if (
                    self.random_crop
                    and width >= self.image_size
                    and height >= self.image_size
                ):
                    # Choose coordinates once and apply the same crop to both.
                    left = random.randint(0, width - self.image_size)
                    top = random.randint(0, height - self.image_size)
                    box = (
                        left,
                        top,
                        left + self.image_size,
                        top + self.image_size,
                    )
                    hazy_img = hazy_img.crop(box)
                    clear_img = clear_img.crop(box)
                else:
                    size = (self.image_size, self.image_size)
                    hazy_img = hazy_img.resize(size, Image.Resampling.BICUBIC)
                    clear_img = clear_img.resize(size, Image.Resampling.BICUBIC)

            hazy_array = np.array(hazy_img, dtype=np.float32, copy=True) / 255.0
            clear_array = np.array(clear_img, dtype=np.float32, copy=True) / 255.0

        hazy_tensor = torch.from_numpy(hazy_array).permute(2, 0, 1).contiguous()
        clear_tensor = torch.from_numpy(clear_array).permute(2, 0, 1).contiguous()

        return {
            "hazy": hazy_tensor,
            "clear": clear_tensor,
            "hazy_path": str(pair.hazy),
            "clear_path": str(pair.clear),
            "haze_level": pair.haze_level,
        }
