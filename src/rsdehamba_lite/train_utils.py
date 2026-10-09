from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_split(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]

