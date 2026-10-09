from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from rsdehamba_lite.data import PairedDehazeDataset, discover_pairs
from rsdehamba_lite.metrics import psnr, ssim
from rsdehamba_lite.model import RSDehambaLite
from rsdehamba_lite.train_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rsdehamba_lite.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RSDehambaLite(base_dim=cfg["base_dim"], depths=cfg["depths"], use_mamba=cfg["use_mamba"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pairs = discover_pairs(cfg["data_root"])
    loader = DataLoader(PairedDehazeDataset(pairs, cfg["image_size"], random_crop=False), batch_size=1)
    buckets = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            pred = model(hazy)
            level = batch["haze_level"][0]
            buckets[level].append((float(psnr(pred, clear)), float(ssim(pred, clear))))

    for level, scores in sorted(buckets.items()):
        mean_psnr = sum(s[0] for s in scores) / len(scores)
        mean_ssim = sum(s[1] for s in scores) / len(scores)
        print(f"{level}: n={len(scores)} psnr={mean_psnr:.3f} ssim={mean_ssim:.4f}")


if __name__ == "__main__":
    main()

