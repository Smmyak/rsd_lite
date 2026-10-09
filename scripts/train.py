from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsdehamba_lite.data import PairedDehazeDataset, discover_pairs
from rsdehamba_lite.metrics import psnr, ssim
from rsdehamba_lite.model import RSDehambaLite, summarize_model
from rsdehamba_lite.train_utils import load_config, set_seed, train_val_split


def validate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            pred = model(hazy)
            total_psnr += float(psnr(pred, clear))
            total_ssim += float(ssim(pred, clear))
            count += 1
    return {"psnr": total_psnr / max(count, 1), "ssim": total_ssim / max(count, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rsdehamba_lite.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.subset_fraction is not None:
        cfg["subset_fraction"] = args.subset_fraction
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RSDehambaLite(
        base_dim=int(cfg["base_dim"]),
        depths=cfg["depths"],
        use_mamba=bool(cfg["use_mamba"]),
    ).to(device)
    print(f"device: {device}")
    print(f"parameters: {summarize_model(model).parameters:,}")

    if args.dry_run:
        x = torch.rand(1, 3, cfg["image_size"], cfg["image_size"], device=device)
        y = model(x)
        print(f"dry-run output shape: {tuple(y.shape)}")
        return

    pairs = discover_pairs(cfg["data_root"])
    if not pairs:
        raise SystemExit("No image pairs found. Run scripts/inspect_dataset.py first.")
    if float(cfg["subset_fraction"]) < 1.0:
        keep = max(1, int(len(pairs) * float(cfg["subset_fraction"])))
        pairs = pairs[:keep]

    train_pairs, val_pairs = train_val_split(pairs, float(cfg["val_fraction"]), int(cfg["seed"]))
    train_loader = DataLoader(
        PairedDehazeDataset(train_pairs, cfg["image_size"], random_crop=True),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        PairedDehazeDataset(val_pairs, cfg["image_size"], random_crop=False),
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    output_dir = Path(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"]
    )

    best_psnr = -1.0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            pred = model(hazy)
            loss = F.l1_loss(pred, clear)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        scheduler.step()
        metrics = validate(model, val_loader, device) if val_pairs else {"psnr": 0.0, "ssim": 0.0}
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch} loss={mean_loss:.5f} val_psnr={metrics['psnr']:.3f} val_ssim={metrics['ssim']:.4f}")

        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics}
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save(state, ckpt_dir / "best.pt")
        if epoch % int(cfg["save_every"]) == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()

