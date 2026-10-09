# RSDehamba-Lite One-Week Project

This repository is a practical, staged starting point for a lightweight RSDehamba-style remote sensing dehazing project. The target variant reduces the paper's VDB block schedule from `[2, 3, 3]` to `[1, 2, 2]` while preserving the broad U-Net, VDB, SSM-branch, and four-direction DSM ideas where the public paper gives enough detail.

Important scope note: this is not an exact reproduction of RSDehamba. The paper describes the architecture at a high level but does not provide complete official implementation details, and no official code was found during the access check. This scaffold is meant to be runnable, testable, and easy to adapt on Kaggle.

## What Is Included

- Reduced VDB model: `RSDehambaLite(depths=[1, 2, 2])`
- Direction-aware scan module with 4 scan paths
- Optional `mamba_ssm` integration when installed
- Pure PyTorch scan fallback for fast bring-up
- SateHaze1K dataset layout inspection
- Paired image dataset loader
- Training/evaluation scripts with PSNR/SSIM
- Shape, gradient, and parameter tests
- Kaggle-oriented setup notes

## Research Sources Checked

- RSDehamba arXiv paper: https://arxiv.org/abs/2405.10030
- SateHaze1K Kaggle reference found in recent literature: https://www.kaggle.com/datasets/xuxingxing233/satehaze1k
- Older SateHaze1K Dropbox reference appears in dataset summaries and papers.
- Public searches found survey/list pages that list RSDehamba paper links, but no official implementation repository.

See `docs/access_checks.md` for details and caveats.

## Quick Start

Use Python 3.10-3.12. Kaggle is recommended for the first real training run because this local workspace currently has Python 3.14 and no ML stack installed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts/env_check.py
pytest -q
```

On Kaggle, create a notebook, add the SateHaze1K dataset if accessible, then upload or clone this project and run:

```bash
git clone https://github.com/Smmyak/rsdehamba-lite.git
cd rsdehamba-lite
pip install -r requirements.txt
pip install -e .
python scripts/inspect_dataset.py --data-root /kaggle/input/satehaze1k
python scripts/train.py --config configs/rsdehamba_lite.json
python scripts/evaluate.py --config configs/rsdehamba_lite.json --checkpoint outputs/checkpoints/best.pt
```

See `KAGGLE.md` for the full step-by-step Kaggle workflow.

## Expected Dataset Layout

The inspector supports common paired layouts and tries to infer hazy/clear image roots:

```text
SateHaze1K/
  Thin/
    hazy/
    clear/
  Moderate/
    hazy/
    clear/
  Thick/
    hazy/
    clear/
```

It also searches for aliases such as `input`, `target`, `GT`, `clean`, `fog`, and `haze`. Run the inspector first because Kaggle mirrors sometimes use different folder names.

## One-Week Plan

1. Day 1: Verify dataset access, inspect folder layout, run model tests.
2. Day 2: Train on a tiny subset to validate the full loop.
3. Day 3-4: Train the `[1,2,2]` variant at `256x256` patches.
4. Day 5: Evaluate PSNR/SSIM by haze level and save sample outputs.
5. Day 6: Compare parameter count and runtime against the paper's reported 1.80M baseline, without claiming exact reproduction.
6. Day 7: Prepare report: motivation, method change, assumptions, results, limitations.

## Project Assumptions

- The paper reports RSDehamba uses a three-level U-Net with VDB blocks `[2,3,3]`, patch size `256x256`, Adam, initial learning rate `2e-4`, cosine annealing to `1e-6`, and RGB PSNR/SSIM.
- This project changes only the block schedule by default: `[1,2,2]`.
- The pure PyTorch fallback is a runnable approximation of the scan branch, not the exact selective scan kernel from Mamba.
- If `mamba_ssm` is installed, the scan module can use it through `--use-mamba`.

## Useful Commands

```bash
python scripts/inspect_dataset.py --data-root /path/to/SateHaze1K
python scripts/train.py --config configs/rsdehamba_lite.json --dry-run
python scripts/train.py --config configs/rsdehamba_lite_smoke.json
python scripts/train.py --config configs/rsdehamba_lite.json
python scripts/evaluate.py --config configs/rsdehamba_lite.json --checkpoint outputs/checkpoints/best.pt
pytest -q
```

