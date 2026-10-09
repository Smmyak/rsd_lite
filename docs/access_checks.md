# Access Checks

Date checked: 2026-10-09

## Paper

The uploaded path `/mnt/data/rsdehmba.pdf` was not available in this Windows workspace. I used the public arXiv page and arXiv HTML/full-text for the scaffold notes:

- https://arxiv.org/abs/2405.10030

Paper facts used:

- The model is a three-level U-Net-like encoder-decoder.
- It uses Vision Dehamba Blocks as the backbone.
- The reported block schedule is `[2, 3, 3]`.
- It uses SSM and a Direction-aware Scan Module with four directions.
- Training uses Adam, `256x256` patches, initial learning rate `2e-4`, cosine annealing to `1e-6`.
- The reported parameter count for RSDehamba is `1.80M`.

## Official Code

Searches for `RSDehamba official code`, `RSDehamba GitHub`, and related terms found paper index/survey pages, but no official implementation repository. CatalyzeX currently shows a request-code style entry rather than a linked repository.

Conclusion: this project should not claim exact reproduction. It should be framed as an RSDehamba-inspired lightweight variant.

## Dataset

Public references indicate SateHaze1K is available on Kaggle:

- https://www.kaggle.com/datasets/xuxingxing233/satehaze1k

Other literature and dataset summaries also reference an older Dropbox `Haze1k.zip` link. Kaggle access was not verified locally because this environment has no Kaggle CLI installed and no Kaggle credentials.

## Local Environment

The local desktop check found Python `3.14.7` with no installed `torch`, `torchvision`, `PIL`, `numpy`, `pytest`, `kaggle`, or `mamba_ssm`. Because PyTorch support for brand-new Python versions can lag, use Kaggle or a Python `3.10-3.12` environment for training.

Action for the user:

1. On Kaggle, open the dataset link above and add it to the notebook input.
2. Run `python scripts/inspect_dataset.py --data-root /kaggle/input/satehaze1k`.
3. If the folder names differ, update `HAZY_ALIASES` and `CLEAR_ALIASES` in `src/rsdehamba_lite/data.py`.

