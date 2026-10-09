# Project Specification

## Goal

Build and evaluate a reduced-VDB RSDehamba-style model for SateHaze1K within one week.

## Baseline From Paper

The paper reports RSDehamba with block schedule `[2, 3, 3]`, a three-level U-Net architecture, VDB blocks, SSM for long-range modeling, and DSM for four-direction scanning. It reports `1.80M` parameters.

## Proposed Variant

`RSDehambaLite-[1,2,2]`:

- Keep the three-level encoder-decoder.
- Keep VDB-style residual blocks.
- Keep a branch shaped like the paper's SSM equations: projection, depthwise convolution, SiLU, DSM, normalization, gated multiplication, and output projection.
- Reduce VDB depth from `[2,3,3]` to `[1,2,2]`.
- Use `base_dim=32` by default to stay below the reported baseline parameter count.

## Non-Reproduction Caveats

The paper does not provide all implementation details needed for exact reproduction. In particular, DSM ordering, selective scan internals, exact channel widths, decoder refinement, and data split scripts are not fully specified in public text. This repository therefore prioritizes a transparent staged implementation.

## Evaluation

Primary metrics:

- RGB PSNR
- RGB SSIM
- Trainable parameter count

Recommended reporting:

- Thin, Moderate, Thick haze subsets separately
- Average across all discovered pairs
- Parameter count comparison against the paper's reported `1.80M`, clearly labeled as reported baseline
- Runtime or images/sec if the Kaggle GPU runtime allows it

## Minimum Success Criteria

- Dataset inspector detects paired images.
- Model passes shape and gradient tests.
- Training dry run succeeds.
- A small subset can overfit or show decreasing L1 loss.
- Full training produces evaluation metrics and sample outputs.

