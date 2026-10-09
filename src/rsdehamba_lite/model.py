from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional dependency
    Mamba = None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceMixer(nn.Module):
    """Mamba when available, otherwise a small linear-time scan fallback."""

    def __init__(self, dim: int, use_mamba: bool = False) -> None:
        super().__init__()
        self.uses_mamba = bool(use_mamba and Mamba is not None)
        if self.uses_mamba:
            self.mixer = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.mixer = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
                nn.SiLU(inplace=True),
                nn.Conv1d(dim, dim, kernel_size=1),
            )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: B, L, C
        if self.uses_mamba:
            return self.mixer(seq)
        mixed = self.mixer(seq.transpose(1, 2))
        return mixed.transpose(1, 2)


class DirectionAwareScan(nn.Module):
    """Four-path scan module inspired by the paper's DSM."""

    def __init__(self, dim: int, use_mamba: bool = False) -> None:
        super().__init__()
        self.mixers = nn.ModuleList([SequenceMixer(dim, use_mamba) for _ in range(4)])
        self.scan_logits = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _to_sequences(x: torch.Tensor) -> list[torch.Tensor]:
        b, c, h, w = x.shape
        row = x.flatten(2).transpose(1, 2)
        row_rev = torch.flip(row, dims=[1])
        col = x.transpose(2, 3).flatten(2).transpose(1, 2)
        col_rev = torch.flip(col, dims=[1])
        return [row, row_rev, col, col_rev]

    @staticmethod
    def _from_sequence(seq: torch.Tensor, direction: int, h: int, w: int) -> torch.Tensor:
        if direction == 1 or direction == 3:
            seq = torch.flip(seq, dims=[1])
        b, _, c = seq.shape
        if direction in (0, 1):
            return seq.transpose(1, 2).reshape(b, c, h, w)
        return seq.transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        weights = torch.softmax(self.scan_logits, dim=0)
        outs = []
        for idx, (seq, mixer) in enumerate(zip(self._to_sequences(x), self.mixers)):
            outs.append(self._from_sequence(mixer(seq), idx, h, w) * weights[idx])
        return torch.stack(outs, dim=0).sum(dim=0)


class StateSpaceBranch(nn.Module):
    def __init__(self, dim: int, use_mamba: bool = False) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim * 2, 1)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dsm = DirectionAwareScan(dim, use_mamba)
        self.norm = nn.GroupNorm(1, dim)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, features = self.in_proj(x).chunk(2, dim=1)
        gate = F.silu(gate)
        features = F.silu(self.dwconv(features))
        scanned = self.norm(self.dsm(features))
        return self.out_proj(gate * scanned)


class VisionDehambaBlock(nn.Module):
    def __init__(self, dim: int, use_mamba: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.ssm = StateSpaceBranch(dim, use_mamba)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = ConvFFN(dim)
        self.gamma1 = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.ssm(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


def make_stage(dim: int, depth: int, use_mamba: bool) -> nn.Sequential:
    return nn.Sequential(*[VisionDehambaBlock(dim, use_mamba) for _ in range(depth)])


class RSDehambaLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_dim: int = 32,
        depths: Iterable[int] = (1, 2, 2),
        use_mamba: bool = False,
    ) -> None:
        super().__init__()
        d1, d2, d3 = list(depths)
        self.embed = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        self.enc1 = make_stage(base_dim, d1, use_mamba)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)
        self.enc2 = make_stage(base_dim * 2, d2, use_mamba)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)
        self.bottleneck = make_stage(base_dim * 4, d3, use_mamba)

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.fuse2 = nn.Conv2d(base_dim * 4, base_dim * 2, 1)
        self.dec2 = make_stage(base_dim * 2, d2, use_mamba)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.fuse1 = nn.Conv2d(base_dim * 2, base_dim, 1)
        self.dec1 = make_stage(base_dim, d1, use_mamba)

        self.refine = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.embed(x)
        e1 = self.enc1(shallow)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self._match_size(self.up2(b), e2)
        d2 = self.dec2(self.fuse2(torch.cat([d2, e2], dim=1)))
        d1 = self._match_size(self.up1(d2), e1)
        d1 = self.dec1(self.fuse1(torch.cat([d1, e1], dim=1)))
        residual = self.refine(d1)
        return torch.clamp(x + residual, 0.0, 1.0)


@dataclass
class ModelSummary:
    parameters: int
    estimated_paper_baseline_parameters: str = "1.80M reported by paper"
    exact_reproduction: bool = False


def summarize_model(model: nn.Module) -> ModelSummary:
    return ModelSummary(parameters=count_parameters(model))

