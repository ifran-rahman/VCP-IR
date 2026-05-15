"""
Loss functions for VCP-IR training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt((pred - target)^2 + eps^2).
    A smooth, robust approximation of L1 that handles outliers better than MSE.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps ** 2).mean()


def laplacian(img: torch.Tensor) -> torch.Tensor:
    """Apply a Laplacian high-frequency filter per channel."""
    B, C, H, W = img.shape
    kernel = torch.tensor(
        [[0, -1,  0],
         [-1,  4, -1],
         [0, -1,  0]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return F.conv2d(img, kernel, padding=1, groups=C)


class CombinedLoss(nn.Module):
    """
    Charbonnier pixel loss + weighted Laplacian high-frequency loss.

    Args:
        charbonnier_eps (float): Smoothing factor for Charbonnier loss.
        hf_weight (float): Weight for the high-frequency (Laplacian) term.
    """

    def __init__(self, charbonnier_eps: float = 1e-3, hf_weight: float = 0.2):
        super().__init__()
        self.charb = CharbonnierLoss(eps=charbonnier_eps)
        self.hf_weight = hf_weight

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_char = self.charb(output, target)
        loss_hf = F.l1_loss(laplacian(output), laplacian(target))
        return l_char + self.hf_weight * loss_hf
