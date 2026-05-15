"""
Dataset class and evaluation metrics for VCP-IR.
"""

import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
from torchvision import transforms


class BlurSharpDataset(Dataset):
    """
    Paired degraded / ground-truth image dataset.

    Matches files by stem (filename without extension) so the two folders
    can contain images in different formats (e.g. .jpg and .png).

    Args:
        blurry_dir (str): Path to degraded images.
        sharp_dir  (str): Path to ground-truth images.
        size (int): Resize all images to this square resolution.
    """

    def __init__(self, blurry_dir: str, sharp_dir: str, size: int = 256):
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.size = size
        self.to_tensor = transforms.ToTensor()

        blurry_dict = {os.path.splitext(f)[0]: f for f in os.listdir(self.blurry_dir)}
        sharp_dict  = {os.path.splitext(f)[0]: f for f in os.listdir(self.sharp_dir)}

        common_keys = sorted(set(blurry_dict.keys()) & set(sharp_dict.keys()))
        if not common_keys:
            raise ValueError(
                f"No matching image stems found between:\n"
                f"  {blurry_dir}\n  {sharp_dir}"
            )

        self.sharp_files  = [sharp_dict[k]  for k in common_keys]
        self.blurry_files = [blurry_dict[k] for k in common_keys]

    def __len__(self) -> int:
        return len(self.sharp_files)

    def __getitem__(self, index: int):
        fname_sharp  = self.sharp_files[index]
        fname_blurry = self.blurry_files[index]

        sharp_im  = cv2.imread(os.path.join(self.sharp_dir,  fname_sharp))
        blurry_im = cv2.imread(os.path.join(self.blurry_dir, fname_blurry))

        if sharp_im is None or blurry_im is None:
            print(f"[WARNING] Skipping unreadable pair: {fname_sharp}")
            return self.__getitem__((index + 1) % len(self.sharp_files))

        sharp_im  = cv2.cvtColor(cv2.resize(sharp_im,  (self.size, self.size)), cv2.COLOR_BGR2RGB)
        blurry_im = cv2.cvtColor(cv2.resize(blurry_im, (self.size, self.size)), cv2.COLOR_BGR2RGB)

        in_tensor = self.to_tensor(blurry_im)
        gt_tensor = self.to_tensor(sharp_im)

        # Per-pixel absolute-error map (used as auxiliary info, not for loss)
        diff_mask = torch.mean(torch.abs(in_tensor - gt_tensor), dim=0, keepdim=True)

        return in_tensor, gt_tensor, diff_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio for normalised tensors in [0, 1].

    Args:
        img1, img2: Tensors of shape (B, C, H, W) or (C, H, W).

    Returns:
        Scalar PSNR value in dB.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim_score(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    SSIM for a single (C, H, W) tensor pair, evaluated on CPU.

    Args:
        img1, img2: Single-image tensors (C, H, W) in [0, 1].

    Returns:
        SSIM float in [-1, 1].
    """
    a = img1.permute(1, 2, 0).detach().cpu().numpy()
    b = img2.permute(1, 2, 0).detach().cpu().numpy()
    return ssim(a, b, channel_axis=-1, data_range=1.0)
