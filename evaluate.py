"""
evaluate.py – Run validation and save visualisations for a trained VCP-IR checkpoint.

Usage
-----
    python evaluate.py \
        --degraded data/degraded \
        --gt       data/gt \
        --checkpoint checkpoints/best_model.pth \
        --out      results/eval

Outputs
-------
    results/eval/<index>.png   – side-by-side grid: input | predicted | target
    Prints mean PSNR and SSIM over the full dataset.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from vcp_ir import VCP_IR_CBAM, BlurSharpDataset, psnr, ssim_score


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained VCP-IR model")
    p.add_argument("--degraded",   required=True, help="Path to degraded images folder")
    p.add_argument("--gt",         required=True, help="Path to ground-truth images folder")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--out",        default="results/eval", help="Output directory for grids")
    p.add_argument("--size",       type=int, default=128, help="Image resize resolution")
    p.add_argument("--workers",    type=int, default=0)
    # Model architecture must match the checkpoint
    p.add_argument("--dim",         type=int, default=48)
    p.add_argument("--num_blocks",  nargs=4, type=int, default=[1, 1, 1, 1], metavar="N")
    p.add_argument("--refinement",  type=int, default=4)
    p.add_argument("--no_decoder",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # --- Dataset ---
    dataset = BlurSharpDataset(args.degraded, args.gt, size=args.size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    # --- Model ---
    model = VCP_IR_CBAM(
        inp_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.refinement,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        decoder=not args.no_decoder,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Evaluate ---
    all_psnr, all_ssim = [], []

    with torch.no_grad():
        for idx, (inp, tgt, _) in enumerate(tqdm(loader, desc="Evaluating")):
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp)

            all_psnr.append(psnr(out, tgt).item())
            all_ssim.append(ssim_score(out[0], tgt[0]))

            inp_v = torch.clamp(inp[:, :3], 0, 1)
            out_v = torch.clamp(out[:, :3], 0, 1)
            tgt_v = torch.clamp(tgt[:, :3], 0, 1)

            grid = make_grid(torch.cat([inp_v, out_v, tgt_v], dim=0), nrow=3, padding=2)
            save_image(grid, os.path.join(args.out, f"{idx:05d}.png"))

    print(f"\n=== Evaluation Results ===")
    print(f"  Images : {len(all_psnr)}")
    print(f"  PSNR   : {np.mean(all_psnr):.2f} dB")
    print(f"  SSIM   : {np.mean(all_ssim):.4f}")
    print(f"  Grids  → {args.out}/")


if __name__ == "__main__":
    main()
