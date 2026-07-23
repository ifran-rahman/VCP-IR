"""
train.py – VCP-IR training script.

Usage
-----
    python train.py \
        --degraded data/degraded \
        --gt       data/gt \
        --epochs   500 \
        --batch    4 \
        --size     128 \
        --out      checkpoints

All CLI arguments have sensible defaults; see --help for the full list.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from vcp_ir import VCP_IR, BlurSharpDataset, CombinedLoss, psnr, ssim_score


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VCP-IR image restoration model")

    # Data
    p.add_argument(
        "--degraded",
        default="/home/ifran/Projects_UBUNTU/Prompt-Aware-Restoration/i2i/Final Files/dataset/ucmix2/degraded",
        help="Path to degraded images folder"
    )

    p.add_argument(
        "--gt",
        default="/home/ifran/Projects_UBUNTU/Prompt-Aware-Restoration/i2i/Final Files/dataset/ucmix2/gt",
        help="Path to ground-truth images folder"
    )

    p.add_argument("--size",      type=int,   default=128)
    p.add_argument("--val_split", type=float, default=0.2)

    # Training
    p.add_argument("--epochs",   type=int,   default=500)
    p.add_argument("--batch",    type=int,   default=4)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--patience", type=int,   default=30)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--workers",  type=int,   default=0)

    # Model
    p.add_argument("--dim",        type=int, default=48)

    p.add_argument(
        "--num_blocks",
        nargs=4,
        type=int,
        default=[1, 1, 1, 1],
        metavar="N"
    )

    p.add_argument("--refinement", type=int, default=4)

    p.add_argument(
        "--no_decoder",
        action="store_true",
        help="Disable VCP prompt injection"
    )

    # I/O
    p.add_argument("--out",     default="VCP_IR")
    p.add_argument("--results", default="results")

    p.add_argument(
        "--resume",
        default="",
        help="Path to checkpoint to resume from"
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out,     exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    full_dataset = BlurSharpDataset(args.degraded, args.gt, size=args.size)
    val_len   = int(args.val_split * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"Dataset: {len(full_dataset)} pairs  |  Train: {train_len}  Val: {val_len}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = VCP_IR(
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

    criterion = CombinedLoss(charbonnier_eps=1e-3, hf_weight=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_psnr   = -1e9

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0)
        best_psnr   = ckpt.get("best_psnr", -1e9)
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, best PSNR {best_psnr:.2f})")
    else:
        print("Starting from scratch.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        for inp, tgt, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out  = model(inp)
                loss = criterion(out, tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_psnr_list, val_ssim_list = [], []

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                for inp, tgt, _ in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    out = model(inp)
                    val_psnr_list.append(psnr(out, tgt).item())
                    val_ssim_list.append(ssim_score(out[0], tgt[0]))

        mean_psnr = float(np.mean(val_psnr_list))
        mean_ssim = float(np.mean(val_ssim_list))

        print(
            f"Epoch [{epoch + 1}/{args.epochs}]  "
            f"Train Loss: {train_loss / len(train_loader):.4f}  |  "
            f"Val PSNR: {mean_psnr:.2f}  SSIM: {mean_ssim:.4f}"
        )

        # --- Checkpoint ---
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            patience_counter = 0
            save_path = os.path.join(args.out, "best_model.pth")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "best_psnr": best_psnr},
                save_path,
            )
            print(f"  ✅ New best (PSNR {best_psnr:.2f}) → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  ⛔ Early stopping after {epoch + 1} epochs.")
                break

        # Periodic checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1},
                os.path.join(args.out, f"epoch_{epoch + 1:04d}.pth"),
            )

    # ------------------------------------------------------------------
    # Final visualisation
    # ------------------------------------------------------------------
    print("Training complete. Saving final validation samples…")
    model.eval()
    sample_inp, sample_tgt, _ = next(iter(val_loader))
    sample_inp = sample_inp.to(device)
    sample_tgt = sample_tgt.to(device)

    with torch.no_grad():
        sample_out = model(sample_inp)

    def clamp3(t):
        return torch.clamp(t[:, :3], 0, 1)

    grid = make_grid(torch.cat([clamp3(sample_inp), clamp3(sample_out), clamp3(sample_tgt)], dim=0), nrow=3)
    save_path = os.path.join(args.results, "final_comparison.png")
    save_image(grid, save_path)
    print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()
