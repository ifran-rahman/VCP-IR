# VCP-IR: Visual Conditioning Prompting for All-in-One Image Restoration

A Restormer-style hierarchical U-Net enhanced with **Visual Conditioning Prompts (VCP)** The model takes a degraded image, derives a visual embedding from it, and uses that embedding to adaptively condition each decoder stage — enabling the network to simultaneously handle multiple restoration tasks (deblurring, denoising, deraining, etc.).

---

## Repository layout

```
vcp_ir/
├── vcp_ir/
│   ├── __init__.py      # Public API
│   ├── model.py         # VCP_IR architecture
│   ├── dataset.py       # BlurSharpDataset + PSNR / SSIM metrics
│   └── losses.py        # CharbonnierLoss, CombinedLoss
├── train.py             # Full training loop
├── evaluate.py          # Evaluation & visualisation
├── sanity_check.py      # Quick forward-pass & model summary
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / download the repo
git clone <repo-url> && cd vcp_ir

# 2. Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install PyTorch for your CUDA version from https://pytorch.org/
#    Example (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Data layout

The dataset expects two parallel folders with **matching filenames** (stems may differ in extension):

```
data/
├── degraded/   ← blurry / noisy / hazy input images
└── gt/         ← clean ground-truth images
```

> Files are matched by stem, so `degraded/0001.jpg` is paired with `gt/0001.png`.

---

## Quickstart

### 1 — Verify the model

```bash
python sanity_check.py --size 128
```

### 2 — Train

```bash
python train.py \
    --degraded data/degraded \
    --gt       data/gt \
    --size     128 \
    --epochs   500 \
    --batch    4 \
    --out      checkpoints
```

Resume from a checkpoint:

```bash
python train.py \
    --degraded data/degraded \
    --gt       data/gt \
    --resume   checkpoints/best_model.pth
```

### 3 — Evaluate

```bash
python evaluate.py \
    --degraded   data/degraded \
    --gt         data/gt \
    --checkpoint checkpoints/best_model.pth \
    --out        results/eval
```

Each output image is a side-by-side grid: **input | prediction | ground truth**.

---

## Key CLI arguments

| Script | Argument | Default | Description |
|--------|----------|---------|-------------|
| `train.py` | `--degraded` | — | Path to degraded images |
| | `--gt` | — | Path to ground-truth images |
| | `--size` | 128 | Image resize resolution |
| | `--epochs` | 500 | Max training epochs |
| | `--batch` | 4 | Batch size |
| | `--lr` | 1e-4 | Adam learning rate |
| | `--patience` | 30 | Early-stopping patience |
| | `--dim` | 48 | Base feature dimension |
| | `--no_decoder` | False | Disable VCP prompt injection |
| | `--resume` | "" | Checkpoint to resume from |
| `evaluate.py` | `--checkpoint` | — | Path to `.pth` file |
| | `--out` | `results/eval` | Output directory |

---

## Architecture overview


* **VCP block**: derives a gating vector from the flattened input image via `LazyLinear`, and produces a spatial prompt that is concatenated into the decoder.  
* **Loss**: Charbonnier pixel loss + 0.2 × Laplacian high-frequency L1 loss.

---
