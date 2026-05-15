"""
sanity_check.py – Verify the model can do a forward pass and print its summary.

Usage
-----
    python sanity_check.py            # default 128x128 input
    python sanity_check.py --size 256 # custom resolution
"""

import argparse
import torch
from torchinfo import summary
from vcp_ir import VCP_IR_CBAM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=128, help="Spatial resolution for test input")
    p.add_argument("--dim",  type=int, default=48)
    p.add_argument("--no_decoder", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VCP_IR_CBAM(
        inp_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=[1, 1, 1, 1],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        decoder=not args.no_decoder,
    ).to(device)

    inp = torch.randn(1, 3, args.size, args.size, device=device)
    out = model(inp)
    assert out.shape == inp.shape, f"Shape mismatch: {out.shape} != {inp.shape}"
    print(f"✅ Forward pass OK — input {tuple(inp.shape)} → output {tuple(out.shape)}\n")

    summary(model, input_size=(1, 3, args.size, args.size))


if __name__ == "__main__":
    main()
