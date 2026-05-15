"""
VCP_IR: Visual Conditioning Prompting for All-in-One Image Restoration
Model architecture definition.
"""

import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torchvision import models


# ---------------------------------------------------------------------------
# Utility: reshape helpers
# ---------------------------------------------------------------------------

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# ---------------------------------------------------------------------------
# Layer Normalisation variants
# ---------------------------------------------------------------------------

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Feed-Forward Network (Gated Depthwise Conv)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ---------------------------------------------------------------------------
# Multi-DConv Head Transposed Self-Attention (MDTA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embedding, Downsample, Upsample
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# CBAM (Convolutional Block Attention Module)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        return self.spatial_attn(x)


# ---------------------------------------------------------------------------
# Visual Conditioning Prompt (VCP)
# ---------------------------------------------------------------------------

class VCP(nn.Module):
    """
    Visual Conditioning Prompt block.
    Modulates encoder/decoder features using a visual embedding derived from
    the input image. Uses LazyLinear so the embedding dim is inferred on the
    first forward pass.
    """

    def __init__(self, feature_dim, vis_dim=512, prompt_dim=64):
        super().__init__()
        self.fc = nn.LazyLinear(feature_dim)
        self.cbam = CBAM(feature_dim)
        self.norm = nn.BatchNorm2d(feature_dim)
        self.beta = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        self.reduce_conv = nn.Conv2d(feature_dim, prompt_dim, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, vis_embed):
        B, C, H, W = x.shape
        gating = torch.sigmoid(self.fc(vis_embed)).unsqueeze(-1).unsqueeze(-1)
        f = self.norm(x)
        f = f * (1 + self.gamma * gating) + self.beta
        f = self.cbam(f)
        prompt = f + x
        prompt = self.reduce_conv(prompt)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)
        return prompt


# ---------------------------------------------------------------------------
# VCP_IR_CBAM – main U-Net style model
# ---------------------------------------------------------------------------

class VCP_IR_CBAM(nn.Module):
    """
    VCP-IR with CBAM attention.

    Args:
        inp_channels (int): Input image channels (default 3).
        out_channels (int): Output image channels (default 3).
        dim (int): Base feature dimension (default 48).
        num_blocks (list[int]): Transformer blocks per encoder/decoder level.
        num_refinement_blocks (int): Refinement stage blocks.
        heads (list[int]): Attention heads per level.
        ffn_expansion_factor (float): FFN hidden-dim multiplier.
        bias (bool): Use bias in convolutions.
        LayerNorm_type (str): 'WithBias' or 'BiasFree'.
        decoder (bool): Enable VCP prompt injection in decoder.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=None,
        num_refinement_blocks=4,
        heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        decoder=True,
    ):
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        super().__init__()
        self.decoder = decoder

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # --- VCP prompt modules ---
        if self.decoder:
            self.prompt1 = VCP(feature_dim=96,  vis_dim=512, prompt_dim=64)
            self.prompt2 = VCP(feature_dim=192, vis_dim=512, prompt_dim=128)
            self.prompt3 = VCP(feature_dim=384, vis_dim=512, prompt_dim=320)

        self.chnl_reduce1 = nn.Conv2d(64,  64,  kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        # --- Encoder ---
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(dim * 2 + 128, dim * 2, kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)

        self.reduce_noise_channel_3 = nn.Conv2d(dim * 4 + 256, dim * 4, kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)

        # --- Bottleneck ---
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # --- Decoder ---
        self.up4_3 = Upsample(dim * 4)
        self.reduce_chan_level3 = nn.Conv2d(dim * 2 + 192, dim * 4, kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim * 4 + 512, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(dim * 4 + 512, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim * 2 + 224, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(dim * 2 + 224, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim * 2)
        self.noise_level1 = TransformerBlock(dim * 2 + 64, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(dim * 2 + 64, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # --- Refinement + output ---
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B = inp_img.size(0)

        # Visual embedding: currently a flat view of the input.
        vis_embed = inp_img.view(B, -1)


        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent, vis_embed)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, vis_embed)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, vis_embed)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        return self.output(out_dec_level1) + inp_img
