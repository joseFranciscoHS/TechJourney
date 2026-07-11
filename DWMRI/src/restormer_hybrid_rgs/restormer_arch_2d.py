"""
Restormer2D: Capacity-matched 2D Restormer for Hybrid RGS.

Ported from the original Restormer (Zamir et al., CVPR 2022):
  https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py

Key differences from the original:
- 3-level hierarchy instead of 4-level (matches Restormer3D used in this repo)
- inp_channels=K (RGS context gradients + masked target), out_channels=1
- Global input residual `out + inp_img` removed (K→1 makes it undefined)
- Learned scale_and_shift + PReLU output activation (matching Restormer3D)
- reflect-pad / crop in forward to handle non-multiple-of-4 spatial sizes safely

All 2D MDTA / GDFN / PixelUnshuffle blocks are structurally identical to the
original; only the outer U-Net topology and I/O interface differ.
"""

import logging
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

##########################################################################
# Layer Normalization (identical to original Restormer)
##########################################################################


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


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


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN) — identical to original
##########################################################################


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA) — identical to original
##########################################################################


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        return self.project_out(out)


##########################################################################
# Transformer Block — identical to original
##########################################################################


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


##########################################################################
# Patch embedding — identical to original
##########################################################################


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        return self.proj(x)


##########################################################################
# Resampling — Conv2d + PixelUnshuffle/PixelShuffle (identical to original)
##########################################################################


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
# Restormer2D — 3-level Hybrid RGS adaptation
##########################################################################


class Restormer2D(nn.Module):
    """
    Capacity-matched 2D Restormer for Hybrid RGS.

    Topology mirrors Restormer3D (3 levels, 2 downsampling stages) but uses
    the original 2D MDTA/GDFN blocks and PixelUnshuffle/PixelShuffle resampling
    from Zamir et al. (CVPR 2022) instead of 3D convolutions and strided Conv3d.

    Hybrid RGS I/O adaptations vs the original Restormer:
    - inp_channels = K  (context gradients + Bernoulli-masked target at index K-1)
    - out_channels = 1  (single denoised target volume)
    - Global residual `out + inp_img` removed (K→1 makes it undefined)
    - scale_and_shift learnable affine + PReLU output activation added

    PixelUnshuffle requires H,W divisible by 2^(n_downsamples) = 4 for 3-level.
    forward() reflect-pads to the next multiple of 4 and crops back, making
    this a no-op for the paper D-Brain sizes (128×128, patches 16/24/32).

    Args:
        inp_channels: K gradient volumes (context + masked target).
        out_channels: 1 for single denoised volume output.
        dim: Base feature dimension. Default 12 matches Restormer3D.
        num_blocks: Transformer blocks per encoder/decoder level (3 values).
        num_refinement_blocks: Refinement blocks after decoder.
        heads: Attention heads per level (3 values).
        ffn_expansion_factor: GDFN expansion factor.
        bias: Conv bias flag.
        LayerNorm_type: 'WithBias' or 'BiasFree'.
        output_activation: 'prelu' (default) or 'sigmoid'.
        scale_and_shift: Learnable output affine (scale * out + shift).
    """

    def __init__(
        self,
        inp_channels: int = 16,
        out_channels: int = 1,
        dim: int = 12,
        num_blocks=(1, 2, 2),
        num_refinement_blocks: int = 2,
        heads=(1, 2, 2),
        ffn_expansion_factor: float = 1.5,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        output_activation: str = "prelu",
        scale_and_shift: bool = True,
    ):
        super().__init__()
        assert len(num_blocks) == 3, (
            "num_blocks must have 3 entries (3-level hierarchy)"
        )
        assert len(heads) == 3, "heads must have 3 entries (3-level hierarchy)"
        self.scale_and_shift = scale_and_shift

        logging.info(
            "Initializing Restormer2D (3-level): inp_channels=%d, out_channels=%d, "
            "dim=%d, num_blocks=%s, heads=%s, ffn_expansion_factor=%.2f, "
            "output_activation=%s, scale_and_shift=%s",
            inp_channels,
            out_channels,
            dim,
            list(num_blocks),
            list(heads),
            ffn_expansion_factor,
            output_activation,
            scale_and_shift,
        )

        _blk = lambda d, h, n: nn.Sequential(  # noqa: E731
            *[
                TransformerBlock(
                    dim=d,
                    num_heads=h,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(n)
            ]
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        # Encoder
        self.encoder_level1 = _blk(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)  # → dim*2, H/2, W/2
        self.encoder_level2 = _blk(int(dim * 2), heads[1], num_blocks[1])
        self.down2_3 = Downsample(int(dim * 2))  # → dim*4, H/4, W/4
        self.latent = _blk(int(dim * 4), heads[2], num_blocks[2])

        # Decoder
        self.up3_2 = Upsample(int(dim * 4))  # → dim*2, H/2, W/2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 4), int(dim * 2), kernel_size=1, bias=bias
        )
        self.decoder_level2 = _blk(int(dim * 2), heads[1], num_blocks[1])

        self.up2_1 = Upsample(int(dim * 2))  # → dim, H, W; skip concat → dim*2
        self.decoder_level1 = _blk(int(dim * 2), heads[0], num_blocks[0])

        self.refinement = _blk(int(dim * 2), heads[0], num_refinement_blocks)

        # Output projection with activation
        act = (
            nn.Sigmoid()
            if output_activation.lower() == "sigmoid"
            else nn.PReLU(out_channels)
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim * 2),
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            act,
        )

        if self.scale_and_shift:
            self.output_scale = nn.Parameter(torch.ones(1))
            self.output_shift = nn.Parameter(torch.ones(1))

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("Restormer2D total parameters: %d", total_params)
        logging.info("Restormer2D trainable parameters: %d", trainable_params)

    def forward(self, inp_img: torch.Tensor, orientation_info=None) -> torch.Tensor:
        del orientation_info  # shared fit/reconstruct API; unused for Restormer-2D
        """
        Args:
            inp_img: (B, K, H, W) — K gradient slices with masked target at index K-1.

        Returns:
            (B, 1, H, W) — denoised target slice.
        """
        # Reflect-pad H/W to the next multiple of 4 (needed for two PixelUnshuffle(2) stages).
        # For the D-Brain paper protocol (128×128 full slices, patches 16/24/32), h%4==0 and
        # w%4==0, so pad_h=pad_w=0 and this is a no-op. For any other input size it is a
        # transparent safety mechanism — no residual path is disturbed.
        _, _, h, w = inp_img.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        x = (
            F.pad(inp_img, (0, pad_w, 0, pad_h), mode="reflect")
            if (pad_h or pad_w)
            else inp_img
        )

        # Encoder
        x1 = self.encoder_level1(self.patch_embed(x))
        x2 = self.encoder_level2(self.down1_2(x1))
        lat = self.latent(self.down2_3(x2))

        # Decoder
        d2 = self.up3_2(lat)
        d2 = self.reduce_chan_level2(torch.cat([d2, x2], dim=1))
        d2 = self.decoder_level2(d2)

        d1 = self.up2_1(d2)
        d1 = self.decoder_level1(torch.cat([d1, x1], dim=1))
        d1 = self.refinement(d1)

        out = self.output(d1)
        if self.scale_and_shift:
            out = self.output_scale * out + self.output_shift

        # Crop back to original spatial size
        return out[:, :, :h, :w]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = Restormer2D(inp_channels=16, out_channels=1)
    for hw in [(32, 32), (128, 128), (30, 31)]:
        y = m(torch.randn(2, 16, *hw))
        assert y.shape == (2, 1, *hw), f"Shape mismatch for {hw}: {y.shape}"
        logging.info("hw=%s → output %s OK", hw, tuple(y.shape))
    n_params = sum(p.numel() for p in m.parameters())
    logging.info("Total params: %d (%.3fM)", n_params, n_params / 1e6)
