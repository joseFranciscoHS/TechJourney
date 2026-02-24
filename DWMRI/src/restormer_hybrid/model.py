"""
Restormer3D: Efficient Transformer for 3D Volumetric Image Restoration

Adapted from the original Restormer (Zamir et al., CVPR 2022) for 3D medical imaging.
All 2D operations have been converted to their 3D equivalents for volumetric DWMRI data.
"""

import logging
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
# 3D Reshape Utilities
##########################################################################

def to_2d(x):
    """Reshape 5D tensor (B, C, D, H, W) to 3D tensor (B, D*H*W, C) for LayerNorm."""
    return rearrange(x, 'b c d h w -> b (d h w) c')


def to_5d(x, d, h, w):
    """Reshape 3D tensor (B, D*H*W, C) back to 5D tensor (B, C, D, H, W)."""
    return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)


##########################################################################
# Layer Normalization for 3D
##########################################################################

class BiasFree_LayerNorm(nn.Module):
    """Layer normalization without bias term."""
    
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
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
    """Layer normalization with bias term."""
    
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
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


class LayerNorm3D(nn.Module):
    """
    Layer Normalization for 3D tensors.
    
    Applies layer normalization over the channel dimension of a 5D tensor (B, C, D, H, W).
    """
    
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm3D, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return to_5d(self.body(to_2d(x)), d, h, w)


##########################################################################
# Gated-DConv Feed-Forward Network (GDFN) - 3D Version
##########################################################################

class FeedForward3D(nn.Module):
    """
    Gated-DConv Feed-Forward Network for 3D data.
    
    Uses depthwise separable 3D convolutions with gating mechanism.
    """
    
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward3D, self).__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv3d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA) - 3D Version
##########################################################################

class Attention3D(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention for 3D data.
    
    Computes channel-wise attention (transposed attention) which is more
    memory efficient for high-resolution 3D volumes.
    """
    
    def __init__(self, dim, num_heads, bias):
        super(Attention3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, d, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Transposed attention: (C/heads, D*H*W) @ (D*H*W, C/heads) -> (C/heads, C/heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w',
                        head=self.num_heads, d=d, h=h, w=w)
        
        out = self.project_out(out)
        return out


##########################################################################
# Transformer Block - 3D Version
##########################################################################

class TransformerBlock3D(nn.Module):
    """
    Transformer block for 3D data.
    
    Consists of Multi-DConv Head Transposed Self-Attention followed by
    Gated-DConv Feed-Forward Network.
    """
    
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock3D, self).__init__()
        
        self.norm1 = LayerNorm3D(dim, LayerNorm_type)
        self.attn = Attention3D(dim, num_heads, bias)
        self.norm2 = LayerNorm3D(dim, LayerNorm_type)
        self.ffn = FeedForward3D(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
# Overlapped Patch Embedding - 3D Version
##########################################################################

class OverlapPatchEmbed3D(nn.Module):
    """
    Overlapped patch embedding using 3x3x3 convolution.
    """
    
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed3D, self).__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


##########################################################################
# Resizing Modules - 3D Version
##########################################################################

class Downsample3D(nn.Module):
    """
    Downsampling module for 3D data using strided convolution.
    Reduces spatial dimensions by factor of 2 while doubling channels.
    """
    
    def __init__(self, n_feat):
        super(Downsample3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.body(x)


class Upsample3D(nn.Module):
    """
    Upsampling module for 3D data using transposed convolution.
    Increases spatial dimensions by factor of 2 while halving channels.
    """
    
    def __init__(self, n_feat):
        super(Upsample3D, self).__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose3d(n_feat, n_feat // 2, kernel_size=2, stride=2, bias=False)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
# Restormer3D - Main Architecture
##########################################################################

class Restormer3D(nn.Module):
    """
    3D Restormer: Efficient Transformer for Volumetric Image Restoration.
    
    Adapted from the original 2D Restormer for 3D medical imaging applications.
    Uses a U-Net style encoder-decoder architecture with transformer blocks.
    
    Args:
        inp_channels: Number of input channels (e.g., 10 for all DWI volumes)
        out_channels: Number of output channels (e.g., 1 for single denoised volume)
        dim: Base feature dimension
        num_blocks: Number of transformer blocks at each encoder/decoder level
        num_refinement_blocks: Number of refinement blocks after decoder
        heads: Number of attention heads at each level
        ffn_expansion_factor: Expansion factor for feed-forward network
        bias: Whether to use bias in convolutions
        LayerNorm_type: Type of layer normalization ('WithBias' or 'BiasFree')
        output_activation: Activation function for output ('prelu' or 'sigmoid')
    """
    
    def __init__(
        self,
        inp_channels=10,
        out_channels=1,
        dim=32,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type='WithBias',
        output_activation='prelu',
    ):
        super(Restormer3D, self).__init__()
        
        logging.info(
            f"Initializing Restormer3D: inp_channels={inp_channels}, out_channels={out_channels}, "
            f"dim={dim}, num_blocks={num_blocks}, heads={heads}, "
            f"ffn_expansion_factor={ffn_expansion_factor}, output_activation={output_activation}"
        )
        
        self.patch_embed = OverlapPatchEmbed3D(inp_channels, dim)
        
        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        
        # Level 1 to Level 2
        self.down1_2 = Downsample3D(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 1), num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        
        # Level 2 to Level 3
        self.down2_3 = Downsample3D(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 2), num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        
        # Level 3 to Level 4 (Latent/Bottleneck)
        self.down3_4 = Downsample3D(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 3), num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])
        
        # Level 4 to Level 3 (Decoder)
        self.up4_3 = Upsample3D(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv3d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 2), num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        
        # Level 3 to Level 2
        self.up3_2 = Upsample3D(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 1), num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        
        # Level 2 to Level 1
        self.up2_1 = Upsample3D(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 1), num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        
        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock3D(
                dim=int(dim * 2 ** 1), num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])
        
        # Output projection
        if output_activation.lower() == 'sigmoid':
            self.output = nn.Sequential(
                nn.Conv3d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.Sigmoid()
            )
        else:
            self.output = nn.Sequential(
                nn.Conv3d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.PReLU(out_channels)
            )
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

    def forward(self, inp_img):
        """
        Forward pass.
        
        Args:
            inp_img: Input tensor of shape (B, num_vols, D, H, W)
        
        Returns:
            Output tensor of shape (B, out_channels, D, H, W)
        """
        # Patch embedding
        inp_enc_level1 = self.patch_embed(inp_img)
        
        # Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        
        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # Refinement and output
        out_dec_level1 = self.refinement(out_dec_level1)
        out = self.output(out_dec_level1)
        
        return out


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = Restormer3D(
        inp_channels=10,
        out_channels=1,
        dim=32,
        num_blocks=[2, 2, 2, 4],
        heads=[1, 2, 4, 8],
    )
    
    # Test forward pass
    x = torch.randn(1, 10, 16, 16, 16)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
