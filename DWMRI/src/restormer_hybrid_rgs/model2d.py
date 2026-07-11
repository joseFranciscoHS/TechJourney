"""
Lightweight residual 2D CNN for the Hybrid RGS capacity ablation (Res-CNN-2D).

This is NOT a Restormer port. For the true 2D Restormer with MDTA/GDFN blocks,
see :class:`restormer_arch_2d.Restormer2D`.

The paper refers to this model as "Res-CNN-2D" in Table tab:3d_vs_2d.
"""

import torch.nn as nn


class _Block2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class ResCNN2D(nn.Module):
    """
    Lightweight residual CNN for the Hybrid RGS 2D capacity ablation (Res-CNN-2D).

    Not a Restormer port — see :class:`restormer_arch_2d.Restormer2D` for MDTA/GDFN.
    Input: (B, K, H, W), Output: (B, 1, H, W)
    """

    def __init__(
        self,
        inp_channels=10,
        out_channels=1,
        dim=24,
        num_blocks=2,
        heads=2,  # unused; kept for call-site parity with Restormer2D kwargs
        ffn_expansion_factor=2.0,  # unused
        bias=False,
        LayerNorm_type="WithBias",  # unused
    ):
        super().__init__()
        self.embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=bias)
        self.blocks = nn.Sequential(*[_Block2D(dim=dim) for _ in range(num_blocks)])
        self.out = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x, orientation_info=None):
        del orientation_info  # shared fit/reconstruct API; unused for Res-CNN-2D
        x = self.embed(x)
        x = self.blocks(x)
        return self.out(x)
