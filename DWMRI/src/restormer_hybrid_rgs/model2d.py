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


class Restormer2D(nn.Module):
    """
    Reduced 2D Restormer for iso-budget pilot runs.
    Input: (B, K, H, W), Output: (B, 1, H, W)
    """

    def __init__(
        self,
        inp_channels=10,
        out_channels=1,
        dim=24,
        num_blocks=2,
        heads=2,
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        self.embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=bias)
        self.blocks = nn.Sequential(*[_Block2D(dim=dim) for _ in range(num_blocks)])
        self.out = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.out(x)
