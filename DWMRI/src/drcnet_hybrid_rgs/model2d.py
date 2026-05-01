import torch.nn as nn


class DenoiserNet2D(nn.Module):
    """
    Lightweight 2D counterpart for slice-wise ablations.
    Input: (B, K, H, W), Output: (B, 1, H, W)
    """

    def __init__(
        self, input_channels: int, base_filters: int = 32, output_channels: int = 1
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.PReLU(base_filters),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.PReLU(base_filters),
            nn.Conv2d(base_filters, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.body(x)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
