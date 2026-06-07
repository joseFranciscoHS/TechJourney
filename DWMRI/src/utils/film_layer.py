import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Predicts per-channel scale (gamma) and shift (beta) from a conditioning
    vector and applies gamma * features + beta.

    Uses identity initialization so the layer starts as a no-op:
    gamma ~ 1.0, beta ~ 0.0.
    """

    def __init__(
        self, cond_dim: int, feature_channels: int, hidden_dim: int = 32
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels * 2),
        )
        self.feature_channels = feature_channels

        # Identity initialization: gamma=1, beta=0
        # Last linear layer output is [gamma_0..gamma_C-1, beta_0..beta_C-1]
        # Set weights to near-zero so output is dominated by bias
        nn.init.zeros_(self.mlp[2].weight)
        # Bias: first C values = 1.0 (gamma), last C = 0.0 (beta)
        with torch.no_grad():
            self.mlp[2].bias[:feature_channels].fill_(1.0)
            self.mlp[2].bias[feature_channels:].fill_(0.0)

    def forward(
        self, features: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, C, ...) feature tensor -- works for any spatial dims
                      e.g. (B, C, D, H, W) for 3D or (B, C, H, W) for 2D
            condition: (B, cond_dim) conditioning vector

        Returns:
            Modulated features with same shape as input.
        """
        # (B, 2*C)
        gamma_beta = self.mlp(condition)
        gamma = gamma_beta[:, :self.feature_channels]
        beta = gamma_beta[:, self.feature_channels:]

        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1, 1) for 5D
        n_spatial = features.dim() - 2
        shape = [features.size(0), self.feature_channels] + [1] * n_spatial
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        return gamma * features + beta
