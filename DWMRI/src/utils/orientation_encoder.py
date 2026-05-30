import torch
import torch.nn as nn
import torch.nn.functional as F


class OrientationEncoder(nn.Module):
    """Map gradient metadata to an additive spatial input encoding."""

    def __init__(self, embed_dim: int = 1024, spatial_size: int = 32):
        super().__init__()
        if embed_dim != spatial_size * spatial_size:
            raise ValueError(
                "embed_dim "
                f"({embed_dim}) must equal spatial_size^2 ({spatial_size**2})"
            )

        self.spatial_size = spatial_size
        self.linear = nn.Linear(4, embed_dim)

    def forward(
        self,
        orientation_info: torch.Tensor,
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            orientation_info: Tensor shaped (B, K, 4), with
                [cos_x, cos_y, cos_z, b_norm] per input volume.
            target_shape: Spatial tensor shape (D, H, W).

        Returns:
            Tensor shaped (B, K, D, H, W), ready to add to the input.
        """
        batch_size, num_channels, _ = orientation_info.shape
        depth, height, width = target_shape
        spatial_size = self.spatial_size

        embedding = F.relu(self.linear(orientation_info))
        pattern_2d = embedding.view(
            batch_size * num_channels, 1, spatial_size, spatial_size
        )
        pattern_xy = F.interpolate(
            pattern_2d,
            size=(depth, height),
            mode="bilinear",
            align_corners=False,
        )
        pattern_xy = pattern_xy.view(batch_size, num_channels, depth, height)

        return pattern_xy.unsqueeze(-1).expand(
            batch_size, num_channels, depth, height, width
        )
