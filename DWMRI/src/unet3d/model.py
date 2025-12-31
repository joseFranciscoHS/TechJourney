"""
Denoising model based on the classical U-Net architecture with 3D convolutions.
Expected input shape: (B, C, X, Y, Z)
Ouput shape: (B, 1, X, Y, Z)
Where the output corresponds to the denoised target volum.
"""

# flake8: noqa: E501
import logging
from collections import OrderedDict

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Double 3D convolution block with batch normalization and activation."""

    def __init__(self, in_channels, out_channels, groups=1, residual=False):
        super(DoubleConv3D, self).__init__()
        self.residual = residual and (in_channels == out_channels)

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            groups=groups,
                        ),
                    ),
                    ("bn", nn.BatchNorm3d(out_channels)),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            out_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            groups=groups,
                        ),
                    ),
                    ("bn", nn.BatchNorm3d(out_channels)),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual:
            out = out + x
        return out


class DownBlock3D(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""

    def __init__(self, in_channels, out_channels, groups=1, residual=False):
        super(DownBlock3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels, groups, residual)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock3D(nn.Module):
    """Upsampling block: TransposeConv + Concatenate + DoubleConv."""

    def __init__(self, in_channels, out_channels, groups=1, residual=False):
        super(UpBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation: skip (out_channels) + upsampled (out_channels) = 2 * out_channels
        self.conv = DoubleConv3D(2 * out_channels, out_channels, groups, residual)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch due to odd dimensions
        diff = [skip.size(i) - x.size(i) for i in range(2, 5)]
        if any(d != 0 for d in diff):
            x = nn.functional.pad(
                x,
                (
                    0,
                    diff[2],
                    0,
                    diff[1],
                    0,
                    diff[0],
                ),
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Unet3D(nn.Module):
    """
    3D U-Net for DWMRI denoising.

    Architecture:
    - Encoder path: Multiple downsampling blocks
    - Bottleneck: Deepest feature representation
    - Decoder path: Upsampling blocks with skip connections
    - Output: Single channel denoised volume

    Args:
        in_channel: Number of input channels (volumes)
        out_channel: Number of output channels (default: 1)
        base_filters: Base number of filters (default: 32)
        groups: Number of groups for grouped convolutions (default: 1)
        residual: Whether to use residual connections (default: False)
        depth: Depth of the U-Net (number of down/up blocks) (default: 4)
    """

    def __init__(
        self,
        in_channel=9,
        out_channel=1,
        base_filters=32,
        groups=1,
        residual=False,
        depth=4,
    ):
        super(Unet3D, self).__init__()
        logging.info(
            f"Initializing Unet3D: in_channel={in_channel}, out_channel={out_channel}, "
            f"base_filters={base_filters}, groups={groups}, residual={residual}, depth={depth}"
        )

        self.depth = depth
        self.residual = residual

        # Input block (encoder entry)
        self.input_conv = DoubleConv3D(in_channel, base_filters, groups, residual)

        # Encoder path (downsampling)
        self.down_blocks = nn.ModuleList()
        filters = base_filters
        for i in range(depth):
            next_filters = filters * 2
            self.down_blocks.append(
                DownBlock3D(filters, next_filters, groups, residual)
            )
            filters = next_filters

        # Bottleneck
        self.bottleneck = DoubleConv3D(filters, filters * 2, groups, residual)
        filters = filters * 2

        # Decoder path (upsampling)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            next_filters = filters // 2
            # Skip connection will have next_filters channels (from corresponding encoder level)
            # After upsampling: next_filters, after concat with skip: 2 * next_filters
            self.up_blocks.append(UpBlock3D(filters, next_filters, groups, residual))
            filters = next_filters

        # Output block - handles final skip connection from input_conv
        # After last up block: base_filters * 2 channels (since we start from base_filters and go through depth up blocks)
        # After concatenating with input_conv skip (base_filters): base_filters * 3 channels
        # Actually, let's recalculate: after depth up blocks from bottleneck, we get base_filters * 2 channels
        # Skip from input_conv: base_filters channels
        # Total: base_filters * 2 + base_filters = base_filters * 3
        self.output_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            base_filters
                            * 3,  # Last up output (base_filters*2) + input_conv skip (base_filters)
                            base_filters,
                            kernel_size=3,
                            padding=1,
                            groups=groups,
                        ),
                    ),
                    ("bn", nn.BatchNorm3d(base_filters)),
                    ("act", nn.PReLU(base_filters)),
                    (
                        "conv_out",
                        nn.Conv3d(
                            base_filters,
                            out_channel,
                            kernel_size=1,
                        ),
                    ),
                ]
            )
        )

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            f"Unet3D model created - Total parameters: {total_params:,}, "
            f"Trainable parameters: {trainable_params:,}"
        )

    def forward(self, x):
        """
        Forward pass through the 3D U-Net.

        Args:
            x: Input tensor of shape (B, C, X, Y, Z)

        Returns:
            Output tensor of shape (B, 1, X, Y, Z)
        """
        logging.debug(f"Unet3D forward: input shape={x.shape}")

        # Encoder path
        skip_connections = []
        x = self.input_conv(x)
        skip_connections.append(x)

        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        # Skip connections are in order: [input_conv, down1, down2, ..., down_depth]
        # We use them in reverse order, starting from the deepest
        for up_block in self.up_blocks:
            skip = skip_connections.pop()  # Get skip from corresponding encoder level
            x = up_block(x, skip)

        # Final skip connection (from input_conv) - concatenate at output level
        final_skip = skip_connections.pop()
        # Handle potential size mismatch due to odd dimensions
        if x.shape[2:] != final_skip.shape[2:]:
            diff = [final_skip.size(i) - x.size(i) for i in range(2, 5)]
            if any(d != 0 for d in diff):
                x = nn.functional.pad(
                    x,
                    (
                        0,
                        diff[2],
                        0,
                        diff[1],
                        0,
                        diff[0],
                    ),
                )
        x = torch.cat([final_skip, x], dim=1)

        # Output
        x = self.output_conv(x)

        logging.debug(f"Unet3D forward: output shape={x.shape}")
        return x
