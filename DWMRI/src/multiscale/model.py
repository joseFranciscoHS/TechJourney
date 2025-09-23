# flake8: noqa: E501
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(FactorizedBlock, self).__init__()
        logging.info(
            f"Initializing FactorizedBlock: in_channels={in_channels}, out_channels={out_channels}, groups={groups}"
        )

        self.conv_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            in_channels,
                            out_channels,
                            kernel_size=(3, 1, 1),
                            padding=(1, 0, 0),
                            groups=groups,
                        ),
                    ),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

        self.conv_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            in_channels,
                            out_channels,
                            kernel_size=(1, 3, 1),
                            padding=(0, 1, 0),
                            groups=groups,
                        ),
                    ),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

        self.conv_3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            in_channels,
                            out_channels,
                            kernel_size=(1, 1, 3),
                            padding=(0, 0, 1),
                            groups=groups,
                        ),
                    ),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

        self.conv_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            3 * out_channels,
                            out_channels,
                            kernel_size=(1, 1, 1),
                            groups=groups,
                        ),
                    ),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

    def forward(self, x):
        logging.debug(f"FactorizedBlock forward: input shape={x.shape}")
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        x_3 = self.conv_3(x)

        x = torch.cat([x_1, x_2, x_3], 1)
        x = self.conv_4(x)
        logging.debug(f"FactorizedBlock forward: output shape={x.shape}")
        return x


class DenoisingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        out_channels,
        inner_convolutions=1,
        residual=False,
        groups=16,
    ):
        super(DenoisingBlock, self).__init__()
        logging.info(
            f"Initializing DenoisingBlock: in_channels={in_channels}, inner_channels={inner_channels}, out_channels={out_channels}, inner_convolutions={inner_convolutions}, residual={residual}"
        )

        self.inner_convolutions = inner_convolutions
        self.residual = residual

        self.input_convolution = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(in_channels, inner_channels, kernel_size=(1, 1, 1)),
                    ),
                    ("act", nn.PReLU(inner_channels)),
                ]
            )
        )

        self.dense_convolutions = nn.ModuleList()
        for i in range(1, inner_convolutions + 1):
            dense_convolution = FactorizedBlock(
                in_channels + i * inner_channels, inner_channels, groups
            )
            self.dense_convolutions.append(dense_convolution)

        self.output_convolution = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            in_channels=in_channels
                            + (inner_convolutions + 1) * inner_channels,
                            out_channels=out_channels,
                            kernel_size=(1, 1, 1),
                        ),
                    ),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

    def forward(self, x):
        logging.debug(f"DenoisingBlock forward: input shape={x.shape}")
        output = self.input_convolution(x)
        output = torch.cat([x, output], 1)

        for i in range(self.inner_convolutions):
            inner_output = self.dense_convolutions[i](output)
            output = torch.cat([output, inner_output], 1)

        output = self.output_convolution(output)

        if self.residual:
            output += x

        logging.debug(f"DenoisingBlock forward: output shape={x.shape}")
        return output


class InputBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(InputBlock, self).__init__()
        logging.info(
            f"Initializing InputBlock: in_channels={in_channels}, inner_channels={inner_channels}, out_channels={out_channels}"
        )
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels

        self.input_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv 0",
                        nn.Conv3d(
                            in_channels,
                            inner_channels,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act 0", nn.PReLU(inner_channels)),
                    (
                        "conv 1",
                        nn.Conv3d(
                            in_channels,
                            inner_channels,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act 1", nn.PReLU(inner_channels)),
                ]
            )
        )

    def forward(self, inputs):
        logging.debug(f"InputBlock forward: input shape={inputs.shape}")
        inputs = self.input_conv(inputs)  # 1 x M x N
        down = self.down_conv(inputs)  # K x M/2 x N/2

        logging.debug(f"InputBlock forward: output shape={inputs.shape}")
        return down, inputs


class OutputBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(OutputBlock, self).__init__()
        logging.info(
            f"Initializing OutputBlock: in_channels={in_channels}, inner_channels={inner_channels}, out_channels={out_channels}"
        )
        self.conv_t = nn.ConvTranspose3d(
            in_channels,
            inner_channels,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
        )
        self.act_t = nn.PReLU(inner_channels)

        self.output_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            inner_channels,
                            out_channels,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act", nn.PReLU(out_channels)),
                ]
            )
        )

    def forward(self, x, residual):
        logging.debug(
            f"OutputBlock forward: input x shape={x.shape}, residual shape={residual.shape}"
        )
        x = self.act_t(self.conv_t(x)) + residual
        return self.output_conv(x)


class GatedBlock(nn.Module):
    def __init__(self, x_channels, h_channels, dense_convs=1, groups=1):
        super(GatedBlock, self).__init__()
        logging.info(
            f"Initializing GatedBlock: x_channels={x_channels}, h_channels={h_channels}, dense_convs={dense_convs}, groups={groups}"
        )

        self.z_t = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            h_channels + x_channels,
                            h_channels,
                            kernel_size=(1, 1, 1),
                        ),
                    ),
                    ("act", nn.Sigmoid()),
                ]
            )
        )

        self.r_t = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            h_channels + x_channels,
                            h_channels,
                            kernel_size=(1, 1, 1),
                        ),
                    ),
                    ("act", nn.Sigmoid()),
                ]
            )
        )

        self.h_t = DenoisingBlock(
            h_channels + x_channels,
            h_channels,
            h_channels,
            dense_convs,
            False,
            groups,
        )

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(), dtype=x.dtype, layout=x.layout, device=x.device)

        concat = torch.cat([h, x], dim=1)
        z_t = self.z_t(concat)
        r_t = self.r_t(concat)
        concat = torch.cat([r_t * h, x], dim=1)
        h_t = self.h_t(concat)

        h_t = (1 - z_t) * h + z_t * h_t

        logging.debug(
            f"GatedBlock forward: h_t shape={h_t.shape}, z_t shape={z_t.shape}, r_t shape={r_t.shape}"
        )
        return h_t


class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module for DWMRI"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module for DWMRI"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module for DWMRI"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        # Channel attention first
        x = x * self.channel_attention(x)
        # Then spatial attention
        x = x * self.spatial_attention(x)
        return x


class SpatialAttention(nn.Module):
    """Spatial attention module for better feature focus"""

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class SinusoidalVolumeEncoder(nn.Module):
    """
    Sinusoidal positional encoding for DWMRI volumes.
    Creates unique positional encodings for each volume position using sine and cosine waves.
    """

    def __init__(
        self, num_volumes=10, embedding_dim=64, max_freq=10000.0, encoding_scale=0.1
    ):
        super(SinusoidalVolumeEncoder, self).__init__()
        logging.info(
            f"Initializing SinusoidalVolumeEncoder: num_volumes={num_volumes}, embedding_dim={embedding_dim}, max_freq={max_freq}, encoding_scale={encoding_scale}"
        )

        self.num_volumes = num_volumes
        self.embedding_dim = embedding_dim
        self.encoding_scale = encoding_scale

        # Create sinusoidal encodings for all possible volume positions
        pe = torch.zeros(num_volumes, embedding_dim)
        position = torch.arange(0, num_volumes).unsqueeze(1).float()

        # Calculate frequency terms using logarithmic spacing
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(math.log(max_freq) / embedding_dim)
        )

        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as buffer (not trainable parameter)
        self.register_buffer("pe", pe)

        logging.info(f"SinusoidalVolumeEncoder created with shape: {pe.shape}")
        logging.info(f"Volume 0 encoding sample: {pe[0, :8].tolist()}")
        logging.info(f"Volume 1 encoding sample: {pe[1, :8].tolist()}")

    def forward(self, volume_features, volume_indices):
        """
        Apply sinusoidal positional encoding to volume features.

        Args:
            volume_features: [batch, channels, x, y, z] - single volume features
            volume_indices: [batch] - which volume each sample represents

        Returns:
            encoded_features: [batch, channels, x, y, z] - features with positional encoding
        """
        logging.debug(
            f"SinusoidalVolumeEncoder forward: volume_features.shape={volume_features.shape}, volume_indices.shape={volume_indices.shape}"
        )

        # Get positional encoding for each volume in the batch
        pos_encoding = self.pe[volume_indices]  # [batch, embedding_dim]

        # Reshape to match spatial dimensions
        # pos_encoding: [batch, embedding_dim] -> [batch, embedding_dim, 1, 1, 1]
        pos_encoding = pos_encoding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Expand to match spatial dimensions
        pos_encoding = pos_encoding.expand(-1, -1, *volume_features.shape[2:])

        # Project positional encoding to match feature channels
        # For now, we'll add the encoding to the first embedding_dim channels
        # and pad with zeros for remaining channels
        batch_size, channels, x, y, z = volume_features.shape

        if channels >= self.embedding_dim:
            # Add scaled positional encoding to first embedding_dim channels
            # Scale encoding to be smaller to avoid breaking [0,1] range
            scaled_encoding = pos_encoding * self.encoding_scale
            encoded_features = volume_features.clone()
            encoded_features[:, : self.embedding_dim] += scaled_encoding
        else:
            # If we have fewer channels than embedding_dim, we need to project
            # Scale encoding to be smaller to avoid breaking [0,1] range
            scaled_encoding = pos_encoding[:, :channels] * self.encoding_scale
            encoded_features = volume_features + scaled_encoding

        logging.debug(
            f"SinusoidalVolumeEncoder forward: output.shape={encoded_features.shape}"
        )
        return encoded_features


class MultiScaleDetailNet(nn.Module):
    """
    Multi-scale detail preservation network for DWMRI reconstruction.
    Uses parallel processing paths to preserve fine details while maintaining global context.
    """

    def __init__(
        self,
        input_channels,
        output_channels=1,
        groups=1,
        dense_convs=2,
        residual=True,
        base_filters=32,
        output_shape=(1, 128, 128, 128),
        device="cpu",
        num_volumes=10,
        use_sinusoidal_encoding=True,
        embedding_dim=64,
        encoding_scale=0.1,
    ):
        super(MultiScaleDetailNet, self).__init__()
        logging.info(
            f"Initializing MultiScaleDetailNet: input_channels={input_channels}, output_channels={output_channels}, groups={groups}, dense_convs={dense_convs}, residual={residual}, base_filters={base_filters}, num_volumes={num_volumes}, use_sinusoidal_encoding={use_sinusoidal_encoding}"
        )

        self.residual = residual
        self.use_sinusoidal_encoding = use_sinusoidal_encoding
        self.num_volumes = num_volumes
        filters_0 = base_filters
        filters_1 = base_filters

        # Add sinusoidal volume encoder
        if self.use_sinusoidal_encoding:
            self.volume_encoder = SinusoidalVolumeEncoder(
                num_volumes=num_volumes,
                embedding_dim=embedding_dim,
                encoding_scale=encoding_scale,
            )
            logging.info(
                f"Added SinusoidalVolumeEncoder with {num_volumes} volumes and {embedding_dim} embedding dimensions"
            )

        # Multi-scale architecture
        # Full resolution path (preserves fine details)
        self.full_res_path = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv3d(
                            input_channels,
                            filters_0 // 2,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act1", nn.PReLU(filters_0 // 2)),
                    (
                        "conv2",
                        nn.Conv3d(
                            filters_0 // 2,
                            filters_0 // 2,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act2", nn.PReLU(filters_0 // 2)),
                ]
            )
        )

        # Half resolution path (global context)
        self.half_res_path = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv3d(
                            input_channels,
                            filters_0,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act1", nn.PReLU(filters_0)),
                    (
                        "down",
                        nn.Conv3d(
                            filters_0,
                            filters_0,
                            kernel_size=(2, 2, 2),
                            stride=(2, 2, 2),
                        ),
                    ),
                    ("act2", nn.PReLU(filters_0)),
                    # Process at half resolution
                    (
                        "conv2",
                        nn.Conv3d(
                            filters_0,
                            filters_0,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act3", nn.PReLU(filters_0)),
                    (
                        "conv3",
                        nn.Conv3d(
                            filters_0,
                            filters_0,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act4", nn.PReLU(filters_0)),
                    # Upsample back to full resolution
                    (
                        "up",
                        nn.ConvTranspose3d(
                            filters_0,
                            filters_0,
                            kernel_size=(2, 2, 2),
                            stride=(2, 2, 2),
                        ),
                    ),
                    ("act5", nn.PReLU(filters_0)),
                ]
            )
        )

        # Detail enhancement block (processes combined features)
        self.detail_enhancement = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv3d(
                            filters_0 // 2 + filters_0,
                            filters_1,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act1", nn.PReLU(filters_1)),
                    (
                        "conv2",
                        nn.Conv3d(
                            filters_1,
                            filters_1,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act2", nn.PReLU(filters_1)),
                ]
            )
        )

        # Gated denoising block for feature refinement
        self.denoising_block = GatedBlock(filters_1, filters_1, dense_convs, groups)

        # CBAM attention for volume-specific adaptation
        self.attention = CBAM3D(filters_1, reduction=8, kernel_size=7)

        # Final output block with skip connection
        self.output_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv3d(
                            filters_1 + filters_0 // 2, filters_0, kernel_size=(1, 1, 1)
                        ),
                    ),
                    ("act1", nn.PReLU(filters_0)),
                    (
                        "conv2",
                        nn.Conv3d(
                            filters_0,
                            output_channels,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act2", nn.PReLU(output_channels)),
                ]
            )
        )

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            f"MultiScaleDetailNet model created - Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}"
        )

    def forward(self, inputs, volume_indices=None):
        """
        Forward pass with multi-scale detail preservation and sinusoidal volume encoding.

        Args:
            inputs: [batch, num_input_volumes, x, y, z] - input volumes
            volume_indices: [batch, num_input_volumes] - volume indices for each input volume

        Returns:
            output: [batch, output_channels, x, y, z] - reconstructed volume
        """
        logging.debug(f"MultiScaleDetailNet forward: input shape={inputs.shape}")

        # Apply sinusoidal volume encoding if enabled
        if self.use_sinusoidal_encoding and volume_indices is not None:
            logging.debug(
                f"Applying sinusoidal encoding with volume_indices shape={volume_indices.shape}"
            )

            # Process each volume individually with its positional encoding
            encoded_volumes = []
            batch_size, num_volumes, x, y, z = inputs.shape

            for vol_idx in range(num_volumes):
                # Get single volume: [batch, 1, x, y, z]
                single_volume = inputs[:, vol_idx : vol_idx + 1]

                # Get volume indices for this volume: [batch]
                vol_indices = volume_indices[:, vol_idx]

                # Apply sinusoidal encoding
                encoded_vol = self.volume_encoder(single_volume, vol_indices)
                encoded_volumes.append(encoded_vol)

            # Combine encoded volumes
            inputs = torch.cat(encoded_volumes, dim=1)
            logging.debug(f"After sinusoidal encoding: inputs shape={inputs.shape}")

        # Multi-scale processing
        # Full resolution path (preserves fine details)
        full_res_features = self.full_res_path(inputs)
        logging.debug(f"Full resolution features shape: {full_res_features.shape}")

        # Half resolution path (global context)
        half_res_features = self.half_res_path(inputs)
        logging.debug(f"Half resolution features shape: {half_res_features.shape}")

        # Combine multi-scale features
        combined_features = torch.cat([full_res_features, half_res_features], dim=1)
        logging.debug(f"Combined features shape: {combined_features.shape}")

        # Detail enhancement
        enhanced_features = self.detail_enhancement(combined_features)
        logging.debug(f"Enhanced features shape: {enhanced_features.shape}")

        # Gated denoising for feature refinement
        h = None
        for i in range(2):  # Reduced iterations for speed
            h = self.denoising_block(enhanced_features, h)
            enhanced_features += h

        # Apply CBAM attention for volume-specific adaptation
        enhanced_features = self.attention(enhanced_features)

        # Final output with skip connection from full resolution path
        final_input = torch.cat([enhanced_features, full_res_features], dim=1)
        output = self.output_block(final_input)

        # Add residual connection if enabled
        if self.residual:
            # Use volume-weighted average as residual
            if self.use_sinusoidal_encoding and volume_indices is not None:
                batch_size, num_volumes, x, y, z = inputs.shape
                volume_weights = torch.ones_like(inputs[:, :1])  # [batch, 1, x, y, z]

                # Apply sinusoidal encoding to create volume-specific weights
                for vol_idx in range(num_volumes):
                    vol_indices = volume_indices[:, vol_idx]
                    pos_encoding = self.volume_encoder.pe[
                        vol_indices
                    ]  # [batch, embedding_dim]

                    # Create weight from positional encoding (use first few dimensions)
                    weight = torch.sigmoid(pos_encoding[:, :4].mean(dim=1))  # [batch]
                    weight = weight.view(batch_size, 1, 1, 1, 1).expand(-1, 1, x, y, z)
                    volume_weights += weight * inputs[:, vol_idx : vol_idx + 1]

                # Normalize weights
                residual = volume_weights / (num_volumes + 1)
            else:
                # Fallback to simple mean
                residual = inputs.mean(dim=1, keepdim=True)

            output = output + residual

        logging.debug(f"MultiScaleDetailNet forward: output shape={output.shape}")
        return output


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss function for preserving fine details and sharp boundaries in DWMRI reconstruction.
    Combines MSE loss with edge preservation loss.
    """

    def __init__(self, alpha=0.5, beta=0.1):
        super(EdgeAwareLoss, self).__init__()
        self.alpha = alpha  # Weight for edge loss
        self.beta = beta  # Weight for gradient loss

        # Sobel edge detection kernels for 3D
        # Create proper 3D Sobel kernels
        sobel_x_2d = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y_2d = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        # Create 3D kernels by stacking 2D kernels
        sobel_x_3d = sobel_x_2d.unsqueeze(0).repeat(3, 1, 1)  # [3, 3, 3]
        sobel_y_3d = sobel_y_2d.unsqueeze(0).repeat(3, 1, 1)  # [3, 3, 3]
        sobel_z_3d = sobel_y_2d.unsqueeze(0).repeat(3, 1, 1)  # [3, 3, 3]

        self.register_buffer("sobel_x", sobel_x_3d.view(1, 1, 3, 3, 3))
        self.register_buffer("sobel_y", sobel_y_3d.view(1, 1, 3, 3, 3))
        self.register_buffer("sobel_z", sobel_z_3d.view(1, 1, 3, 3, 3))

        logging.info(f"EdgeAwareLoss initialized with alpha={alpha}, beta={beta}")

    def detect_edges(self, x):
        """Detect edges in 3D volume using Sobel operators"""
        # Ensure input has correct number of channels
        if x.dim() == 4:  # [batch, channels, x, y, z]
            x = x.mean(dim=1, keepdim=True)  # Average across channels

        # Convert Sobel kernels to match input dtype for mixed precision compatibility
        sobel_x = self.sobel_x.to(dtype=x.dtype)
        sobel_y = self.sobel_y.to(dtype=x.dtype)
        sobel_z = self.sobel_z.to(dtype=x.dtype)

        edges_x = F.conv3d(x, sobel_x, padding=1)
        edges_y = F.conv3d(x, sobel_y, padding=1)
        edges_z = F.conv3d(x, sobel_z, padding=1)

        # Compute magnitude of gradient
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2 + edges_z**2 + 1e-8)
        return edge_magnitude

    def gradient_loss(self, pred, target):
        """Compute gradient-based loss for edge preservation"""
        pred_grad_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_z = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        target_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        target_grad_y = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_z = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])

        grad_loss = (
            F.mse_loss(pred_grad_x, target_grad_x)
            + F.mse_loss(pred_grad_y, target_grad_y)
            + F.mse_loss(pred_grad_z, target_grad_z)
        )

        return grad_loss

    def forward(self, pred, target):
        """
        Compute edge-aware loss

        Args:
            pred: Predicted volume [batch, channels, x, y, z]
            target: Target volume [batch, channels, x, y, z]

        Returns:
            loss: Combined loss value
        """
        # Base MSE loss
        mse_loss = F.mse_loss(pred, target)

        # Edge preservation loss
        pred_edges = self.detect_edges(pred)
        target_edges = self.detect_edges(target)
        edge_loss = F.mse_loss(pred_edges, target_edges)

        # Gradient loss for better edge preservation
        grad_loss = self.gradient_loss(pred, target)

        # Combined loss
        total_loss = mse_loss + self.alpha * edge_loss + self.beta * grad_loss

        logging.debug(
            f"Loss components - MSE: {mse_loss:.6f}, Edge: {edge_loss:.6f}, Grad: {grad_loss:.6f}, Total: {total_loss:.6f}"
        )

        return total_loss
