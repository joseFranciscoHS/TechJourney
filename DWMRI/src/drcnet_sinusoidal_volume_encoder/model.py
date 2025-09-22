# flake8: noqa: E501
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn


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
                        nn.Conv3d(
                            in_channels, inner_channels, kernel_size=(1, 1, 1)
                        ),
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
            h = torch.zeros(
                x.size(), dtype=x.dtype, layout=x.layout, device=x.device
            )

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
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
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
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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
    def __init__(self, num_volumes=10, embedding_dim=64, max_freq=10000.0):
        super(SinusoidalVolumeEncoder, self).__init__()
        logging.info(
            f"Initializing SinusoidalVolumeEncoder: num_volumes={num_volumes}, embedding_dim={embedding_dim}, max_freq={max_freq}"
        )
        
        self.num_volumes = num_volumes
        self.embedding_dim = embedding_dim
        
        # Create sinusoidal encodings for all possible volume positions
        pe = torch.zeros(num_volumes, embedding_dim)
        position = torch.arange(0, num_volumes).unsqueeze(1).float()
        
        # Calculate frequency terms using logarithmic spacing
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           -(math.log(max_freq) / embedding_dim))
        
        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Store as buffer (not trainable parameter)
        self.register_buffer('pe', pe)
        
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
        logging.debug(f"SinusoidalVolumeEncoder forward: volume_features.shape={volume_features.shape}, volume_indices.shape={volume_indices.shape}")
        
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
            # Add positional encoding to first embedding_dim channels
            encoded_features = volume_features.clone()
            encoded_features[:, :self.embedding_dim] += pos_encoding
        else:
            # If we have fewer channels than embedding_dim, we need to project
            # For simplicity, we'll just add to all channels
            encoded_features = volume_features + pos_encoding[:, :channels]
        
        logging.debug(f"SinusoidalVolumeEncoder forward: output.shape={encoded_features.shape}")
        return encoded_features


class DenoiserNet(nn.Module):
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
    ):
        super(DenoiserNet, self).__init__()
        logging.info(
            f"Initializing DenoiserNet: input_channels={input_channels}, output_channels={output_channels}, groups={groups}, dense_convs={dense_convs}, residual={residual}, base_filters={base_filters}, num_volumes={num_volumes}, use_sinusoidal_encoding={use_sinusoidal_encoding}"
        )
        groups = groups

        dense_convs = dense_convs
        self.residual = residual
        self.use_sinusoidal_encoding = use_sinusoidal_encoding
        self.num_volumes = num_volumes
        filters_0 = base_filters
        filters_1 = filters_0
        
        # Add sinusoidal volume encoder
        if self.use_sinusoidal_encoding:
            self.volume_encoder = SinusoidalVolumeEncoder(
                num_volumes=num_volumes,
                embedding_dim=embedding_dim
            )
            logging.info(f"Added SinusoidalVolumeEncoder with {num_volumes} volumes and {embedding_dim} embedding dimensions")

        self.input_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            input_channels,
                            filters_0,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act", nn.PReLU(filters_0)),
                ]
            )
        )
        
        # Add CBAM to input block for early volume-specific attention
        self.input_attention = CBAM3D(filters_0, reduction=8, kernel_size=5)

        self.down_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv3d(
                            filters_0,
                            filters_1,
                            kernel_size=(2, 2, 2),
                            stride=(2, 2, 2),
                        ),
                    ),
                    ("act", nn.PReLU(filters_1)),
                ]
            )
        )

        self.up_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.ConvTranspose3d(
                            filters_1,
                            filters_0,
                            kernel_size=(2, 2, 2),
                            stride=(2, 2, 2),
                        ),
                    ),
                    ("act", nn.PReLU(filters_0)),
                ]
            )
        )

        self.denoising_block = GatedBlock(
            filters_1, filters_1, dense_convs, groups
        )
        
        # Add CBAM attention for better feature focus and volume-specific adaptation
        self.attention = CBAM3D(filters_1, reduction=8, kernel_size=7)

        self.output_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv 0",
                        nn.Conv3d(
                            2 * filters_0, filters_0, kernel_size=(1, 1, 1)
                        ),
                    ),
                    ("act 0", nn.PReLU(filters_0)),
                    (
                        "conv 1",
                        nn.Conv3d(
                            filters_0,
                            output_channels,
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1),
                        ),
                    ),
                    ("act 1", nn.PReLU(output_channels)),
                ]
            )
        )

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logging.info(
            f"DenoiserNet model created - Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}"
        )

    def forward(self, inputs, volume_indices=None):
        """
        Forward pass with sinusoidal volume encoding.
        
        Args:
            inputs: [batch, num_input_volumes, x, y, z] - input volumes
            volume_indices: [batch, num_input_volumes] - volume indices for each input volume
            
        Returns:
            output: [batch, output_channels, x, y, z] - reconstructed volume
        """
        logging.debug(f"DenoiserNet forward: input shape={inputs.shape}")
        
        # Apply sinusoidal volume encoding if enabled
        if self.use_sinusoidal_encoding and volume_indices is not None:
            logging.debug(f"Applying sinusoidal encoding with volume_indices shape={volume_indices.shape}")
            
            # Process each volume individually with its positional encoding
            encoded_volumes = []
            batch_size, num_volumes, x, y, z = inputs.shape
            
            for vol_idx in range(num_volumes):
                # Get single volume: [batch, 1, x, y, z]
                single_volume = inputs[:, vol_idx:vol_idx+1]
                
                # Get volume indices for this volume: [batch]
                vol_indices = volume_indices[:, vol_idx]
                
                # Apply sinusoidal encoding
                encoded_vol = self.volume_encoder(single_volume, vol_indices)
                encoded_volumes.append(encoded_vol)
            
            # Combine encoded volumes
            inputs = torch.cat(encoded_volumes, dim=1)
            logging.debug(f"After sinusoidal encoding: inputs shape={inputs.shape}")
        
        # Continue with existing architecture
        # taking as base output image the mean of the inputs over volumes
        # i.e. mean of the X training volumes
        output_image = inputs.mean(dim=1, keepdim=True)
        up_0 = self.input_block(inputs)
        # Apply CBAM attention to input features for volume-specific adaptation
        up_0 = self.input_attention(up_0)
        x = self.down_block(up_0)

        # x 1
        h = None
        for i in range(4):
            h = self.denoising_block(x, h)
            x += h

        # Apply CBAM attention for volume-specific feature refinement
        x = self.attention(x)
        
        up_3 = self.up_block(x)

        return self.output_block(torch.cat([up_0, up_3], 1)) + output_image
