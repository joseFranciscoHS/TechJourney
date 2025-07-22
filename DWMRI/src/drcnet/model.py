# flake8: noqa: E501
import logging
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
    ):
        super(DenoiserNet, self).__init__()
        logging.info(
            f"Initializing DenoiserNet: input_channels={input_channels}, output_channels={output_channels}, groups={groups}, dense_convs={dense_convs}, residual={residual}, base_filters={base_filters}"
        )
        groups = groups

        dense_convs = dense_convs
        self.residual = residual
        filters_0 = base_filters
        filters_1 = filters_0

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

        self.output_image = torch.rand(output_shape, dtype=torch.float)

    def forward(self, inputs):
        logging.debug(f"DenoiserNet forward: input shape={inputs.shape}")
        up_0 = self.input_block(inputs)
        x = self.down_block(up_0)

        # x 1
        h = None
        for i in range(4):
            h = self.denoising_block(x, h)
            x += h

        up_3 = self.up_block(x)

        return (
            self.output_block(torch.cat([up_0, up_3], 1)) + self.output_image
        )
