import logging

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, p=0.3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channel, mid_channel, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            mid_channel, out_channel, kernel_size=3, padding=1
        )
        # self.nonlinear1 = nn.LeakyReLU(0.1)
        self.nonlinear1 = nn.ReLU()
        self.Dropout = nn.Dropout(p)
        self.BN1 = nn.BatchNorm2d(num_features=mid_channel)
        self.BN2 = nn.BatchNorm2d(num_features=out_channel)
        logging.debug(
            f"ResBlock initialized - in: {in_channel}, mid: {mid_channel}, out: {out_channel}, dropout: {p}"
        )

    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        out2 = self.nonlinear1(out1)
        out3 = self.BN2(self.conv2(self.Dropout(out2)))
        out = x + out3
        return out


class Self2self(nn.Module):
    def __init__(self, in_channel, out_channel, p):
        super(Self2self, self).__init__()

        self.n_blks = 10
        self.channel_size = 64
        logging.info(
            f"Self2self model initializing - in_channel: {in_channel}, out_channel: {out_channel}, dropout_p: {p}"
        )
        logging.info(
            f"Model architecture - n_blks: {self.n_blks}, channel_size: {self.channel_size}"
        )

        layers = [
            ResBlock(
                self.channel_size, self.channel_size, self.channel_size, p=p
            )
        ]
        for i in range(self.n_blks - 1):
            layers.append(
                ResBlock(
                    self.channel_size,
                    self.channel_size,
                    self.channel_size,
                    p=p,
                )
            )

        self.convs = nn.Sequential(*layers)
        self.conv_first = nn.Conv2d(
            in_channel, self.channel_size, kernel_size=3, padding=1
        )
        self.conv_last = nn.Conv2d(
            self.channel_size, out_channel, kernel_size=3, padding=1
        )
        self.conv_last2 = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logging.info(
            f"Self2self model created - Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}"
        )

    def forward(self, x):
        logging.debug(f"Self2self forward pass - input shape: {x.shape}")

        out1 = self.conv_first(x)
        out2 = self.convs(out1)
        out2 += out1
        out3 = self.conv_last(out2)
        out = self.conv_last2(out3)
        out = self.relu(out)

        logging.debug(f"Self2self forward pass - output shape: {out.shape}")
        return out
