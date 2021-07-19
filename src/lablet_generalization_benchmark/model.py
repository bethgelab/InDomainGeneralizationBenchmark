# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn


class VanillaCNN(nn.Module):
    def __init__(self, number_of_classes: int, number_of_channels: int):
        """
        Standard ConvNet Architecture similar to the Locatello Disentanglement
        Library models.
        Args:
            number_of_classes: number of classes in the dataset
            number_of_channels: number channels of the input image
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(number_of_channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, number_of_classes),  # B, number_of_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
