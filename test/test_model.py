# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import lablet_generalization_benchmark.model as models


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("number_of_classes", [2, 10])
@pytest.mark.parametrize("number_of_channels", [1, 3])
def test_model(batch_size, number_of_classes, number_of_channels):
    model = models.VanillaCNN(
        number_of_classes,
        number_of_channels)
    out = model(torch.zeros(batch_size, number_of_channels, 64, 64))
    assert out.shape == (batch_size, number_of_classes)
