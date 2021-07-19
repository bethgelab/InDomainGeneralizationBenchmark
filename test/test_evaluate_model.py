# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch

import lablet_generalization_benchmark.evaluate_model as evaluate_model


def test_evaluate_model():
    number_of_images = 10
    images = np.zeros((number_of_images, 1, number_of_images, number_of_images),
                      dtype=np.float32)
    for i in range(number_of_images):
        images[i, :, :, i] = i
    targets = np.arange(number_of_images, dtype=np.float32)

    def model_fn(images):
        return np.max(images.reshape(images.shape[0], -1),
                      axis=1) / number_of_images

    class Dataset:
        def __init__(self, images, targets):
            self.images = torch.tensor(images)
            self.targets = torch.tensor(targets)
            self._factor_names = [str(i) for i in range(len(images))]

        @property
        def normalized_targets(self):
            return (self.targets / number_of_images).numpy()

    class DataLoader():
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            for i in range(2):
                yield self.dataset.images, self.dataset.targets / 10.

    scores = evaluate_model.evaluate_model(model_fn,
                                           dataloader=DataLoader(
                                               Dataset(images, targets)))

    for key in scores.keys():
        assert 'mse' in key or 'rsquared' in key
        if 'mse' in key:
            assert scores[key] == 0
        elif 'rsquared' in key:
            assert scores[key] == 1
        else:
            raise Exception('only mse and rsquared should be implemented')


def test_rsquared():
    # Test score with optimal solution.
    targets = np.arange(10) / 10.
    rsquared = evaluate_model.RSquared(targets)
    predictions = np.arange(10) / 10.
    assert rsquared(targets, predictions) == 1

    # Test score with solution predicting the mean.
    predictions = np.empty_like(targets)
    predictions.fill(np.mean(targets))
    assert rsquared(targets, predictions) == 0

    # Test score with solution predicting zero.
    predictions = np.zeros_like(targets)
    assert rsquared(targets, predictions) < 0
