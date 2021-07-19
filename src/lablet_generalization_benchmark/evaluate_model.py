# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np


class RSquared:
    def __init__(self, all_targets: np.ndarray):
        variance_per_factor = (
                (all_targets -
                 all_targets.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)
        self.variance_per_factor = variance_per_factor

    def __call__(self,
                 predictions: np.ndarray,
                 targets: np.ndarray) -> np.ndarray:
        assert predictions.shape == targets.shape
        mse_loss_per_factor = np.mean((predictions - targets) ** 2, axis=0)
        return 1 - mse_loss_per_factor / self.variance_per_factor


def evaluate_model(model_fn, dataloader):
    """ Returns the benchmark scores of a given model under a particular dataset

    Args:
        model_fn: a function of the model that has an array of images as input
        and returns the predicted targets
        dataloader: a dataset on which the model shall be evaluated
    Returns:
        scores (dict): a dict with the score for each metric
    """

    targets = []
    predictions = []
    for image_batch, target_batch in dataloader:
        image_batch, target_batch = image_batch.numpy(), target_batch.numpy()
        batch_prediction = model_fn(image_batch)
        targets.append(target_batch)
        predictions.append(batch_prediction)
    targets = np.vstack(targets)
    predictions = np.vstack(predictions)

    squared_diff = (targets - predictions) ** 2
    targets_in_0_1 = dataloader.dataset.normalized_targets
    r_squared = RSquared(targets_in_0_1)

    r_squared_per_factor = r_squared(predictions, targets)
    mse_per_factor = np.mean(squared_diff, axis=0)

    # book keeping
    scores = dict()
    scores['rsquared'] = np.mean(r_squared_per_factor)
    scores['mse'] = np.mean(squared_diff)

    factor_names = dataloader.dataset._factor_names
    for factor_index, factor_name in enumerate(factor_names):
        scores['rsquared_{}'.format(
            factor_name)] = r_squared_per_factor[factor_index]
        scores['mse_{}'.format(factor_name)] = mse_per_factor[factor_index]
    return scores
