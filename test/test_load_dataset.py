# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

import lablet_generalization_benchmark.load_dataset as load_dataset


def test_load_dataset(path):
    batch_size = 10
    dataloader = load_dataset.load_dataset('dsprites',
                                           dataset_path=path,
                                           batch_size=batch_size)
    for images, targets in dataloader:
        break
    assert images.shape[0] == batch_size
    assert images.shape[0] == targets.shape[0]
    assert len(images.shape) == 4
    assert len(dataloader.dataset._factor_names) == \
           len(dataloader.dataset._factor_sizes)
    assert len(targets.shape) == 2

    datasets = ['dsprites', 'shapes3d', 'mpi3d']
    splits = ['random', 'interpolation', 'extrapolation']
    # Check whether all splits are disjoint and cover the whole grid.
    for dataset in datasets:
        for split in splits:
            dataloader_test = load_dataset.load_dataset(
                dataset_name=dataset,
                variant=split,
                mode='test',
                dataset_path=path,
            )

            dataloader_train = load_dataset.load_dataset(
                dataset_name=dataset,
                variant=split,
                mode='train',
                dataset_path=path,
            )

            targets_test = dataloader_test.dataset._dataset_targets
            targets_train = dataloader_train.dataset._dataset_targets
            test_mask = np.zeros(dataloader_test.dataset._factor_sizes,
                                 dtype=np.bool)
            inds = [targets_test[:, i] for i in range(targets_test.shape[1])]
            test_mask[tuple(inds)] = True

            train_mask = np.zeros(dataloader_train.dataset._factor_sizes,
                                  dtype=np.bool)
            inds = [targets_train[:, i] for i in range(targets_train.shape[1])]
            train_mask[tuple(inds)] = True
            assert np.all(test_mask ^ train_mask)  # disjoint, complete splits

            num_total = np.prod(dataloader_test.dataset._factor_sizes)
            num_test = dataloader_test.dataset._dataset_targets.shape[0]
            num_train = dataloader_train.dataset._dataset_targets.shape[0]

            # roughly a 70:30 split
            assert np.round(num_test / num_total, 1) == 0.7
            assert np.round(num_train / num_total, 1) == 0.3
