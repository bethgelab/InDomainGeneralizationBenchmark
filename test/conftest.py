# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest


def pytest_addoption(parser):
    parser.addoption("--path", default=None, type=str, action='store')


@pytest.fixture(scope='session')
def path(request):
    path = request.config.option.path
    if path is None:
        pytest.skip()
    return path
