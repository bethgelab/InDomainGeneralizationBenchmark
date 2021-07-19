#!/bin/bash

# Runs tests in on standard setup.
# Make sure to install pytest and pytest-cov via:
# pip install pytest pytest-cov

# To be safe, switch into the folder that contains this script.
cd "$( cd "$( dirname "$0" )" && pwd )"

env PYTHONPATH=./src python3 -m pytest -v
