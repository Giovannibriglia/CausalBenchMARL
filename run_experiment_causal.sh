#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#

# Activate the virtual environment
source MyEnv/bin/activate

# Set the display environment variable
export DISPLAY=':99.0'

# Start a virtual framebuffer in the background
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

# Install the package
python setup.py install

# Change directory to examples/running
# shellcheck disable=SC2164
cd examples/running

# Run the experiment using Python 3.10
python3.10 -m run_experiment_causal

