#!/bin/bash

#=============================================================================
# Copyright (C) 2023, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
# https://bitbucket.org/leggedrobotics/noesis
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# Authors: Vassilios Tsounis, tsounisv@ethz.ch
#=============================================================================

# Set paths
BUILD_DIR=~/cmake_ws/cmake-build-release
APP_DIR=${BUILD_DIR}/src/noesis/examples/rl/raisim
APP_NAME=noesis_rl_train_kinova3_example

# Build experiment
cd ${BUILD_DIR}
make ${APP_NAME} -j

# Create data directories
DATA_DIR=~/.noesis/proc/${APP_NAME}
DATA_DIR=$(eval realpath -m "${DATA_DIR}")
mkdir -p ${DATA_DIR}

# Run a set of experiments over random seeds
for SEED in 0
do
  ${APP_DIR}/${APP_NAME} \
    --log_path=${DATA_DIR} \
    --batch_size=64 \
    --iterations=5000 \
    --time_step=0.01 \
    --time_limit=2.0 \
    --discount_factor=0.995 \
    --goal_noise_factor=1.0 \
    --reset_noise_factor=1.0 \
    --randomization_factor=1.0 \
    --observations_noise_factor=1.0 \
    --use_pid_controller=true \
    --use_simulator_pid=false \
    --seed=${SEED}
done


