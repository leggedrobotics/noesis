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
#BUILD_DIR=${WORKSPACE_DIR}/cmake-build-release
BUILD_DIR=~/Development/deepgait/deepgait_cpp/cmake-build-release
APP_DIR=${BUILD_DIR}/src/noesis/examples/rl/raisim
APP_NAME=noesis_rl_train_capler_ppo_example

# Build experiment
cd ${BUILD_DIR}
make ${APP_NAME} -j

# Run experiment
#for GRAPH in "baseline" "shared-layer" "shared-net" "state-dep-stddev" "state-dep-stddev-shared-layer" "state-dep-stddev-shared-net"
for GRAPH in "baseline"
do
  # Create data directories
  DATA_DIR=~/.noesis/proc/${APP_NAME}/${GRAPH}
  DATA_DIR=$(eval realpath -m "${DATA_DIR}")
  mkdir -p ${DATA_DIR}
  # Run a set of experiments over random seeds
	for SEED in 1
  do
    ${APP_DIR}/${APP_NAME} \
      --log_path=${DATA_DIR} \
      --batch_size=16 \
      --iterations=300 \
      --goal_noise_factor=0.0 \
      --reset_noise_factor=1.0 \
      --state_noise_factor=1.0 \
      --randomization_factor=1.0 \
      --graph_file="graph-${GRAPH}.py" \
      --seed=${SEED}
  done
done

## Run experiment
#for TERMINAL_VALUE in 0.0 -1.0 -2.0 -3.0 -5.0 1.0
#do
#  # Create data directories
#  DATA_DIR=~/.noesis/proc/${APP_NAME}/"tv_${TERMINAL_VALUE}"
#  DATA_DIR=$(eval realpath -m "${DATA_DIR}")
#  mkdir -p ${DATA_DIR}
#  # Run a set of experiments over random seeds
#	for SEED in 1 2 3
#  do
#    ${APP_DIR}/${APP_NAME} \
#      --log_path=${DATA_DIR} \
#      --batch_size=16 \
#      --iterations=300 \
#      --goal_noise_factor=0.0 \
#      --reset_noise_factor=1.0 \
#      --state_noise_factor=1.0 \
#      --randomization_factor=1.0 \
#      --terminal_value="${TERMINAL_VALUE}" \
#      --seed=${SEED}
#  done
#done


