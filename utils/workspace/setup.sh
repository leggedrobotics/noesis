#!/usr/bin/env bash

#=============================================================================
# Copyright (C) 2023, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

# Configurations
WS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"
echo "[setup.sh]: Activating workspace: ${WS_DIR}"

# Workspace paths
export WORKSPACE_DIR=${WS_DIR}
export SOURCE_DIR=${WS_DIR}/src
export DATA_DIR=${WS_DIR}/data
export BIN_DIR=${WS_DIR}/bin
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${WS_DIR}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WS_DIR}/lib/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${BIN_DIR}/lib
export PATH=${PATH}:${BIN_DIR}/bin

# EOF
