#=============================================================================
# Copyright (C) 2020, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
cmake_minimum_required(VERSION 3.10)
project(deepgait_ws)

# Include our project helper utility
include(ProjectHelper.cmake)

# Initialize the CMake Project
project_init()

# Build Options
project_enable_option(TF_USE_GPU)
project_enable_option(TF_USE_IN_SOURCE)
#project_enable_option(Noesis_USE_RAISIM)
#project_enable_option(Noesis_USE_CXX17)
#project_enable_option(Noesis_USE_SIMD)
#project_enable_option(Noesis_USE_TCMALLOC)
#project_enable_option(Noesis_USE_SANITIZERS)
project_enable_option(Noesis_BUILD_TESTS)

## TensorFlow
project_add_module(src/tensorflow-cpp/tensorflow)
project_add_module(src/tensorflow-cpp/examples)

## MuJoCo
if(Noesis_USE_MUJOCO)
  project_add_module(src/mujoco-cpp/mujoco)
  project_add_module(src/mujoco-cpp/mujoco_cpp)
endif()

## raisim
if(Noesis_USE_RAISIM)
  project_add_module(src/raisim/raisimLib)
  project_add_module(src/raisim/raisimOgre)
endif()

## Noesis
project_add_module(src/noesis/noesis)
project_add_module(src/noesis/examples)

# THIS STEP IS NECESSARY, DO NOT REMOVE
project_build()

# EOF
