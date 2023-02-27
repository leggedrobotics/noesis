#=============================================================================
# Copyright (C) 2020, Robotic Systems Lab, ETH Zurich
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
cmake_minimum_required(VERSION 3.10)
project(noesis_examples VERSION 0.2.0 LANGUAGES CXX)

# Include GNU standard install directories module
include(GNUInstallDirs)

#==
# Options
#==

option(Noesis_USE_MUJOCO "Enables support for the MuJoCo physics engine." OFF)
option(Noesis_USE_RAISIM "Enables support for the RaiSim physics engine." OFF)

#==
# Dependencies
#==

# Noesis: C++ library for robotic AI powered by TensorFlow
find_package(noesis CONFIG REQUIRED)

#==
# Build configuration
#==

# Helper macro for adding target applications
macro(add_noesis_example_app APP_NAME APP_SRC)
  add_executable(${APP_NAME} ${APP_SRC})
  target_link_libraries(${APP_NAME} PUBLIC noesis::noesis)
  list(APPEND NOESIS_EXAMPLE_APPS ${APP_NAME})
endmacro()

#==
# Components
#==

# Examples for how to use the core elements of the framework
add_subdirectory(framework)

# Gym environment examples
add_subdirectory(gym/core)
add_subdirectory(gym/classic)
add_subdirectory(rl/classic)

# Example RL problems using the MuJoCo physics engine
if(Noesis_USE_MUJOCO)
  add_subdirectory(gym/mujoco)
  add_subdirectory(rl/mujoco)
endif()

# Example RL problems using the RaiSim physics engine
if (Noesis_USE_RAISIM)
  add_subdirectory(gym/raisim)
  add_subdirectory(rl/raisim)
endif()

# EOF
