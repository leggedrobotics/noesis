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

#==
# Unit tests for MuJoCo environments
#==

message(STATUS "${PROJECT_NAME}: Including tests for MuJoCo environments")

## MuJoCo wrappers.
set(TEST_MUJOCO_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MujocoSimulationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MujocoVisualizerTest.cpp
)
add_gtest(test_${PROJECT_NAME}_mujoco_wrapper "${TEST_MUJOCO_SRC}"  noesis::noesis)

# Hopper: Plannar (2D) single-legged hopping system.
set(TEST_HOPPER_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/HopperEnvironmentTest.cpp
)
add_gtest(test_${PROJECT_NAME}_mujoco_hopper "${TEST_HOPPER_SRC}" noesis::noesis)

# HalfCheetah: Plannar (2D) simplified runner.
set(TEST_HALFCHEETAH_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/HalfcheetahEnvironmentTest.cpp
)
add_gtest(test_${PROJECT_NAME}_mujoco_halfcheetah "${TEST_HALFCHEETAH_SRC}" noesis::noesis)

# Walker2D: Plannar (2D) bipedal walker.
set(TEST_WALKER2D_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Walker2dEnvironmentTest.cpp
)
add_gtest(test_${PROJECT_NAME}_mujoco_walker2d "${TEST_WALKER2D_SRC}" noesis::noesis)

# Ant: Simplified 8-DoF quadrupedal walker.
set(TEST_ANT_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AntEnvironmentTest.cpp
)
add_gtest(test_${PROJECT_NAME}_mujoco_ant "${TEST_ANT_SRC}" noesis::noesis)

# Humanoid: Standard MuJoCo rag-doll humanoid walker.
set(TEST_HUMANOID_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/HumanoidEnvironmentTest.cpp
)
add_gtest(test_${PROJECT_NAME}_mujoco_humanoid "${TEST_HUMANOID_SRC}" noesis::noesis)

# EOF
