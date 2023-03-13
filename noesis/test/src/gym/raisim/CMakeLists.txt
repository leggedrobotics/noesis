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

#==
# Unit tests for RaiSim environments
#==

message(STATUS "${PROJECT_NAME}: Including tests for RaiSim environments")

## MuJoCo wrappers.
set(TEST_RAISIM_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CaplerSimulationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Kinova3SimulationTest.cpp
)
add_gtest(test_${PROJECT_NAME}_gym_raisim "${TEST_RAISIM_SRC}"  noesis::noesis)

# EOF
