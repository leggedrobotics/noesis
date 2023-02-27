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

# Noesis system (and platform) interfaces
set(TEST_SYSTEM_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/system/TimeTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/system/TensorFlowTest.cpp
)
add_gtest(test_${PROJECT_NAME}_system "${TEST_SYSTEM_SRC}" noesis::noesis)

# Noesis general-purpose utilities
set(TEST_UTILS_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/utils/StringUtilsTest.cpp
)
add_gtest(test_${PROJECT_NAME}_utils "${TEST_UTILS_SRC}" noesis::noesis)

# Noesis math operations and facilities.
set(TEST_MATH_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/math/RandomNumberGeneratorTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/math/StatisticsOperationsTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/math/TensorStatisticTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/math/TensorTupleStatisticTest.cpp
)
add_gtest(test_${PROJECT_NAME}_math "${TEST_MATH_SRC}" noesis::noesis)

# Noesis data logging interfaces
set(TEST_LOGGING_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/log/TimerTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/log/TensorBoardTest.cpp
)
add_gtest(test_${PROJECT_NAME}_logging "${TEST_LOGGING_SRC}" noesis::noesis)

# Noesis Tensor wrappers and interfaces
set(TEST_TENSORS_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/core/TensorsSpecTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/core/TensorAllocatorTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/core/TensorMapTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/core/TensorTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/core/TensorTupleTest.cpp
)
add_gtest(test_${PROJECT_NAME}_tensors "${TEST_TENSORS_SRC}" noesis::noesis)

# Noesis data logging interfaces
set(TEST_HYPERPARAMS_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/hyperparam/HyperParameterTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/hyperparam/HyperParameterTupleTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/hyperparam/HyperParameterManagerTest.cpp
)
add_gtest(test_${PROJECT_NAME}_hyperparams "${TEST_HYPERPARAMS_SRC}" noesis::noesis)

### Classic
#add_subdirectory(test/classic)
### MuJoCo
if(Noesis_USE_MUJOCO)
  add_subdirectory(src/gym/mujoco)
endif()
### raisim
if (Noesis_USE_RAISIM)
  add_subdirectory(src/gym/raisim)
endif()

# EOF
