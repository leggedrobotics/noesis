#=============================================================================
# Copyright (C) 2020, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# Authors: Vassilios Tsounis, tsounisv@ethz.ch
#=============================================================================

# Helper macro function for finding and adding GTest to a project's build tree.
macro(find_gtest)
  # Enables CMake unit-test integration for GTest.
  enable_testing()
  set(GOOGLETEST_VERSION 1.10.0)
  include(GoogleTest)
  # Set root directory for GTest
  set(GOOGLETEST_ROOT "/usr/src/gtest")
  # Check if Google's Testing framework exists in the current system.
  if (NOT EXISTS "${GOOGLETEST_ROOT}")
    message(FATAL_ERROR
      "'${GOOGLETEST_ROOT}' directory not found!\n"
      "Please install the Google Testing framework using APT:\n"
      "sudo apt install libgtest-dev googletest"
    )
  else()
    message(STATUS "Adding GoogleTest from: ${GOOGLETEST_ROOT}")
  endif()
  # Add the GTest CMake package to the current build tree if `gtest` is not already defined.
  if(NOT TARGET gtest)
    # GTest: Google Testing Framework
    add_subdirectory("${GOOGLETEST_ROOT}" ${CMAKE_BINARY_DIR}/googletest)
    set(GTEST_FROM_SOURCE_FOUND "True" CACHE INTERNAL "")
    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include" CACHE INTERNAL "")
    set(GTEST_LIBRARIES gtest gtest_main CACHE INTERNAL "")
    message(STATUS "Found GoogleTest")
    message(STATUS "GoogleTest:")
    message(STATUS "  Includes: ${GTEST_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${GTEST_LIBRARIES}")
  endif()
endmacro()

# Helper macro function for adding tests to the CMake build.
macro(add_gtest TEST_NAME FILES LIBRARIES)
  add_executable(${TEST_NAME} ${FILES})
  target_link_libraries(${TEST_NAME} PUBLIC gtest gtest_main ${LIBRARIES})
  gtest_discover_tests(${TEST_NAME})
endmacro()

# EOF
