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
# Authors: Vassilios Tsounis, tsounsiv@ethz.ch
#=============================================================================
cmake_minimum_required(VERSION 3.10)
project(noesis VERSION 0.2.0 LANGUAGES CXX)

# Include CMake helper modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#==
# Options
#==

option(Noesis_USE_MUJOCO "Enables support for the MuJoCo physics engine." OFF)
option(Noesis_USE_RAISIM "Enables support for the RaiSim physics engine." OFF)
option(Noesis_USE_RAISIM_OGRE "Enables support for RaiSim visualizations in OGRE." OFF)
option(Noesis_USE_CXX17 "Enables the use of C++17 as the required standard." OFF)
option(Noesis_USE_SIMD "Enables the use of CPU SIMD commands such as SSE, AVX and FMA." OFF)
option(Noesis_USE_TCMALLOC "Enables use of TCMALLOC for memory allocations and sanitizing." OFF)
option(Noesis_USE_SANITIZERS "Enables the use of GCC memory and thread sanitizers." OFF)
option(Noesis_BUILD_TESTS "Build unit tests." OFF)

#==
# Dependencies
#==

# OpenMP: The OpenMP API specification for parallel programming (https://www.openmp.org/)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP:")
  message(STATUS "  Version: " "${OpenMP_CXX_VERSION}")
  message(STATUS "  Spec: " "${OpenMP_CXX_SPEC_DATE}")
  message(STATUS "  Library names: " "${OpenMP_CXX_LIB_NAMES}")
  message(STATUS "  Libraries: " "${OpenMP_CXX_LIBRARIES}")
  message(STATUS "  C++ Compiler flags: " "${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found! Please install using APT:\nsudo apt install libomp-dev")
endif()

# Boost: Free, peer-reviewed and portable C++ source libraries.
find_package(Boost REQUIRED COMPONENTS filesystem thread program_options)
if(Boost_FOUND)
  message(STATUS "Boost:")
  message(STATUS "  Version: " "${Boost_VERSION}")
  message(STATUS "  Includes: " "${Boost_INCLUDE_DIR}")
  message(STATUS "  Libraries: " "${Boost_LIBRARIES}")
else()
  message(FATAL_ERROR "Boost not found! Please install using APT:\nsudo apt install libboost-all-dev")
endif()

# TinyXML: TinyXML is a simple, small, C++ XML parser that can be easily integrating into other programs.
find_package(PkgConfig)
pkg_check_modules(TinyXML REQUIRED tinyxml)
if(TinyXML_FOUND)
  message(STATUS "TinyXML:")
  message(STATUS "  Version: " "${TinyXML_VERSION}")
  message(STATUS "  Includes: " "${TinyXML_INCLUDEDIR}")
  message(STATUS "  Libraries: " "${TinyXML_LIBRARIES}")
else()
  message(FATAL_ERROR "TinyXML not found! Please install using APT:\nsudo apt install libtinyxml-dev")
endif()

# SFML: Simple and Fast Multimedia Library

if(EXISTS /usr/share/SFML/cmake/Modules)
  list(APPEND CMAKE_MODULE_PATH /usr/share/SFML/cmake/Modules)
endif()
find_package(SFML 2 COMPONENTS system window graphics REQUIRED)
if(SFML_FOUND)
  set(SFML_LIBRARIES sfml-system sfml-window sfml-graphics)
  message(STATUS "SFML2:")
  message(STATUS "  Version: ${SFML_VERSION}")
  message(STATUS "  Libraries: ${SFML_LIBRARIES}")
else()
  message(FATAL_ERROR "SFML2 not found! Please install using APT:\nsudo apt install libsfml-dev")
endif()

# TensorFlow: An Open Source Machine Learning Framework for Everyone (https://tensorflow.org)
find_package(TensorFlow CONFIG REQUIRED)
if(NOT TensorFlow_FOUND)
  message(FATAL_ERROR "TensorFlow not found! Please install from 'https://github.com/leggedrobotics/tensorflow-cpp'.")
endif()

# MuJoCo:  Multi-Joint dynamics with Contact advanced physics simulation
if(Noesis_USE_MUJOCO)
  find_package(mujoco_cpp CONFIG REQUIRED)
  if(NOT mujoco_cpp_FOUND)
    message(FATAL_ERROR "mujoco_cpp not found! Please install from 'https://github.com/leggedrobotics/mujoco_cpp'.")
  endif()
  list(APPEND NOESIS_OPTIONAL_DEPS mujoco::mujoco_cpp)
  list(APPEND NOESIS_OPTIONAL_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/gym/mujoco/simulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gym/mujoco/visualizer.cpp
  )
endif()

# RaiSim: RaiSim, A physics engine for robotics and AI research
if(Noesis_USE_RAISIM)
  find_package(raisim CONFIG REQUIRED)
  if(NOT raisim_FOUND)
    message(FATAL_ERROR "raisim not found! Please install from 'https://github.com/vastsoun/raisimLib'.")
  endif()
  list(APPEND NOESIS_OPTIONAL_DEPS raisim::raisim)
  if(Noesis_USE_RAISIM_OGRE)
    find_package(raisimOgre CONFIG REQUIRED)
    if(NOT raisimOgre_FOUND)
      message(FATAL_ERROR "raisimOgre not found! Please install from 'https://github.com/vastsoun/raisimOgre'.")
    endif()
    list(APPEND NOESIS_OPTIONAL_DEPS raisim::raisimOgre)
    list(APPEND NOESIS_OPTIONAL_SRC
      ${CMAKE_CURRENT_LIST_DIR}/src/gym/raisim/visualizer.cpp
    )
  endif()
endif()

#==
# Build configurations
#==

# Retrieve the root project directory
set(NOESIS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../")
get_filename_component(NOESIS_ROOT ${NOESIS_ROOT} ABSOLUTE)

# Define default build configurations
set(NOESIS_COMPILE_FEATURES "")
set(NOESIS_COMPILE_OPTIONS "-Wl,--no-as-needed" "-mtune=native" ${OpenMP_CXX_FLAGS})
set(NOESIS_COMPILE_DEFINITIONS "-DNOESIS_ROOT=${NOESIS_ROOT}")

# Optionally enable use of CPU SIMD instruction
if(Noesis_USE_CXX17)
  message(STATUS "Noesis: Enabling C++17 requirement.")
  list(APPEND NOESIS_COMPILE_FEATURES "cxx_std_17")
else()
  message(STATUS "Noesis: Using default C++14 requirement.")
  list(APPEND NOESIS_COMPILE_FEATURES "cxx_std_14")
endif()

# Optionally enable use of CPU SIMD instruction
if(Noesis_USE_SIMD)
  message(STATUS "Noesis: Enabling SIMD compiler options.")
  list(APPEND NOESIS_COMPILE_OPTIONS
    "-march=native"
    "-ffast-math"
  )
else()
  message(STATUS "Noesis: Using default compiler options.")
endif()

# Optionally enable use of TCMALLOC for improved memory allocation performance and memory profiling
if(Noesis_USE_TCMALLOC)
  message(STATUS "Noesis: Enabling TCMALLOC as the memory allocator.")
  list(APPEND NOESIS_COMPILE_OPTIONS
    "-fno-builtin-malloc"
    "-fno-builtin-calloc"
    "-fno-builtin-realloc"
    "-fno-builtin-free"
  )
  set(PROFILING_LIBS profiler tcmalloc)
else()
  message(STATUS "Noesis: Using default GCC memory allocator.")
endif()

# Conditionally set internal preprocessor flag for debug builds
if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  list(APPEND NOESIS_COMPILE_DEFINITIONS -DNOESIS_DEBUG)
endif()

#==
# Build targets
#==

# Define source files
file(GLOB NOESIS_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/*.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/framework/*/*.cpp
)
list(APPEND NOESIS_SRC ${NOESIS_OPTIONAL_SRC})

# Define the noesis core library target
add_library(${PROJECT_NAME} STATIC ${NOESIS_SRC})
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    ${TinyXML_LIBRARIES}
    ${SFML_LIBRARIES}
    TensorFlow::TensorFlow
    ${NOESIS_OPTIONAL_DEPS}
    ${PROFILING_LIBS}
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TinyXML_INCLUDEDIR}>
    $<BUILD_INTERFACE:${SFML_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)
target_compile_features(${PROJECT_NAME} PUBLIC ${NOESIS_COMPILE_FEATURES})
target_compile_options(${PROJECT_NAME} PUBLIC ${NOESIS_COMPILE_OPTIONS})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${NOESIS_COMPILE_DEFINITIONS})
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(noesis::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

#==
# Package targets
#==

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # TODO: RUNTIME DESTINATION bin for BASH files?
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

export(TARGETS ${PROJECT_NAME}
  NAMESPACE noesis::
  FILE ${PROJECT_NAME}-targets.cmake
)

install(EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE noesis::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

#==
# Package configuration
#==

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

write_basic_package_version_file(cmake/${PROJECT_NAME}-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

#==
# Package installation
#==

# Install CMake package configuration
install(
  FILES
    ${CMAKE_CURRENT_LIST_DIR}/cmake/GTestMacros.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}-config-version.cmake
  DESTINATION
    ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Install C++ library headers
install(
  DIRECTORY
    ${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/include/stb
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}
)

#==
# Unit tests
#==

if(Noesis_BUILD_TESTS)
  message(STATUS "Noesis: Enabling unit tests.")
  include(cmake/GTestMacros.cmake)
  find_gtest()
  add_subdirectory(test)
endif()

# EOF
