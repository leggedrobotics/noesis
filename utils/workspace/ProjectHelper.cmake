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

macro(project_init)
  set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
  list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local" "${CMAKE_CURRENT_LIST_DIR}/../lib")
  if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/.project.cpp")
    file(WRITE "${CMAKE_CURRENT_LIST_DIR}/src/.project.cpp" "")
  endif()
endmacro()

macro(project_set_eigen_path EIGEN_PATH)
  set(Eigen3_DIR ${EIGEN_PATH}/cmake/ CACHE INTERNAL "")
endmacro()

macro(project_add_catkin_ws CATKIN_WS_DIR)
  list(APPEND CMAKE_PREFIX_PATH "/opt/ros/noetic")
  file(GLOB ROS_MSG_HEADERS ${CATKIN_WS_DIR}/devel/include/*_msgs)
  file(GLOB ROS_CUSTOM_MSG_HEADERS ${CATKIN_WS_DIR}/devel/.private/*/include)
  if (ROS_MSG_HEADERS)
    include_directories("${ROS_MSG_HEADERS}")
  endif()
  if (ROS_CUSTOM_MSG_HEADERS)
    include_directories("${ROS_CUSTOM_MSG_HEADERS}")
  endif()
endmacro()

macro(project_set_option VARIABLE VALUE)
  set(${VARIABLE} ${VALUE} CACHE INTERNAL "")
endmacro()

macro(project_enable_option VARIABLE)
  project_set_option(${VARIABLE} ON)
endmacro()

macro(project_disable_option VARIABLE)
  project_set_option(${VARIABLE} OFF)
endmacro()

macro(project_add_prefix PREFIX)
  list(INSERT CMAKE_PREFIX_PATH 0 ${PREFIX})
endmacro()

macro(project_add_module MODULE)
  add_subdirectory(${MODULE})
  get_property(curr_project_includes DIRECTORY ${MODULE} PROPERTY INCLUDE_DIRECTORIES)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${MODULE}/include" AND IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${MODULE}/include")
    list(APPEND project_includes "${CMAKE_CURRENT_LIST_DIR}/${MODULE}/include")
  endif()
  list(APPEND project_includes ${curr_project_includes})
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/${MODULE})
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR}/${MODULE})
endmacro()

macro(project_build)
  if (project_includes)
    list(REMOVE_DUPLICATES project_includes)
  endif()
  foreach(dir ${project_includes})
    if(NOT ${dir} MATCHES "^${CMAKE_CURRENT_LIST_DIR}/src")
      list(REMOVE_ITEM project_includes ${dir})
    endif()
  endforeach()
  add_library(libproject ${project_includes} src/.project.cpp )
endmacro()

# EOF
