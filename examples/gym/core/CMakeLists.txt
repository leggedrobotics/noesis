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

#==
# Build targets
#==

add_noesis_example_app(noesis_gym_simple_example ${CMAKE_CURRENT_LIST_DIR}/src/simple.cpp)
add_noesis_example_app(noesis_gym_vectorized_example ${CMAKE_CURRENT_LIST_DIR}/src/vectorized.cpp)
add_noesis_example_app(noesis_gym_synchronized_example ${CMAKE_CURRENT_LIST_DIR}/src/synchronized.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/gym
)
# EOF
