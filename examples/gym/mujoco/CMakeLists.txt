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

add_noesis_example_app(noesis_gym_hopper_example ${CMAKE_CURRENT_LIST_DIR}/src/hopper.cpp)
#add_noesis_example_app(noesis_gym_halfcheetah_example ${CMAKE_CURRENT_LIST_DIR}/src/halfcheetah.cpp)
#add_noesis_example_app(noesis_gym_walker2d_example ${CMAKE_CURRENT_LIST_DIR}/src/walker2d.cpp)
#add_noesis_example_app(noesis_gym_ant_example ${CMAKE_CURRENT_LIST_DIR}/src/ant.cpp)
#add_noesis_example_app(noesis_gym_humanoid_example ${CMAKE_CURRENT_LIST_DIR}/src/humanoid.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/gym
)

# EOF
