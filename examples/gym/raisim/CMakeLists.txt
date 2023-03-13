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
# Authors: Vassilios Tsounis, tsounisv@ethz.ch
#=============================================================================

#==
# Build targets
#==

add_noesis_example_app(noesis_gym_capler_example ${CMAKE_CURRENT_LIST_DIR}/src/capler.cpp)
add_noesis_example_app(noesis_gym_kinova3_example ${CMAKE_CURRENT_LIST_DIR}/src/kinova3.cpp)
#add_noesis_example_app(noesis_gym_ballbot_example ${CMAKE_CURRENT_LIST_DIR}/src/ballbot.cpp)
#add_noesis_example_app(noesis_gym_anymal_example ${CMAKE_CURRENT_LIST_DIR}/src/anymal.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/gym
)

# EOF
