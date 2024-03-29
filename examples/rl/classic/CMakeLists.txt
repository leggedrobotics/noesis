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

add_noesis_example_app(noesis_rl_pendulum_example ${CMAKE_CURRENT_LIST_DIR}/src/pendulum/pendulum.cpp)
add_noesis_example_app(noesis_rl_cartpole_example ${CMAKE_CURRENT_LIST_DIR}/src/cartpole/cartpole.cpp)
add_noesis_example_app(noesis_rl_acrobot_example ${CMAKE_CURRENT_LIST_DIR}/src/acrobot/acrobot.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/rl
)

# EOF
