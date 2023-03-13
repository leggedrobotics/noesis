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

#add_noesis_example_app(train_mujoco_rl_agent ${CMAKE_CURRENT_LIST_DIR}/src/train_mujoco.cpp)
#add_noesis_example_app(test_mujoco_rl_agent ${CMAKE_CURRENT_LIST_DIR}/src/test_mujoco.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/rl
)

# EOF
