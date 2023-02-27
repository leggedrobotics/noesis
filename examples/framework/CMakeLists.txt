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

add_noesis_example_app(noesis_hello_example ${CMAKE_CURRENT_LIST_DIR}/src/hello.cpp)
add_noesis_example_app(noesis_message_example ${CMAKE_CURRENT_LIST_DIR}/src/message.cpp)
add_noesis_example_app(noesis_prng_example ${CMAKE_CURRENT_LIST_DIR}/src/random.cpp)
add_noesis_example_app(noesis_hyperparams_example ${CMAKE_CURRENT_LIST_DIR}/src/hyperparams.cpp)
add_noesis_example_app(noesis_tensorboard_example ${CMAKE_CURRENT_LIST_DIR}/src/tensorboard.cpp)
add_noesis_example_app(noesis_graph_example ${CMAKE_CURRENT_LIST_DIR}/src/graph.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/framework
)

# EOF
