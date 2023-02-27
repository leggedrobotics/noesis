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

# Capler agent trained w/ PPO
add_noesis_example_app(noesis_rl_train_capler_ppo_example ${CMAKE_CURRENT_LIST_DIR}/src/capler/ppo/train.cpp)
add_noesis_example_app(noesis_rl_test_capler_ppo_example ${CMAKE_CURRENT_LIST_DIR}/src/capler/ppo/test.cpp)

# Capler agent trained w/ TRPO
add_noesis_example_app(noesis_rl_train_capler_trpo_example ${CMAKE_CURRENT_LIST_DIR}/src/capler/trpo/train.cpp)
add_noesis_example_app(noesis_rl_test_capler_trpo_example ${CMAKE_CURRENT_LIST_DIR}/src/capler/trpo/test.cpp)

# Kinova3 agent trained w/ PPO
add_noesis_example_app(noesis_rl_sample_kinova3_example ${CMAKE_CURRENT_LIST_DIR}/src/kinova3/sample.cpp)
add_noesis_example_app(noesis_rl_train_kinova3_example ${CMAKE_CURRENT_LIST_DIR}/src/kinova3/train.cpp)
add_noesis_example_app(noesis_rl_test_kinova3_example ${CMAKE_CURRENT_LIST_DIR}/src/kinova3/test.cpp)
add_noesis_example_app(noesis_rl_perf_kinova3_example ${CMAKE_CURRENT_LIST_DIR}/src/kinova3/perf.cpp)

#==
# Install
#==

install(
  TARGETS ${NOESIS_EXAMPLE_APPS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/rl
)

# EOF
