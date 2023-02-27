/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Markus Staeuble
 * @email     markus.staeuble@mavt.ethz.ch
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_MUJOCO_ANT_ANT_ACTIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_ANT_ANT_ACTIONS_HPP_

// Noesis
#include "noesis/mdp/types.hpp"

namespace noesis {
namespace gym {

template<typename ScalarType_>
class AntActions : public noesis::mdp::Actions<ScalarType_>
{
public:
  //! Helper index for accessing each tensor instance
  enum Index {
    Ctrl = 0,
  };

  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit AntActions(const std::string &scope = "", size_t time_size = 0, size_t batch_size = 0) :
    noesis::mdp::Actions<ScalarType_>(scope, time_size, batch_size)
  {
    // Joint commands are the torques
    this->addTensor("ctrl", {8});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_ANT_ANT_ACTIONS_HPP_

/* EOF */
