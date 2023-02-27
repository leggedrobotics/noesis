/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_MUJOCO_HOPPER_HOPPER_ACTIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_HOPPER_HOPPER_ACTIONS_HPP_

// noesis
#include <noesis/mdp/types.hpp>

namespace noesis {
namespace gym {

template<typename ScalarType_>
class HopperActions : public noesis::mdp::Actions<ScalarType_>
{
public:
  //! Helper index for accessing each action Tensor
  enum Index {
    Ctrl = 0,
  };
  //!
  explicit HopperActions(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Actions<ScalarType_>(name_scope, time_size, batch_size)
  {
    this->addTensor("ctrl", {3});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_HOPPER_HOPPER_ACTIONS_HPP_

/* EOF */
