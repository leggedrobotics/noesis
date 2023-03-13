/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Markus Staeuble
 * @email     markus.staeuble@mavt.ethz.ch
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_MUJOCO_WALKER2D_WALKER2D_ACTIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_WALKER2D_WALKER2D_ACTIONS_HPP_

// noesis
#include <noesis/mdp/types.hpp>

namespace noesis {
namespace gym {

template<typename ScalarType_>
class Walker2dActions : public noesis::mdp::Actions<ScalarType_>
{
public:
  //! Helper index for accessing each action Tensor
  enum Index {
    Ctrl = 0,
  };
  
  //!
  explicit Walker2dActions(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Actions<ScalarType_>(name_scope, time_size, batch_size)
  {
    this->addTensor("ctrl", {6});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_WALKER2D_WALKER2D_ACTIONS_HPP_

/* EOF */
