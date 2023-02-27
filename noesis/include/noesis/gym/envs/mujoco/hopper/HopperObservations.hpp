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
#ifndef NOESIS_GYM_ENVS_MUJOCO_HOPPER_HOPPER_OBSERVATIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_HOPPER_HOPPER_OBSERVATIONS_HPP_

// noesis
#include <noesis/mdp/types.hpp>

namespace noesis {
namespace gym {

template<typename ScalarType_>
class HopperObservations : public noesis::mdp::Observations<ScalarType_>
{
public:
  //! Helper index for accessing each observation Tensor
  enum Index {
    GenCoord = 0,
    GenVel,
  };
  
  //! The unique constructor defines the observations type
  explicit HopperObservations(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Observations<ScalarType_>(name_scope, time_size, batch_size)
  {
    // Generalized coordinates, except for absolute X position
    this->addTensor("qpos", {3 + 3 - 1});
    // Generalized velocities
    this->addTensor("qvel", {3 + 3});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_HOPPER_HOPPER_OBSERVATIONS_HPP_

/* EOF */
