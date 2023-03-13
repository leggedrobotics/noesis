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
#ifndef NOESIS_GYM_ENVS_MUJOCO_HALFCHEETAH_HALFCHEETAH_OBSERVATIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_HALFCHEETAH_HALFCHEETAH_OBSERVATIONS_HPP_

// noesis
#include <noesis/mdp/types.hpp>

namespace noesis {
namespace gym {

template<typename ScalarType_>
class HalfcheetahObservations : public noesis::mdp::Observations<ScalarType_>
{
public:
  //! Helper index for accessing each observation Tensor
  enum Index {
    GenCoord = 0,
    GenVel,
  };

  //! The unique constructor defines the observations type
  explicit HalfcheetahObservations(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Observations<ScalarType_>(name_scope, time_size, batch_size)
  {
    // Generalized coordinates, except for absolute X position
    this->addTensor("qpos", {3 + 6 - 1});
    // Generalized velocities
    this->addTensor("qvel", {3 + 6});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_HALFCHEETAH_HALFCHEETAH_OBSERVATIONS_HPP_

/* EOF */
