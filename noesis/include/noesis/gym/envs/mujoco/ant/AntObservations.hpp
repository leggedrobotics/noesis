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
#ifndef NOESIS_GYM_ENVS_MUJOCO_ANT_ANT_OBSERVATIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_ANT_ANT_OBSERVATIONS_HPP_

// Noesis
#include "noesis/mdp/types.hpp"

namespace noesis {
namespace gym {

template<typename ScalarType_>
class AntObservations : public noesis::mdp::Observations<ScalarType_>
{
public:
  //! Helper index for accessing each observation Tensor
  enum Index {
    GenCoord = 0,
    GenVel,
    CartExtFrc
  };

  //! The unique constructor defines the observations type
  explicit AntObservations(const std::string &scope = "", size_t time_size = 0, size_t batch_size = 0) :
    noesis::mdp::Observations<ScalarType_>(scope, time_size, batch_size)
  {
    // Generalized coordinates, except for absolute X-Y position
    this->addTensor("qpos", {7 + 8 - 2});
    // Generalized velocities
    this->addTensor("qvel", {6 + 8});
    // COM-based external force on body := 14x6, 14 bodies x 6D wrench
    this->addTensor("cfrc_ext", {14u * 6u});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_ANT_ANT_OBSERVATIONS_HPP_

/* EOF */
