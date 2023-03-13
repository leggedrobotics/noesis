/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Markus Staeuble
 * @email     markus.staeuble@mavt.ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_MUJOCO_HUMANOID_HUMANOID_OBSERVATIONS_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_HUMANOID_HUMANOID_OBSERVATIONS_HPP_

// noesis
#include <noesis/mdp/types.hpp>

namespace noesis {
namespace gym {

template<typename ScalarType_>
class HumanoidObservations : public noesis::mdp::Observations<ScalarType_>
{
public:
  //! Helper index for accessing each observation Tensor
  enum Index {
    GenCoord = 0,
    GenVel,
    ComInert,
    ComVel,
    ActForces,
    CartExtFrc
  };
  
  //! The unique constructor defines the observations type
  explicit HumanoidObservations(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Observations<ScalarType_>(name_scope, time_size, batch_size)
  {
    // Generalized coordinates, except for absolute X-Y position
    this->addTensor("qpos", {7 + 17 - 2});
    // Generalized velocities
    this->addTensor("qvel", {6 + 17});
    // COM-based body inertia and mass := 14x10, 14 bodies x 10
    this->addTensor("cinert", {14u * 10u});
    // COM-based velocity [3D rot; 3D tran] := 14 * 6D
    this->addTensor("cvel", {14u * 6u});
    // Actuator Forces
    this->addTensor("qfrc_actuator", {6 + 17});
    // COM-based external force on body := 14x6, 14 bodies x 6D wrench
    this->addTensor("cfrc_ext", {14u * 6u});
  }
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_HUMANOID_HUMANOID_OBSERVATIONS_HPP_

/* EOF */
