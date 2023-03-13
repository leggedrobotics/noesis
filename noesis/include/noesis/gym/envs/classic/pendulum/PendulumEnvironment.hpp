/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_PENDULUM_PENDULUM_ENVIRONMENT_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_PENDULUM_PENDULUM_ENVIRONMENT_HPP_

// Noesis
#include "noesis/framework/math/ops.hpp"
#include "noesis/gym/core/Environment.hpp"

namespace noesis {
namespace gym {

class PendulumObservations : public noesis::mdp::Observations<float>
{
public:
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit PendulumObservations(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Observations<float>(name_scope, time_size, batch_size)
  {
    // The observation consists of the following: [cos(theta), sin(theta), theta_dot]
    this->addTensor("motion", {3});
  }
};

class PendulumActions : public noesis::mdp::Actions<float>
{
public:
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit PendulumActions(const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Actions<float>(scope, time_size, batch_size)
  {
    // The action consists of a simple scalar torque applied at the center of rotation
    this->addTensor("torque", {1});
  }
};

/*!
 * Implementation of the classic inverted pendulum environment.
 * Reference: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
 * State : theta, theta_dot
 * Observation: cos(theta), sin(theta), theta_dot
 * Action: torque
 */
class PendulumEnvironment final : public ::noesis::gym::Environment<float>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Aliases
  using Base = ::noesis::gym::Environment<float>;
  using Scalar = typename Base::Scalar;
  using Observations = typename Base::Observations;
  using Actions = typename Base::Actions;
  using Rewards = typename Base::Rewards;
  using Metrics = typename Base::Metrics;
  using Termination = typename Base::Termination;
  using Terminations = typename Base::Terminations;
  using Names = typename Base::Names;

  /*
   * Instantiation
   */

  explicit PendulumEnvironment(
      Scalar reset_noise_factor=1.0f,
      Scalar randomization_factor=0.0f,
      Scalar discount_factor = 0.99f,
      Scalar time_step = 0.05f,
      Scalar time_limit = 10.0f,
      const std::string& name="Pendulum",
      const std::string& scope="/",
      bool verbose=false):
    Base(PendulumObservations().spec(), PendulumActions().spec(),
      {"benchmark"}, {},
      discount_factor, time_limit, time_step, 1u, 1u,
      name, scope, verbose),
    resetNoiseFactor_(reset_noise_factor),
    randomizationFactor_(randomization_factor)
  {
    NNOTIFY_IF(isVerbose(), "[PendulumEnvironment]: New instance at: " << std::hex << this);
  }

  ~PendulumEnvironment() final = default;

  /*
   * Configurations
   */

  void setMaxTorque(Scalar max_torque) {
    maxTorque_ = max_torque;
  }

  /*
   * Properties
   */

  Scalar getPosition() const {
    return q_;
  }

  Scalar getVelocity() const {
    return dq_;
  }

  Scalar getTorque() const {
    return tau_;
  }

  Scalar getMaxTorque() const {
    return maxTorque_;
  }

private:

  void randomize(int seed) override {
    prng_.seed(static_cast<uint32_t>(seed));
    mass_ *= 1.0f + 0.1f * prng_.sampleUnitUniform() * randomizationFactor_;
    length_ *= 1.0f + 0.1f * prng_.sampleUnitUniform() * randomizationFactor_;
    gravity_ *= 1.0f + 0.05f * prng_.sampleUnitUniform() * randomizationFactor_;
  }

  bool initialize(Observations& observations, Terminations& terminations) override {
    q_ = prng_.sampleUniform(-initialStateLimits_(0), initialStateLimits_(0)) * resetNoiseFactor_;
    dq_ = prng_.sampleUniform(-initialStateLimits_(1), initialStateLimits_(1)) * resetNoiseFactor_;
    tau_ = 0.0f;
    // Set observations
    observations[0](0,0).asEigenMatrix() << std::cos(q_), std::sin(q_), dq_;
    // NOTE: No termination condition
    terminations.back() = Termination();
    // Success
    return true;
  }
  
  bool transition(
      const Actions& actions,
      Observations& observations,
      Rewards& rewards,
      Terminations& terminations,
      Metrics& metrics) override {
    // Retrieve environment time-step
    const auto dt = time_step();
    // Saturate output to model actuator dynamics
    tau_ = noesis::math::clip(actions[0][0], -maxTorque_, maxTorque_);
    // Update velocity from force-acceleration dynamics
    dq_ += 3.0f*dt*(-gravity_*std::sin(q_+ static_cast<float>(M_PI))/(2.0f*length_) + tau_/(mass_*length_*length_));
    // Integrate velocity to compute position
    q_ += dq_*dt;
    // Saturate velocity to emulate electric motor dynamics
    dq_ = noesis::math::clip(dq_, -maxSpeed_, maxSpeed_);
    // Set observations
    observations[0](0,0).asEigenMatrix() << std::cos(q_), std::sin(q_), dq_;
    // Compute reward
    const auto theta = normalized_angle(q_);
    const Scalar costs = 1.0f * theta*theta + 0.1f * dq_*dq_ + 0.001f * tau_*tau_;
    const Scalar reward = noesis::math::lgsk(-costs);
    rewards[0] = static_cast<Scalar>(reward);
    // NOTE: No termination condition
    terminations.back() = Termination();
    // Transitions always succeed
    return true;
  }

  static inline Scalar normalized_angle(Scalar phi) {
    constexpr auto pi = static_cast<Scalar>(M_PI);
    return std::fmod(phi + pi, 2.0f * pi) - pi;
  }

private:
  // Random number generation
  noesis::math::RandomNumberGenerator<Scalar> prng_;
  //! @brief Internal helper to define limits of the dynamics state used for initial state sampling.
  Eigen::Vector2f initialStateLimits_{Eigen::Vector2f(static_cast<Scalar>(M_PI), 1.0f)};
  //! System dynamics variables
  Scalar q_{0.0f};
  Scalar dq_{0.0f};
  Scalar tau_{0.0f};
  //! Dynamics configurations
  Scalar mass_{1.0f};
  Scalar length_{1.0f};
  //! Gravity constant
  Scalar gravity_{10.0f};
  //! Physical limits
  Scalar maxTorque_{2.0f};
  Scalar maxSpeed_{8.0f};
  // Cost and initial state hyper-parameters
  Scalar resetNoiseFactor_{1.0f};
  Scalar randomizationFactor_{0.0f};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_PENDULUM_PENDULUM_ENVIRONMENT_HPP_

/* EOF */
