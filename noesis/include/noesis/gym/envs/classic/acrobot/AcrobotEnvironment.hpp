/*!
 * @author    HaoChih Lin
 * @email     hlin@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_ACROBOT_ACROBOT_ENVIRONMENT_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_ACROBOT_ACROBOT_ENVIRONMENT_HPP_

// Noesis
#include "noesis/framework/math/ops.hpp"
#include "noesis/gym/core/Environment.hpp"

namespace noesis {
namespace gym {

class AcrobotObservations : public noesis::mdp::Observations<float>
{
public:
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit AcrobotObservations(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Observations<float>(name_scope, time_size, batch_size)
  {
    // The observation consists of the sin() and cos() of the two rotational joint angles and the joint angular velocities:
    // [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]
    this->addTensor("motion", {6});
  }
};

class AcrobotActions : public noesis::mdp::Actions<float>
{
public:
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit AcrobotActions(const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Actions<float>(scope, time_size, batch_size)
  {
    // The action consists of a simple scalar torque applied at the center of rotation
    this->addTensor("torque", {1});
  }
};

/*!
 * Implementation of the classic Acrobot task in C++
 * reference: https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py
 * Generalized coordinate = theta1, theta2 (theta2: relative angle)
 * State : theta1, theta2, theta1_dot, theta2_dot. (0,0,0,0) means two links point down
 * Observation: cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot
 * Action : torque on the joint between the two acrobot links [-1, 0 ,1]
 */
class AcrobotEnvironment final : public ::noesis::gym::Environment<float>
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

  explicit AcrobotEnvironment(
      Scalar reset_noise_factor=1.0f,
      Scalar randomization_factor=0.0f,
      Scalar action_noise_factor=0.0f,
      Scalar discount_factor = 0.99f,
      Scalar time_step = 0.2f,
      Scalar time_limit = 100.0f,
      const std::string& name="Acrobot",
      const std::string& scope="/",
      bool verbose=false):
    Base(AcrobotObservations().spec(), AcrobotActions().spec(),
      {"benchmark"}, {},
      discount_factor, time_limit, time_step, 1u, 1u,
      name, scope, verbose),
    resetNoiseFactor_(reset_noise_factor),
    randomizationFactor_(randomization_factor),
    actionNoiseFactor_(action_noise_factor)
  {
    NNOTIFY_IF(isVerbose(), "[AcrobotEnvironment]: New instance at: " << std::hex << this);
  }
  
  ~AcrobotEnvironment() final = default;
  
  /*
   * Configurations
   */

  void setMaxTorque(Scalar max) { maxTorque_ = max; }
  
  void setMaxFirstLinkVelocity(Scalar max) { maxVelocity1_ = max; }
  
  void setMaxSecondLinkVelocity(Scalar max) { maxVelocity2_ = max; }
  
  /*
   * Properties
   */

  const Eigen::Vector2f& getPositions() const { return q_; }
  
  const Eigen::Vector2f& getVelocities() const { return dq_; }
  
  Scalar getTorque() const { return tau_; }
  
  Scalar getMaxTorque() const { return maxTorque_; }
  
  Scalar getMaxFirstLinkVelocity() const { return maxVelocity1_; }
  
  Scalar getMaxSecondLinkVelocity() const { return maxVelocity2_; }
  
private:
  
  void randomize(int seed) override {
    linkMass1_ *= 1.0 + 0.1 * prng_.sampleUnitUniform() * randomizationFactor_;
    linkMass2_ *= 1.0 + 0.1 * prng_.sampleUnitUniform() * randomizationFactor_;
    linkLength1_ *= 1.0 + 0.1 * prng_.sampleUnitUniform() * randomizationFactor_;
    linkLength2_ *= 1.0 + 0.1 * prng_.sampleUnitUniform() * randomizationFactor_;
    gravity_ *= 1.0 + 0.05 * prng_.sampleUnitUniform() * randomizationFactor_;
  }
  
  bool initialize(Observations& observations, Terminations& terminations) override {
    // Sample a random initial state
    q_(0) = prng_.sampleUniform(-0.1f, 0.1f);
    q_(1) = prng_.sampleUniform(-0.1f, 0.1f);
    dq_(0) = prng_.sampleUniform(-0.1f, 0.1f);
    dq_(1) = prng_.sampleUniform(-0.1f, 0.1f);
    tau_ = 0.0;
    // Set initial observations
    fill_observations(observations);
    // TC1: Check if the end-effector has reached the target height
    terminations.back() = check_if_target_reached();
  }
  
  bool transition(
      const Actions& actions,
      Observations& observations,
      Rewards& rewards,
      Terminations& terminations,
      Metrics& metrics) override {
    // Compute the appropriate discrete-valued torque based on the current action
    tau_ = torque_from_action(actions);
    // Perform integration of the physics
    const auto state_t = Eigen::Vector4f(q_(0), q_(1), dq_(0), dq_(1));
    const auto state_tp1 = rk4(state_t, tau_);
    // Saturate positions and velocities to emulate electric motor dynamics
    q_(0) = noesis::math::wrap(state_tp1(0), -static_cast<Scalar>(M_PI), static_cast<Scalar>(M_PI));
    q_(1) = noesis::math::wrap(state_tp1(1), -static_cast<Scalar>(M_PI), static_cast<Scalar>(M_PI));
    dq_(0) = noesis::math::clip(state_tp1(2), -maxVelocity1_, maxVelocity1_);
    dq_(1) = noesis::math::clip(state_tp1(3), -maxVelocity2_, maxVelocity2_);
    // Set next observations
    fill_observations(observations);
    // TC1: Check if the end-effector has reached the target height
    terminations.back() = check_if_target_reached();
    // Compute reward
    if (terminations.back().type == Termination::Type::TerminalState) {
      rewards[0] = 0.0f;
    } else {
      rewards[0] = -1.0f;
    }
    // Transitions always succeed
    return true;
  }
  
  inline Scalar torque_from_action(const Actions& actions) {
    // NOTE: Acrobot uses discrete actions. We thus use a method which returns
    // the element of a Eigen::Matrix<> which is nearest to the specified scalar action.
    Scalar torque = maxTorque_ * noesis::math::select_nearest(actions_, actions[0][0]);
    // We optionally apply additional action noise.
    torque += prng_.sampleUnitUniform() * actionNoiseFactor_;
    // Return the total final torque computed from actions and additional process noise
    return torque;
  }
  
  inline void fill_observations(Observations& observations) const {
    observations[0](0,0).asEigenMatrix() << std::cos(q_(0)), std::sin(q_(0)), std::cos(q_(1)), std::sin(q_(1)), dq_(0), dq_(1);
  }
  
  inline Termination check_if_target_reached () const {
    if (-std::cos(q_(0)) - std::cos(q_(0) + q_(1)) > 1.0f) {
      return {0.0f, Termination::Type::TerminalState};
    } else {
      return {};
    }
  }
  
  Eigen::Vector4f derivative(const Eigen::Vector4f& x_t, Scalar u_t) {
    Eigen::Vector4f dx_t;
    const auto linkCenter1 = linkLength1_ / 2.0f;
    const auto linkCenter2 = linkLength2_ / 2.0f;
    const auto d1 =
      linkMass1_ * std::pow(linkCenter1, 2)
      + linkMass2_ * (std::pow(linkLength1_, 2)
      + std::pow(linkCenter2, 2)
      + 2.0 * linkLength1_ * linkCenter2 * std::cos(x_t(1)))
      + linkMoment1_ + linkMoment2_;
    const auto d2 =
      linkMass2_ * (std::pow(linkCenter2, 2)
      + linkLength1_ * linkCenter2 * std::cos(x_t(1)))
      + linkMoment2_;
    const auto phi2 = linkMass2_ * linkCenter2 * gravity_ * std::cos(x_t(0) + x_t(1) - M_PI / 2.0);
    const auto phi1 =
      -linkMass2_ * linkLength1_ * linkCenter2 * std::pow(x_t(3), 2) * std::sin(x_t(1))
      - 2.0f * linkMass2_ * linkLength1_ * linkCenter2 * x_t(3) * x_t(2) * std::sin(x_t(1))
      + (linkMass1_ * linkCenter1 + linkMass2_ * linkLength1_) * gravity_ * std::cos(x_t(0) - M_PI / 2.0f)
      + phi2;
    dx_t(3) =
      (u_t + d2 / d1 * phi1 - linkMass2_ * linkLength1_ * linkCenter2 * std::pow(x_t(2), 2) * std::sin(x_t(1)) - phi2)
      / (linkMass2_ * std::pow(linkCenter2, 2) + linkMoment2_ - d2 * d2 / d1);
    dx_t(2) = -(d2 * dx_t(3) + phi1) / d1;
    dx_t(0) = x_t(2);
    dx_t(1) = x_t(3);
    return dx_t;
  }
  
  Eigen::Vector4f rk4(const Eigen::Vector4f& x_t, Scalar u_t) {
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Scalar>> dx(4);
    Eigen::Vector4f x_tp1;
    const auto dt = time_step();
    dx[0] = derivative(x_t, u_t);
    x_tp1 = x_t + dt * 0.5f * dx[0];
    dx[1] = derivative(x_tp1, u_t);
    x_tp1 = x_t + dt * 0.5f * dx[1];
    dx[2] = derivative(x_tp1, u_t);
    x_tp1 = x_t + dt * dx[2];
    dx[3] = derivative(x_tp1, u_t);
    x_tp1 = x_t + dt / 6.0f * (dx[0] + 2.0f * dx[1] + 2.0f * dx[2] + dx[3]);
    return x_tp1;
  }
  
private:
  // Random number generation
  noesis::math::RandomNumberGenerator<Scalar> prng_;
  //! The available discrete torques values
  Eigen::Vector3f actions_{Eigen::Vector3f(-1.0f, 0.0f, +1.0f)};
  //! System dynamics variables
  Eigen::Vector2f q_{Eigen::Vector2f::Zero()};
  Eigen::Vector2f dq_{Eigen::Vector2f::Zero()};
  Scalar tau_{0.0f};
  //! Dynamics configurations
  Scalar linkMass1_{1.0f};
  Scalar linkMass2_{1.0f};
  Scalar linkLength1_{1.0f};
  Scalar linkLength2_{1.0f};
  Scalar linkMoment1_{1.0f};
  Scalar linkMoment2_{1.0f};
  //! Gravity constant
  Scalar gravity_{9.81f};
  //! Physical limits
  Scalar maxTorque_{1.0};
  Scalar maxVelocity1_{4.0f * static_cast<Scalar>(M_PI)};
  Scalar maxVelocity2_{9.0f * static_cast<Scalar>(M_PI)};
  // Cost and initial state hyper-parameters
  Scalar resetNoiseFactor_{1.0f};
  Scalar randomizationFactor_{0.0f};
  Scalar actionNoiseFactor_{0.0f};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_ACROBOT_ACROBOT_ENVIRONMENT_HPP_

/* EOF */
