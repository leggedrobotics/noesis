/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_CARTPOLE_CARTPOLE_ENVIRONMENT_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_CARTPOLE_CARTPOLE_ENVIRONMENT_HPP_

// Noesis
#include "noesis/gym/core/Environment.hpp"
#include "noesis/framework/math/ops.hpp"

namespace noesis {
namespace gym {

class CartpoleObservations : public noesis::mdp::Observations<float>
{
public:
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit CartpoleObservations(const std::string& name_scope = "", size_t time_size = 0, size_t batch_size = 0):
    noesis::mdp::Observations<float>(name_scope, time_size, batch_size)
  {
    // The observation consists of the following: [cart_position, cart_velocity, pole_angle, pole_tip_velocity]
    this->addTensor("motion", {4});
  }
};

class CartpoleActions : public noesis::mdp::Actions<float>
{
public:
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit CartpoleActions(const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Actions<float>(scope, time_size, batch_size)
  {
    // The action consists of a simple scalar torque applied at the center of rotation
    this->addTensor("force", {1});
  }
};

/*!
 * Implementation of the classic cart-pole environment
 * Reference: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 *
 * State : x, theta, x_dot, theta_dot
 * Observation: x, x_dot, theta, pole_length * theta_dot
 * Action: discrete values in [0, 1] which map to [-10N, +10N] force acting on cart
 */
class CartpoleEnvironment final : public ::noesis::gym::Environment<float>
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

  explicit CartpoleEnvironment(
      Scalar reset_noise_factor=1.0f,
      Scalar randomization_factor=0.0f,
      Scalar discount_factor = 0.99f,
      Scalar time_step = 0.02f,
      Scalar time_limit = 4.0f,
      const std::string& name="Cartpole",
      const std::string& scope="/",
      bool verbose=false):
    Base(CartpoleObservations().spec(), CartpoleActions().spec(),
      {"benchmark"}, {},
      discount_factor, time_limit, time_step, 1u, 1u,
      name, scope, verbose),
    resetNoiseFactor_(reset_noise_factor),
    randomizationFactor_(randomization_factor)
  {
    NNOTIFY_IF(isVerbose(), "[CartpoleEnvironment]: New instance at: " << std::hex << this);
  }
  
  ~CartpoleEnvironment() override = default;
  
  /*
   * Configurations
   */

  void setMaxForce(Scalar max_force) { maxForce_ = max_force; }
  
  /*
   * Properties
   */

  const Eigen::Vector2f& getPositions() const { return q_; }
  
  const Eigen::Vector2f& getVelocities() const { return dq_; }
  
  Scalar getForce() const { return tau_; }
  
  Scalar getMaxForce() const { return maxForce_; }
  
  Scalar getPoleLength() const { return poleLength_; }
  
private:
  
  void randomize(int seed) override {
    massPole_ *= 1.0f + 0.1f * prng_.sampleUnitUniform() * randomizationFactor_;
    massCart_ *= 1.0f + 0.1f * prng_.sampleUnitUniform() * randomizationFactor_;
    poleLength_ *= 1.0f + 0.1f * prng_.sampleUnitUniform() * randomizationFactor_;
    gravity_ *= 1.0f + 0.1f * prng_.sampleUnitUniform() * randomizationFactor_;
  }
  
  bool initialize(Observations& observations, Terminations& terminations) override {
    // Sample a random initial state
    q_(0) = prng_.sampleUniform(-0.05f, 0.05f) * resetNoiseFactor_;
    q_(1) = prng_.sampleUniform(-0.05f, 0.05f) * resetNoiseFactor_;
    dq_(0) = prng_.sampleUniform(-0.05f, 0.05f) * resetNoiseFactor_;
    dq_(1) = prng_.sampleUniform(-0.05f, 0.05f) * resetNoiseFactor_;
    tau_ = 0.0f;
    // Set initial observations
    fill_observations(observations);
    // TC1: Check if the end-effector has reached the target height
    terminations.back() = check_if_terminal();
  }
  
  bool transition(
      const Actions& actions,
      Observations& observations,
      Rewards& rewards,
      Terminations& terminations,
      Metrics& metrics) override {
    // Retrieve the commanded force from the specified action.
    tau_ = 0.1*force_from_action(actions);
    // Compute forward dynamics.
    const auto dt = time_step();
    float cosTheta = std::cos(q_[1]);
    float sinTheta = std::sin(q_[1]);
    float totalMass = massPole_ + massCart_;
    float halfLength = poleLength_ / 2.0f;
    float temp = (tau_ + massPole_ * halfLength * dq_[1] * dq_[1] * sinTheta) / totalMass;
    float ddp = (gravity_ * sinTheta - cosTheta * temp) / (halfLength * (4.0f / 3.0f - massPole_ * cosTheta * cosTheta / totalMass));
    float ddc = (temp - halfLength * massPole_ * ddp * cosTheta) / totalMass;
    // Update generalized coordinate x
    q_[0] = q_[0] + dt * dq_[0];
    dq_[0] = dq_[0] + dt * ddc;
    // Update generalized coordinate theta
    q_[1] = q_[1] + dt * dq_[1];
    q_[1] = noesis::math::wrap(q_[1], static_cast<Scalar>(-M_PI), static_cast<Scalar>(M_PI));
    dq_[1] = dq_[1] + dt * ddp;
    // Set next observations
    fill_observations(observations);
    // TC1: Check if the end-effector has reached the target height
    terminations.back() = check_if_terminal();
    // Compute reward
    if (terminations.back().type == Termination::Type::TerminalState) {
      rewards[0] = 0.0f;
    } else {
      rewards[0] = 1.0f;
    }
    // Transitions always succeed
    return true;
  }
  
  inline Scalar force_from_action(const Actions& actions) {
    // NOTE: Acrobot uses discrete actions. We thus use a method which returns
    // the element of a Eigen::Matrix<> which is nearest to the specified scalar action.
    Eigen::Vector2f forces = 2.0f * actions_ - Eigen::Vector2f::Ones();
    Scalar force = maxForce_ * noesis::math::select_nearest(forces, actions[0][0]);
    // Return the total final force computed from actions
    return force;
  }
  
  inline void fill_observations(Observations& observations) const {
    constexpr auto xmax = std::numeric_limits<Scalar>::max();
    Eigen::Vector4f obsLimits = Eigen::Vector4f(2.0f*cartThreshold_, xmax, 2.0f*poleThreshold_, xmax);
    Eigen::Vector4f obs;
    obs <<  q_(0), dq_(0), q_(0), dq_(1) * poleLength_;
    // NOTE: Cartpole clips observations to within certain bounds for the cart and pole configurations.
    obs = noesis::math::clip(obs, obsLimits);
    observations[0](0,0).asEigenMatrix() << obs;
  }
  
  inline Termination check_if_terminal() const { // TODO: CHECK WHAT THIS RETURNS
    if (std::abs(q_(0)) >= cartThreshold_ || std::abs(q_(1)) >= poleThreshold_) {
      return {0.0f, Termination::Type::TerminalState};
    } else {
      return {};
    }
  }
  
private:
  // Random number generation
  noesis::math::RandomNumberGenerator<Scalar> prng_;
  //! The available discrete actions values
  Eigen::Vector2f actions_{Eigen::Vector2f(0.0f, 1.0f)};
  //! System dynamics variables
  Eigen::Vector2f q_{Eigen::Vector2f::Zero()};
  Eigen::Vector2f dq_{Eigen::Vector2f::Zero()};
  Scalar tau_{0.0f};
  //! Dynamics configurations
  Scalar massCart_{1.0f};
  Scalar massPole_{0.1f};
  Scalar poleLength_{1.0f};
  //! Gravity constant
  Scalar gravity_{9.8f};
  //! Physical limits
  Scalar maxForce_{10.0f};
  Scalar cartThreshold_{2.4f};
  Scalar poleThreshold_{12.0f * static_cast<Scalar>(M_PI) / 180.0f};
  // Cost and initial state hyper-parameters
  Scalar resetNoiseFactor_{1.0f};
  Scalar randomizationFactor_{0.0f};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_CARTPOLE_CARTPOLE_ENVIRONMENT_HPP_

/* EOF */
