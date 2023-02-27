/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_CAPLER_ENVIRONMENT_HPP_
#define NOESIS_GYM_ENVS_RAISIM_CAPLER_ENVIRONMENT_HPP_

// Noesis
#include "noesis/framework/utils/macros.hpp"
#include "noesis/framework/math/ops.hpp"
#include "noesis/gym/core/Environment.hpp"
#include "noesis/gym/envs/raisim/common/logging.hpp"
#include "noesis/gym/envs/raisim/capler/CaplerSimulation.hpp"

namespace noesis {
namespace gym {

template<typename ScalarType_>
class CaplerObservations : public noesis::mdp::Observations<ScalarType_>
{
public:
  //! Helper index for accessing each observation Tensor
  enum Index {
    Target = 0,
    Motion,
    History,
  };
  //! The unique constructor defines the observations type
  explicit CaplerObservations(const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Observations<ScalarType_>(scope, time_size, batch_size) {
    /// Target base height
    this->addTensor("target", {1});
    // Base + joint positions and velocities
    this->addTensor("motion", {6});
    // Actions history
    this->addTensor("history", {2});
  }
};

template<typename ScalarType_>
class CaplerActions : public noesis::mdp::Actions<ScalarType_>
{
public:
  //! Helper index for accessing each tensor instance
  enum Index {
    Commands = 0,
  };
  //! Explicit constructor ensures that each instantiation properly configures all tensors
  explicit CaplerActions(const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Actions<ScalarType_>(scope, time_size, batch_size)
  {
    // Joint commands can be either torques or joint-space PD position references
    this->addTensor("commands", {2});
  }
};

class CaplerEnvironment final: public noesis::gym::Environment<float>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Aliases
  using Base = noesis::gym::Environment<float>;
  using Scalar = typename Base::Scalar;
  using Observations = typename Base::Observations;
  using Actions = typename Base::Actions;
  using Rewards = typename Base::Rewards;
  using Metrics = typename Base::Metrics;
  using Termination = typename Base::Termination;
  using Terminations = typename Base::Terminations;
  using Names = typename Base::Names;
  using VectorX = Eigen::VectorXf;
  using MatrixX = Eigen::MatrixXf;
  
  // Constants
  static constexpr int Nj = CaplerSimulation::Nj;
  
  /*
   * Instantiation
   */
  
  explicit CaplerEnvironment(
      double time_step = 0.01,
      double time_limit = 5.0,
      double discount_factor = 0.995,
      double terminal_value=-5.0,
      double randomization_factor=0.0,
      double reset_noise_factor=0.0,
      double state_noise_factor=0.0,
      double goal_noise_factor=0.0,
      bool use_pid_controller=true,
      bool enable_logging=false,
      const std::string& name="Capler",
      const std::string& scope="/",
      bool verbose=false):
    Base(CaplerObservations<Scalar>().spec(), CaplerActions<Scalar>().spec(),
      {"total", "base_height", "joint_velocity", "joint_torque", "foot_slip", "foot_support", "action"},
      {"joint_power", "action_norm"},
      discount_factor, time_limit, time_step, 1u, 1u, name, scope, verbose),
    stateScale_(VectorX::Constant(CaplerObservations<Scalar>().datum_size(), 1.0)),
    stateOffset_(VectorX::Constant(CaplerObservations<Scalar>().datum_size(), 0.0)),
    stateNoise_(VectorX::Constant(CaplerObservations<Scalar>().datum_size(), 0.0)),
    actionScale_(VectorX::Constant(CaplerActions<Scalar>().datum_size(), 1.0)),
    actionOffset_(VectorX::Constant(CaplerActions<Scalar>().datum_size(), 0.0)),
    terminalValue_(terminal_value),
    randomizationFactor_(randomization_factor),
    resetNoiseFactor_(reset_noise_factor),
    stateNoiseFactor_(state_noise_factor),
    goalNoiseFactor_(goal_noise_factor),
    usingPidController_(use_pid_controller)
  {
    NNOTIFY_IF(isVerbose(), "[CaplerEnvironment]: New instance at: " << std::hex << this);
    // Simulation
    simulation_ = std::make_unique<CaplerSimulation>(CaplerSimulation::World::Grid, use_pid_controller);
    // Initialize action
    command_ = (usingPidController_) ? simulation_->getNominalJointConfiguration() : Eigen::VectorXd::Zero(Nj);
    // State transforms
    stateScale_ <<
      VectorX::Constant(1, 1.0), // desired base height
      VectorX::Constant(1, 1.0), // measured base height
      VectorX::Constant(1, 1.0), // measured base velocity
      VectorX::Constant(Nj, 0.5), // measured joint positions
      VectorX::Constant(Nj, 1.0f/30.0f), // measured joint velocities
      VectorX::Constant(Nj, 0.5); // action history
    stateOffset_ <<
      VectorX::Constant(1, 0.5), // desired base height
      VectorX::Constant(1, 0.5), // measured base height
      VectorX::Constant(1, 0.0), // measured base velocity
      simulation_->getNominalJointConfiguration().template cast<Scalar>(), // measured joint positions
      VectorX::Constant(Nj, 0.0), // measured joint velocities
      VectorX::Constant(Nj, 0.0); // action history
    stateNoise_ <<
      VectorX::Constant(1, 0.0), // desired base height - no noise necessary
      VectorX::Constant(1, 0.005), // measured base height
      VectorX::Constant(1, 0.01), // measured base velocity
      VectorX::Constant(Nj, 0.01), // measured joint positions
      VectorX::Constant(Nj, 0.02), // measured joint velocities
      VectorX::Constant(Nj, 0.0); // action history - no noise necessary
    if (usingPidController_) {
      stateScale_.tail(Nj) = VectorX::Constant(Nj, 0.5);
      stateOffset_.tail(Nj) = simulation_->getNominalJointConfiguration().template cast<Scalar>();
    } else {
      stateScale_.tail(Nj) = VectorX::Constant(Nj, static_cast<Scalar>(1.0/simulation_->getMaxTorque()));
      stateOffset_.tail(Nj) = VectorX::Constant(Nj, 0.0);
    }
    // Action transforms
    if (usingPidController_) {
      actionScale_.setConstant(1.0);
      actionOffset_ = simulation_->getNominalJointConfiguration().template cast<Scalar>();
    } else {
      actionScale_.setConstant(static_cast<Scalar>(simulation_->getMaxTorque()));
      actionOffset_.setZero();
    }
    // TensorBoard Logging
    if(enable_logging) {
      logger_ = std::make_unique<noesis::log::TensorBoardLogger>("", "policy", "policy", "", false);
      // Reward/cost terms
      logger_->addLoggingSignal("Rewards/base_height");
      logger_->addLoggingSignal("Rewards/joint_velocity");
      logger_->addLoggingSignal("Rewards/joint_torque");
      logger_->addLoggingSignal("Rewards/foot_slip");
      logger_->addLoggingSignal("Rewards/foot_support");
      logger_->addLoggingSignal("Rewards/action");
      logger_->addLoggingSignal("Rewards/total");
      logger_->addLoggingSignal("Rewards/total_normalized");
      // State measurements
      logger_->addLoggingSignal("State/Base/position_z");
      logger_->addLoggingSignal("State/Base/linear_velocity_z");
      logging::add_foot(*logger_, "State/FOOT");
      logging::add_joint(*logger_, "State/HFE");
      logging::add_joint(*logger_, "State/KFE");
      logger_->startup();
    }
  }
  
  ~CaplerEnvironment() final = default;
  
  /*
   * Configurations
   */

  void setRandomizationFactor(double factor) {
    randomizationFactor_ = factor;
  }
  
  void setResetNoiseFactor(double factor) {
    resetNoiseFactor_ = factor;
  }
  
  void setStateNoiseFactor(double factor) {
    stateNoiseFactor_ = factor;
  }
  
  void setGoalNoiseFactor(double factor) {
    goalNoiseFactor_ = factor;
  }
  
  /*
   * Properties
   */

  const CaplerSimulation& simulation() const { return *simulation_; }
  
  CaplerSimulation& simulation() { return *simulation_; }
  
  const Position& goal() const { return W_r_WG_; }
  
  /*
   * Operations
   */

  void randomize(int seed) override {
    simulation_->randomize(seed, randomizationFactor_);
    prng_.seed(seed);
  }
  
  bool initialize(Observations& observations, Terminations& terminations) override {
    // Sample new initial state and check if it is within the terminal set.
    sample_initial_state();
    // Initialize joint commands
    command_ = (usingPidController_) ? simulation_->getJointPositions() : Eigen::VectorXd::Zero(Nj);
    // Write the new observations resulting from the initialization.
    fill_observations(observations);
    // Check if the initial state sampling resulted in a valid state.
    terminations.back() = check_for_termination();
    // Signal that the initialization is valid.
    // NOTE: Invalid initializations are not possible in this implementation.
    return true;
  }
  
  bool transition(
      const Actions& actions,
      Observations& observations,
      Rewards& rewards,
      Terminations& terminations,
      Metrics& metrics) override {
    if (!actions.allFinite()) {
      NWARNING("[" << namescope() << "]: Actions are invalid: " << actions);
      stepIsValid_ = false;
    }
    // Action static transformation
    extract_commands(actions);
    // Step the physics simulation
    step_simulation();
    // Compute reward/cost and termination conditions
    if (stepIsValid_) {
      compute_rewards(rewards);
      terminations.back() = check_for_termination();
    } else {
      rewards.setZero();
      terminations.back() = {0, Termination::Type::InvalidState};
      auto ns = namescope();
      NWARNING("[" << ns << "]: Invalid state detected!");
      NWARNING("[" << ns << "]: qj: " << simulation_->getJointPositions().transpose());
      NWARNING("[" << ns << "]: W_r_WB: " << simulation_->getPositionWorldToBaseInWorldFrame().transpose());
      NWARNING("[" << ns << "]: W_r_WF:\n" << simulation_->getPositionWorldToFootInWorldFrame().transpose());
    }
    // Write the new observations resulting from the transition.
    fill_observations(observations);
    // Record performance metrics
    const auto& dq_j = simulation_->getJointVelocities();
    const auto& tau_j = simulation_->getJointTorques();
    metrics["joint_power"] = static_cast<Scalar>(dq_j.transpose()*tau_j);
    metrics["action_norm"] = static_cast<Scalar>(command_.norm());
    // Optional logging
    if (logger_) { update_state_logging(); }
    // Indicate if mdp transition is valid
    return stepIsValid_;
  }
  
  std::string info() const override {
    std::stringstream ss;
    ss << "\nW_r_WB_z: " << simulation_->getPositionWorldToBaseInWorldFrame().z();
    ss << "\nW_v_WB_z: " << simulation_->getLinearVelocityWorldToBaseInWorldFrame().z();
    ss << "\nq_j: " << simulation_->getJointPositions().transpose();
    ss << "\ndq_j: " << simulation_->getJointVelocities().transpose();
    ss << "\ntau_j: " << simulation_->getJointTorques().transpose();
    return ss.str();
  }

private:
  
  inline void sample_initial_state() {
    // Sample goal position
    W_r_WG_ << 0.0, 0.0, 1.2 + prng_.sampleUniform(-0.5, 0.7)*goalNoiseFactor_;
    // Sample initial base position
    Position r_WB(0.0, 0.0, 0.8 + prng_.sampleUniform(-0.3, 0.3)*resetNoiseFactor_);
    // Sample initial foot position
    Position r_BF = simulation_->getNominalFootPositionInBaseFrame();
    const double r_xz = prng_.sampleUniform(0.1, 0.45)*resetNoiseFactor_;
    const double theta_xz = prng_.sampleUniform(-M_PI_4, M_PI_4)*resetNoiseFactor_;
    r_BF.x() = r_xz*std::sin(theta_xz);
    r_BF.z() = -r_xz*std::cos(theta_xz);
    Position r_WF = r_WB + r_BF;
    r_WF.z() = std::max(0.025, r_WF.z());
    r_BF = r_WF - r_WB;
    // Compute initial joint configuration using sampled foot position
    auto qj = simulation_->foot_inverse_kinematics(r_BF);
    // Sample initial base and joint velocities
    Position v_WB(0.0, 0.0, prng_.sampleUniform(-2.0, 2.0)*resetNoiseFactor_);
    Eigen::VectorXd dqj = Eigen::VectorXd::Zero(CaplerSimulation::Nj);
    dqj(0) += prng_.sampleUniform(-3.0, 3.0)*resetNoiseFactor_;
    dqj(1) += prng_.sampleUniform(-3.0, 3.0)*resetNoiseFactor_;
    // Initial state
    Eigen::VectorXd u = Eigen::VectorXd::Zero(CaplerSimulation::Nu);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(CaplerSimulation::Nq);
    q << r_WB.z(), qj;
    u << v_WB.z(), dqj;
    simulation_->reset(q, u);
    if (isVerbose()) {
      auto ns = namescope();
      NNOTIFY("[" << ns << "]: Sampling new initial state!");
      NINFO("[" << ns << "]: Initial State: r_WB: " << r_WB.transpose());
      NINFO("[" << ns << "]: Initial State: r_BF: " << r_BF.transpose());
      NINFO("[" << ns << "]: Initial State: qj: " << qj.transpose());
    }
  }
  
  inline void extract_commands(const Actions& actions) {
    // Local references to actions
    using Index = CaplerActions<Scalar>::Index;
    const auto commands = actions[Index::Commands].asFlat();
    // Extract joint commands from actions
    command_ = (commands.cwiseProduct(actionScale_) + actionOffset_).template cast<double>();
  }
  
  inline void step_simulation() {
    // Advance the physics according to the configured control decimation
    const auto decimation = static_cast<size_t>(this->time_step() / simulation_->getTimeStep());
    for (size_t i = 0; i < decimation; i++) {
      // Advance the physics simulation by a single dynamics time-step
      stepIsValid_ = simulation_->step(command_);
    }
  }
  
  inline void compute_rewards(Rewards& rewards) {
    // Constituent costs
    Scalar baseHeightCost = 0;
    Scalar jointVelocityCost = 0;
    Scalar jointTorqueCost = 0;
    Scalar footSlipCost = 0;
    Scalar footSupportCost = 0;
    Scalar actionCost = 0;
    // Initialize reward terms and respective weights
    VectorX weights = VectorX::Constant(this->tasks().size(), 1.0);
    rewards.setZero();
    // Set reward weights
    weights(1) = 1.0; // Base height
    weights(2) = 0.001; // Joint velocity
    weights(3) = 0.0001; // Joint torque
    weights(4) = 0.1; // Foot slip
    weights(5) = 1.0; // Foot support
    weights(6) = 0.0; // Action rate
    // Local references
    const auto dt = this->time_step();
    const auto gamma = this->discount_factor();
    const auto& W_r_WB = simulation_->getPositionWorldToBaseInWorldFrame();
    const auto& W_r_WF = simulation_->getPositionWorldToFootInWorldFrame();
    const auto& W_v_WF = simulation_->getLinearVelocityWorldToFootInWorldFrame();
    const auto c_F = simulation_->getFootContact();
    auto dq_j = simulation_->getJointVelocities();
    auto tau_j = simulation_->getJointTorques();
    // Base
    const double zb_error = W_r_WG_.z() - W_r_WB.z();
    baseHeightCost = -static_cast<Scalar>(zb_error*zb_error);
    // Joints
    dq_j = ::noesis::math::deadzone(dq_j, 20.0);
    jointVelocityCost = -static_cast<Scalar>(dq_j.squaredNorm());
    tau_j = ::noesis::math::deadzone(tau_j, 50.0);
    jointTorqueCost = -static_cast<Scalar>(tau_j.squaredNorm());
    // Feet
    footSlipCost = -static_cast<Scalar>(c_F) * static_cast<Scalar>(W_v_WF.head(2).norm());
    footSupportCost = -static_cast<Scalar>(c_F) * static_cast<Scalar>(W_r_WF.x()*W_r_WF.x());
    // Action cost
    actionCost = -static_cast<Scalar>(command_.transpose() * R_ * command_);
    // Fill rewards
    auto rew = rewards.asFlat();
    rew(1) = baseHeightCost;
    rew(2) = jointVelocityCost;
    rew(3) = jointTorqueCost;
    rew(4) = footSlipCost;
    rew(5) = footSupportCost;
    rew(6) = actionCost;
    // Total (weighted sum)
    rew(0) = rew.cwiseProduct(weights).sum();
    // Optional logging of cost/reward terms
    if(logger_) {
      logger_->appendScalar("Rewards/base_height", baseHeightCost);
      logger_->appendScalar("Rewards/joint_velocity", jointVelocityCost);
      logger_->appendScalar("Rewards/joint_torque", jointTorqueCost);
      logger_->appendScalar("Rewards/foot_slip", footSlipCost);
      logger_->appendScalar("Rewards/foot_support", footSupportCost);
      logger_->appendScalar("Rewards/action", actionCost);
      logger_->appendScalar("Rewards/total", rew(0));
      logger_->appendScalar("Rewards/total_normalized", rew(0)*(1-gamma));
    }
    // Scale all costs according to the discount horizon
    rew *= (1-gamma);
  }
  
  inline Termination check_for_termination() {
    // TC1: Check if base has active contacts.
    if (simulation_->getBaseContacts() != 0) { return {static_cast<Scalar>(terminalValue_), Termination::Type::TerminalState, 1}; }
    // TC2: Check if any of the legs have active contacts.
    if (simulation_->getLegContacts() != 0) { return {static_cast<Scalar>(terminalValue_), Termination::Type::TerminalState, 2}; }
    // TC3: Check for knee over-extension.
    const auto dq_j = simulation_->getJointPositions();
    if (dq_j(1) < -0.2) { return {static_cast<Scalar>(terminalValue_), Termination::Type::TerminalState, 3}; }
    // Fall-through for inactive termination conditions.
    return Termination();
  }
  
  inline void fill_observations(Observations& observations) {
    // Extract state measurements
    observations_from_kinematics(observations);
    // Noise injection
    add_observations_noise(observations);
    // Local references to observations
    using Index = CaplerObservations<Scalar>::Index;
    auto& target = observations[Index::Target][0];
    auto motion = observations[Index::Motion].asFlat();
    auto history = observations[Index::History].asFlat();
    // Static scaling + offset
    target = (target - stateOffset_(0)) * stateScale_(0);
    motion = (motion - stateOffset_.segment<6>(1)).cwiseProduct(stateScale_.segment<6>(1));
    history = (history - stateOffset_.segment<Nj>(7)).cwiseProduct(stateScale_.segment<Nj>(7));
    // Check for invalid values
    if (!observations.allFinite()) {
      NWARNING("[" << namescope() << "]: Observations are invalid: " << observations);
      stepIsValid_ = false;
    }
  }
  
  inline void observations_from_kinematics(Observations& observations) {
    // Local references to internal system state
    const auto& q = simulation_->getGeneralizedCoordinates();
    const auto& u = simulation_->getGeneralizedVelocities();
    const auto& x = command_;
    // Local references to observations
    using Index = CaplerObservations<Scalar>::Index;
    auto& target = observations[Index::Target][0];
    auto motion = observations[Index::Motion].asFlat();
    auto history = observations[Index::History].asFlat();
    // References
    target = static_cast<Scalar>(W_r_WG_.z());
    // Measurements
    motion(0) = static_cast<Scalar>(q(0));
    motion(1) = static_cast<Scalar>(u(0));
    motion.template segment<Nj>(2) = q.tail<Nj>().template cast<Scalar>();
    motion.template segment<Nj>(4) = u.tail<Nj>().template cast<Scalar>();
    history.template segment<Nj>(0) = x.template cast<Scalar>();
  }
  
  inline void add_observations_noise(Observations& observations) {
    using Index = CaplerObservations<Scalar>::Index;
    auto& target = observations[Index::Target][0];
    auto motion = observations[Index::Motion].asFlat();
    auto history = observations[Index::History].asFlat();
    // Add Gaussian noise according to the configured standard-deviations for the relevant quantities
    target += stateNoise_(0) * prng_.sampleStandardNormal() * stateNoiseFactor_;
    for (int i = 0; i < motion.size(); ++i) { motion(i) += stateNoise_(1 + i) * prng_.sampleStandardNormal() * stateNoiseFactor_; }
    for (int i = 0; i < history.size(); ++i) { history(i) += stateNoise_(7 + i) * prng_.sampleStandardNormal() * stateNoiseFactor_; }
  }
  
  void update_state_logging() {
    // Base
    const auto& W_r_WB = simulation_->getPositionWorldToBaseInWorldFrame();
    const auto& W_v_WB = simulation_->getLinearVelocityWorldToBaseInWorldFrame();
    // Joints
    const auto& q_j = simulation_->getJointPositions();
    const auto& dq_j = simulation_->getJointVelocities();
    const auto& tau_j = simulation_->getJointTorques();
    // Foot
    const auto& W_r_WF = simulation_->getPositionWorldToFootInWorldFrame();
    const auto& W_v_WF = simulation_->getLinearVelocityWorldToFootInWorldFrame();
    const auto c_F = simulation_->getFootContact();
    // Append base measurements
    logger_->appendScalar("State/Base/position_z", W_r_WB.z());
    logger_->appendScalar("State/Base/linear_velocity_z", W_v_WB.z());
    // Append joint measurements
    std::vector<std::string> joints = {"HFE", "KFE"};
    for (int j = 0; j < static_cast<int>(joints.size()); ++j) {
      logging::append_joint(*logger_, "State/" + joints[j], command_(j), q_j(j), dq_j(j), tau_j(j));
    }
    // Append foot measurements
    logging::append_foot(*logger_, "State/FOOT", W_r_WF, W_v_WF, c_F);
  }

private:
  // Random number generation
  noesis::math::RandomNumberGenerator<double> prng_;
  // Logging
  std::unique_ptr<noesis::log::TensorBoardLogger> logger_{nullptr};
  // Simulation
  std::unique_ptr<CaplerSimulation> simulation_{nullptr};
  // Observation and action normalization
  VectorX stateScale_;
  VectorX stateOffset_;
  VectorX stateNoise_;
  VectorX actionScale_;
  VectorX actionOffset_;
  // Action buffers
  Eigen::VectorXd command_{Eigen::VectorXd::Zero(Nj)};
  Eigen::MatrixXd R_{Eigen::MatrixXd::Identity(Nj,Nj)};
  // Goal Position
  Position W_r_WG_{Position::Zero()};
  // Cost and initial state hyper-parameters
  double terminalValue_{-5.0};
  // Cost and initial state hyper-parameters
  double randomizationFactor_{0.0};
  double resetNoiseFactor_{0.0};
  double stateNoiseFactor_{0.0};
  double goalNoiseFactor_{0.0};
  // Internals
  bool stepIsValid_{true};
  bool usingPidController_{true};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_CAPLER_ENVIRONMENT_HPP_

/* EOF */
