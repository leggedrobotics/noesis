/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Ennio Filicicchia
 * @email     efilicicc@student.ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_KINOVA3_ENVIRONMENT_HPP_
#define NOESIS_GYM_ENVS_RAISIM_KINOVA3_ENVIRONMENT_HPP_

// C/C++
#include <algorithm>

// Noesis
#include <noesis/framework/utils/macros.hpp>
#include <noesis/gym/core/Environment.hpp>
#include <noesis/gym/envs/raisim/common/logging.hpp>
#include <noesis/gym/envs/raisim/common/filters.hpp>
#include <noesis/gym/envs/raisim/kinova3/Kinova3Simulation.hpp>
#include <noesis/gym/envs/raisim/kinova3/Kinova3Visualizer.hpp>

namespace noesis {
namespace gym {

class Kinova3Observations : public noesis::mdp::Observations<float>
{
public:
  explicit Kinova3Observations(size_t history_size=0, const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Observations<float>(scope, time_size, batch_size) {
    this->addTensor("targets", {7}); // 3x1 xyz [m], 4x1 quaternion [q]
    this->addTensor("positions", {7}); // 7x1 joint angles [rad]
    this->addTensor("velocities", {7}); // 7x1 joint velocities [rad/s]
    if (history_size > 0) { this->addTensor("history", {history_size}); } // // Nh x 1 observation history (can be used for anything)
  }
};

class Kinova3Actions : public noesis::mdp::Actions<float>
{
public:
  explicit Kinova3Actions(const std::string& scope="", size_t time_size=0, size_t batch_size=0):
    noesis::mdp::Actions<float>(scope, time_size, batch_size) {
    this->addTensor("commands", {7}); // 7x1 joint commands [Nm (torque), rad (pid)]
  }
};

class Kinova3Environment final: public noesis::gym::Environment<float>
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
  using Logger = noesis::log::TensorBoardLogger;
  
  // Constants
  static constexpr int Nj = 7;
  static constexpr int Nh = 3*Nj;
  
  /*
   * Instantiation
   */
  
  explicit Kinova3Environment(
      double time_step = 0.01,
      double time_limit = 2.0,
      double discount_factor = 0.995,
      double goal_noise_factor=0.0,
      double reset_noise_factor=0.0,
      double randomization_factor=0.0,
      double observations_noise_factor=0.0,
      bool use_pid_controller=true,
      bool use_simulator_pid=true,
      Logger* logger=nullptr,
      bool visualize=false,
      const std::string& name="Kinova3",
      const std::string& scope="/",
      bool verbose=false):
    Base(Kinova3Observations(Nh).spec(), Kinova3Actions().spec(),
      {"total",
        "position_error",
        "orientation_error",
        "linear_acceleration",
        "angular_acceleration",
        "joint_torque",
        "joint_velocity",
        "joint_position",
        "action_limits",
        "action_rate"},
      {"joints/power",
        "joints/arm/velocity_norm",
        "joints/arm/torque_norm",
        "joints/wrist/velocity_norm",
        "joints/wrist/torque_norm",
        "ee/position_error_norm",
        "ee/orientation_error_norm",
        "ee/linear_velocity_norm",
        "ee/linear_acceleration_norm",
        "ee/angular_velocity_norm",
        "ee/angular_acceleration_norm"},
      discount_factor, time_limit, time_step, 1u, 1u, name, scope, verbose),
    logger_(logger),
    observationsScale_(VectorX::Constant(Kinova3Observations(Nh).datum_size(), 1.0)),
    observationsOffset_(VectorX::Constant(Kinova3Observations(Nh).datum_size(), 0.0)),
    observationsNoise_(VectorX::Constant(Kinova3Observations(Nh).datum_size(), 0.0)),
    actionScale_(VectorX::Constant(Kinova3Actions().datum_size(), 1.0)),
    actionOffset_(VectorX::Constant(Kinova3Actions().datum_size(), 0.0)),
    randomizationFactor_(randomization_factor),
    resetNoiseFactor_(reset_noise_factor),
    observationsNoiseFactor_(observations_noise_factor),
    goalNoiseFactor_(goal_noise_factor)
  {
    NNOTIFY_IF(isVerbose(), "[Kinova3Environment]: New instance at: " << std::hex << this);
    // Simulation
    simulator_ = std::make_unique<Kinova3Simulation>(Kinova3Simulation::World::Empty);
    simulator_->setUsePidJointController(use_pid_controller);
    simulator_->setUseSimulatorPid(use_simulator_pid);
    // Visualizer
    if (visualize) {
      Kinova3VisConfig config;
      config.window_width = 1920;
      config.window_height = 1080;
      config.anti_aliasing = 8;
      config.real_time_factor = 1.0;
      config.use_rendering_thread = true;
      config.show_gui = true;
      config.show_goal_pose = true;
      config.show_goal_force = false;
      visualizer_ = std::make_unique<Kinova3Visualizer>(config, *simulator_);
      visualizer_->launch();
    }
    // Initialize action members
    const auto Na = this->actions()[0].asFlat().size();
    command_ = Eigen::VectorXd::Zero(Na);
    if (use_pid_controller) { command_.segment<Nj>(0) = simulator_->getNominalJointConfiguration(); }
    // Action filter
    commandFilter_ = std::make_unique<filters::FiniteDifferencesFilter>(Na);
    commandFilter_->initialize(command_);
    commandFiltered_ = commandFilter_->output();
    R_ = Eigen::MatrixXd::Identity(Na, Na);
    // We define the joint bias differently to the nominal
    qjOffset_ << 0.0, 0.0, 0.0, M_PI_2, 0.0, 0.0, -M_PI_2;
    // State transforms
    observationsScale_ <<
      VectorX::Constant(3, 1.0),  // target end-effector position
      VectorX::Constant(4, 1.0),  // target end-effector orientation
      VectorX::Constant(Nj, 1.0), // measured joint positions
      VectorX::Constant(Nj, 1.0), // measured joint velocities
      VectorX::Constant(Nh, 1.0); // history
    observationsOffset_ <<
      VectorX::Constant(3, 0.0),  // target end-effector position
      VectorX::Constant(4, 0.0),  // target end-effector orientation
      qjOffset_.cast<Scalar>(),   // measured joint positions
      VectorX::Constant(Nj, 0.0), // measured joint velocities
      VectorX::Constant(Nh, 0.0); // history
    observationsNoise_ <<
      VectorX::Constant(3, 0.0),   // target end-effector position
      VectorX::Constant(4, 0.0),   // target end-effector orientation
      VectorX::Constant(Nj, 0.01), // measured joint positions
      VectorX::Constant(Nj, 0.02), // measured joint velocities
      VectorX::Constant(Nh, 0.0);  // history
    // Action transforms
    const auto& tauMax = simulator_->getMaxJointTorques();
    if (use_pid_controller) {
      actionScale_.segment<Nj>(0).setConstant(M_PI_2);
      actionOffset_.segment<Nj>(0) = qjOffset_.cast<Scalar>();
      //actionScale_.segment<Nj>(Nj).setConstant(1.0);
      //actionOffset_.segment<Nj>(Nj).setZero();
      //actionScale_.segment<Nj>(2*Nj) = tauMax.cast<Scalar>();
      //actionOffset_.segment<Nj>(2*Nj).setZero();
    } else {
      actionScale_.segment<Nj>(0) = tauMax.cast<Scalar>();
      actionOffset_.segment<Nj>(0).setZero();
    }
    // TensorBoard Logging
    if (logger_) {
      logging::add_3d_vector(*logger_, this->name() + "/orientation_error");
      logging::add_3d_vector(*logger_, this->name() + "/position_error");
    }
  }
  
  ~Kinova3Environment() final = default;
  
  /*
   * Configurations
   */
  
  void setGoalNoiseFactor(double factor) {
    goalNoiseFactor_ = factor;
  }
  
  void setResetNoiseFactor(double factor) {
    resetNoiseFactor_ = factor;
  }
  
  void setRandomizationFactor(double factor) {
    randomizationFactor_ = factor;
  }
  
  void setObservationNoiseFactor(double factor) {
    observationsNoiseFactor_ = factor;
  }
  
  void setGoal(const RotationMatrix& R, const Position& r) {
    R_G_ = transformFrameFromWorldToEndEffector(R);
    r_G_ = r;
  }
  
  void setGoal(const Pose& pose) {
    setGoal(pose.first, pose.second);
  }
  
  /*
   * Properties
   */

  const auto& rng() const { return prng_; }
  
  auto& rng() { return prng_; }
  
  const Kinova3Simulation& simulation() const { return *simulator_; }
  
  Kinova3Simulation& simulation() { return *simulator_; }
  
  const Kinova3Visualizer& visualization() const { return *visualizer_; }
  
  Kinova3Visualizer& visualization() { return *visualizer_; }
  
  const filters::Filter& commandFilter() const { return *commandFilter_; }
  
  filters::Filter& commandFilter() { return *commandFilter_; }
  
  std::pair<RotationMatrix, Position> goal() const { return {R_G_, r_G_}; }
  
  const Vector3& orientationError() const { return perfOrientationError_; }
  
  const Vector3& positionError() const { return perfPositionError_; }
  
  /*
   * Operations
   */

  inline void updateObservationsFrom(
      const RotationMatrix& R_error,
      const Position& r_error,
      const Eigen::VectorXd& qj,
      const Eigen::VectorXd& dqj,
      const Eigen::VectorXd& x,
      Observations& observations,
      bool add_noise = false) {
    // References
    auto targets = observations[0].asFlat();
    targets << r_error.cast<Scalar>(), math::rotation_matrix_to_quaternion(R_error).cast<Scalar>();
    targets = (targets - observationsOffset_.segment<7>(0)).cwiseProduct(observationsScale_.segment<7>(0));
    // Measurements
    auto positions = observations[1].asFlat();
    positions = qj.cast<Scalar>();
    if (add_noise) {
      for (int i = 0; i < Nj; ++i) { positions(i) += observationsNoise_(7 + i) * prng_.sampleStandardNormal() * observationsNoiseFactor_; }
    }
    positions = (positions - observationsOffset_.segment<Nj>(7)).cwiseProduct(observationsScale_.segment<Nj>(7));
    auto velocities = observations[2].asFlat();
    velocities = dqj.cast<Scalar>();
    if (add_noise) {
      for (int i = 0; i < Nj; ++i) { velocities(i) += observationsNoise_(7 + Nj + i) * prng_.sampleStandardNormal() * observationsNoiseFactor_; }
    }
    velocities = (velocities - observationsOffset_.segment<Nj>(7 + Nj)).cwiseProduct(observationsScale_.segment<Nj>(7 + Nj));
    // History
    if (Nh > 0) {
      auto history = observations[3].asFlat();
      history = x.cast<Scalar>();
      history = (history - observationsOffset_.segment<Nh>(7 + Nj + Nj)).cwiseProduct(observationsScale_.segment<Nh>(7 + Nj + Nj));
    }
  }
  
  inline Eigen::VectorXd jointCommandsFrom(const Actions& actions) const {
    // Local references to actions
    const auto action = actions[0].asFlat();
    // Extract joint commands from actions
    return (action.cwiseProduct(actionScale_) + actionOffset_).cast<double>();
  }
  
protected:
  
  /*
   * Environment API
   */
  
  void randomize(int seed) override {
    simulator_->randomize(seed, randomizationFactor_);
    prng_.seed(seed);
  }
  
  bool initialize(Observations& observations, Terminations& terminations) override {
    // Sample new initial state and check if it is within the terminal set.
    sampleInitialState();
    // Initialize joint commands
    command_.setZero();
    if (simulator_->isUsingPidController()) { command_.segment<Nj>(0) = simulator_->getJointPositions(); }
    // Initialize command filter
    commandFilter_->initialize(command_);
    commandFiltered_ = commandFilter_->output();
    if (resetNoiseFactor_ > 0) {
      auto& filterState = commandFilter_->state();
      for (int i = 0; i < filterState.size(); i++) { filterState(i) += prng_.sampleNormal(0.0, 0.1) * resetNoiseFactor_; }
    }
    // Write the new observations resulting from the initialization.
    computeObservations(observations);
    // Check if the initial state sampling resulted in a valid state.
    terminations.back() = checkTerminationConditions();
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
    // Capture raw action
    action_ = actions[0].asFlat().cast<double>();
    // Action to joint command mappings
    command_ = jointCommandsFrom(actions);
    // Step the physics simulation
    stepIsValid_ = stepSimulation();
    // Compute the step-wise computations only if the step was valid.
    if (stepIsValid_) {
      // Update action filter
      commandFiltered_ = commandFilter_->advance(command_);
      // Update tracking errors
      const auto& R_ee = simulator_->getOrientationWorldToEndEffector();
      const auto& r_ee = simulator_->getPositionWorldToEndEffectorInWorldFrame();
      perfOrientationError_ = math::quaternion_distance(R_ee, R_G_);
      perfPositionError_ = r_G_ - r_ee;
      // Write the new observations resulting from the transition.
      computeObservations(observations);
      // Compute the task reward/cost.
      computeTaskRewards(rewards);
      // Check if termination conditions have been activated.
      terminations.back() = checkTerminationConditions();
      // Record problem-specific performance metrics.
      updateMetrics(metrics);
      // Optional logging
      if (logger_) { updateLogging(); }
    } else {
      rewards.setZero();
      terminations.back() = {0, Termination::Type::InvalidState};
      auto ns = namescope();
      NWARNING("[" << ns << "]: Invalid state detected!");
      NWARNING("[" << ns << "]: qj: " << simulator_->getJointPositions().transpose());
      NWARNING("[" << ns << "]: W_r_ee:\n" << simulator_->getPositionWorldToEndEffectorInWorldFrame().transpose());
    }
    // Indicate if mdp transition is valid
    return stepIsValid_;
  }
  
  std::string info() const override {
    std::stringstream ss;
    ss << "\nq_j: " << simulator_->getJointPositions().transpose();
    ss << "\ndq_j: " << simulator_->getJointVelocities().transpose();
    ss << "\ntau_j: " << simulator_->getJointTorques().transpose();
    return ss.str();
  }
  
  inline RotationMatrix transformFrameFromWorldToEndEffector(const RotationMatrix& R) {
    return simulator_->transformFrameFromWorldToEndEffector(R);
  }
  
  inline RotationMatrix transformFrameFromWorldToEndEffector(const EulerRpy& rpy) {
    return transformFrameFromWorldToEndEffector(math::euler_angles_to_rotation_matrix(rpy));
  }
  
  inline RotationMatrix transformFrameFromWorldToEndEffector(double r, double p, double y) {
    return transformFrameFromWorldToEndEffector(math::euler_angles_to_rotation_matrix(EulerRpy(r, p, y)));
  }

private:
  
  /*
   * Internals
   */

  inline bool isInWorkspace(const Position& r) const {
    const auto r_ee = r.norm();
    const auto height = r.z();
    const auto radius = r.head<2>().norm();
    if (r_ee > 0.9) { return false; }
    if (height < 0.3 || height > 0.9) { return false; }
    if (radius < 0.3 || radius > 0.9) { return false; }
    return true;
  }

  inline void sampleInitialState() {
    // Sample initial arm state
    auto q0 = simulator_->getNominalJointConfiguration();
    auto u0 = simulator_->getGeneralizedVelocities();
    for (int i = 0; i < q0.size(); ++i) { q0(i) += resetNoiseFactor_ * prng_.sampleUniform(-M_PI/10.0, M_PI/10.0); }
    for (int i = 0; i < u0.size(); ++i) { u0(i) = resetNoiseFactor_ * prng_.sampleUniform(-1.0, 1.0); }
    simulator_->reset(q0, u0);
    // Sample goal position
    R_G_ = simulator_->getOrientationWorldToEndEffector();
    r_G_ = simulator_->getPositionWorldToEndEffectorInWorldFrame();
    if (goalNoiseFactor_ > 0.0) {
      do {
        const auto x = goalNoiseFactor_ * prng_.sampleUniform(-0.1, 0.1);
        const auto y = goalNoiseFactor_ * prng_.sampleUniform(-0.3, 0.3);
        const auto z = goalNoiseFactor_ * prng_.sampleUniform(-0.2, 0.2);
        r_G_ = Position(0.6, 0.0, 0.6) + Position(x, y, z);
        const auto roll = goalNoiseFactor_ * prng_.sampleUniform(-0.1, 0.1);
        auto pitch = goalNoiseFactor_ * prng_.sampleUniform(-0.1, 0.1);
        if (z < 0.5) { pitch = std::abs(pitch); }
        const auto yaw = goalNoiseFactor_ * prng_.sampleUniform(-0.1, 0.1);
        R_G_ = transformFrameFromWorldToEndEffector(roll, pitch, yaw);
      } while(!isInWorkspace(r_G_));
    }
    NNOTIFY_IF(isVerbose(), "[" << namescope() << "]: Sampling new initial state!"
      << "\nq = " << q0.transpose()
      << "\nu = " << u0.transpose());
  }

  inline bool stepSimulation() {
    bool stepIsValid = true;
    // Advance the physics according to the configured control decimation
    const auto decimation = static_cast<size_t>(this->time_step() / simulator_->getTimeStep());
    for (size_t i = 0; i < decimation; i++) {
      // Advance the physics simulation by a single dynamics time-step
      stepIsValid &= simulator_->step(command_);
      // Update the visualization if enabled
      if (visualizer_) {
        if (visualizer_->isEnabled()) {
          visualizer_->update(R_G_, r_G_);
        }
      }
    }
    return stepIsValid;
  }
  
  inline void computeTaskRewards(Rewards& rewards) {
    // Initialize reward terms and respective weights
    VectorX weights = VectorX::Constant(this->tasks().size(), 1.0);
    auto rew = rewards.asFlat();
    
    // Local constants
    const auto dt = this->time_step();
    const auto gamma = this->discount_factor();
    
    // Set reward weights
    weights(1) = 1.0; // Position error
    weights(2) = 1.0; // Orientation error
    weights(3) = 0.0; // Linear acceleration
    weights(4) = 0.0; // Angular acceleration
    weights(5) = 0.0; // Joint torque
    weights(6) = 0.0; // Joint velocity
    weights(7) = 0.0; // Joint position
    weights(8) = 0.0; // Action penalty
    weights(9) = 0.0; // Action rate
    
    // Penalize end-effector position error
    rew(1) = 1.0 - 5.0 * perfPositionError_.norm();
    
    // Penalize end-effector orientation error
    rew(2) = 1.0 - 5.0 * perfOrientationError_.norm();
  
    // Penalize end-effector linear accelerations
    const auto W_v_ee = simulator_->getLinearVelocityWorldToEndEffectorInWorldFrame();
    W_a_ee_ = (1.0/dt) * (W_v_ee - W_v_ee_);
    W_v_ee_ = W_v_ee;
    rew(3) = std::max(-1.0, 1.0 - 0.001 * W_a_ee_.squaredNorm() * dt*dt);
    
    // Penalize end-effector angular accelerations
    const auto W_omega_ee = simulator_->getAngularVelocityOfEndEffectorInWorldFrame();
    W_gamma_ee_ = (1.0/dt) * (W_omega_ee - W_omega_ee_);
    W_omega_ee_ = W_omega_ee;
    rew(4) = std::max(-1.0, 1.0 - 0.001 * W_gamma_ee_.squaredNorm() * dt*dt);
    
    // Penalize exceeding the maximum joint torque
    auto tau_j = simulator_->getJointTorques();
    tau_j = ::noesis::math::deadzone(tau_j, 55.0);
    rew(5) = std::max(-1.0, 1.0 - 0.0001 * tau_j.squaredNorm());
    
    // Penalize exceeding the maximum joint velocity
    auto dq_j = simulator_->getJointVelocities();
    dq_j = ::noesis::math::deadzone(dq_j, 2.0);
    rew(6) = std::max(-1.0, 1.0 - 0.001 * dq_j.squaredNorm());
    
    // Penalize deviation from the nominal joint configuration
    const auto& q_j = simulator_->getJointPositions();
    const auto& q_j_0 = simulator_->getNominalJointConfiguration();
    rew(7) = std::max(-1.0, 1.0 - 1.0 * (q_j_0 - q_j).squaredNorm());
    
    // Penalize exceeding the nominal reward limits of [-1, 1]
    const auto alpha = ::noesis::math::deadzone(action_, 1.0).norm();
    rew(8) = 1.0 - 1.0 * alpha;
    
    // Penalize high second-order action rates
    const double zeta = commandFiltered_.transpose() * R_ * commandFiltered_;
    rew(9) = 1.0 - 0.01 * zeta;
    
    // Compute the total reward (weighted sum)
    rew(0) = rew.dot(weights) * (1.0 - gamma);
  }
  
  inline Termination checkTerminationConditions() {
//   // TC1: Check if base has active contacts.
//   if (simulator_->getArmContacts() != 0) { return {terminalValue_, Termination::Type::TerminalState, 1}; }
//   // TC2: Check if any of the legs have active contacts.
//   if (simulator_->getWristContacts() != 0) { return {terminalValue_, Termination::Type::TerminalState, 2}; }
    // Fall through for uneventful transitions.
    return Termination();
  }
  
  inline void computeObservations(Observations& observations) {
    // Local references to internal system state
    const auto& R_ee = simulator_->getOrientationWorldToEndEffector();
    const auto& r_ee = simulator_->getPositionWorldToEndEffectorInWorldFrame();
    const auto& qj = simulator_->getJointPositions();
    const auto& dqj = simulator_->getJointVelocities();
    const auto& x = commandFilter_->state();
    // Extract observations measurements
    updateObservationsFrom((R_G_.transpose() * R_ee), (r_G_ - r_ee), qj, dqj, x, observations, true);
  }
  
  inline void updateMetrics(Metrics& metrics) {
    // Record problem-specific performance metrics.
    const auto& dq_j = simulator_->getJointVelocities();
    const auto& tau_j = simulator_->getJointTorques();
    metrics["joints/power"] = static_cast<Scalar>(dq_j.dot(tau_j));
    metrics["joints/arm/velocity_norm"] = static_cast<Scalar>(dq_j.head(4).norm());
    metrics["joints/arm/torque_norm"] = static_cast<Scalar>(tau_j.head(4).norm());
    metrics["joints/wrist/velocity_norm"] = static_cast<Scalar>(dq_j.tail<3>().norm());
    metrics["joints/wrist/torque_norm"] = static_cast<Scalar>(tau_j.tail<3>().norm());
    metrics["ee/position_error_norm"] = static_cast<Scalar>(perfPositionError_.norm());
    metrics["ee/orientation_error_norm"] = static_cast<Scalar>(perfOrientationError_.norm());
    metrics["ee/linear_velocity_norm"] = static_cast<Scalar>(W_v_ee_.norm());
    metrics["ee/linear_acceleration_norm"] = static_cast<Scalar>(W_a_ee_.norm());
    metrics["ee/angular_velocity_norm"] = static_cast<Scalar>(W_omega_ee_.norm());
    metrics["ee/angular_acceleration_norm"] = static_cast<Scalar>(W_gamma_ee_.norm());
  }
  
  inline void updateLogging() {
    // Append performance measurements
    logging::append_3d_vector(*logger_, this->name() + "/orientation_error", perfOrientationError_);
    logging::append_3d_vector(*logger_, this->name() + "/position_error", perfPositionError_);
  }

private:
  // Components
  noesis::math::RandomNumberGenerator<double> prng_;
  std::unique_ptr<Kinova3Simulation> simulator_{nullptr};
  std::unique_ptr<Kinova3Visualizer> visualizer_{nullptr};
  noesis::log::TensorBoardLogger* logger_{nullptr};
  // Observation and action normalization
  VectorX observationsScale_;
  VectorX observationsOffset_;
  VectorX observationsNoise_;
  VectorX actionScale_;
  VectorX actionOffset_;
  // Action buffers
  Eigen::VectorXd qjOffset_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd action_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd command_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd commandFiltered_{Eigen::VectorXd::Zero(Nj)};
  std::unique_ptr<filters::Filter> commandFilter_{nullptr};
  Eigen::MatrixXd R_{Eigen::MatrixXd::Identity(Nj, Nj)};
  // Tracking buffers
  Vector3 perfOrientationError_{Vector3::Zero()};
  Vector3 perfPositionError_{Vector3::Zero()};
  // Buffers
  LinearVelocity W_v_ee_{LinearVelocity::Zero()};
  AngularVelocity W_omega_ee_{AngularVelocity::Zero()};
  LinearVelocity W_a_ee_{LinearVelocity::Zero()};
  AngularVelocity W_gamma_ee_{AngularVelocity::Zero()};
  // Goals pose
  RotationMatrix R_G_{RotationMatrix::Identity()};
  Position r_G_{Position::Zero()};
  // Cost and initial observations hyper-parameters
  double goalNoiseFactor_{0.0};
  double resetNoiseFactor_{0.0};
  double randomizationFactor_{0.0};
  double observationsNoiseFactor_{0.0};
  // Internals
  Scalar terminalValue_{0.0};
  bool stepIsValid_{true};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_KINOVA3_ENVIRONMENT_HPP_

/* EOF */
