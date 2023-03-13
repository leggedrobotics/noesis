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
#ifndef NOESIS_GYM_ENVS_MUJOCO_WALKER2D_WALKER2D_ENVIRONMENT_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_WALKER2D_WALKER2D_ENVIRONMENT_HPP_

// Noesis
#include <noesis/framework/utils/macros.hpp>
#include <noesis/framework/log/tensorboard.hpp>

// Gym
#include "noesis/gym/core/Environment.hpp"
#include <noesis/gym/envs/mujoco/common/visualizer.hpp>
#include "noesis/gym/envs/mujoco/common/simulation.hpp"
#include "noesis/gym/envs/mujoco/walker2d/Walker2dObservations.hpp"
#include "noesis/gym/envs/mujoco/walker2d/Walker2dActions.hpp"

namespace noesis {
namespace gym {

/*!
 * @brief Implementation of the mujoco walker2d environment.
 * Goal: Make a two-dimensional bipedal robot walk forward as fast as possible
 * Reference: https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v3.py
 * Generalized co-ordinates: base pose (cartesian co-ordinates, orientation), joint co-ordinates
 * Generalized velocities: base velocity, joint velocities
 * State : Generalized coordinates, Generalized velocities
 * Observation: Generalized coordinates (except for absolute X position), Generalized velocities, COM-based body inertia, COM-based velocity, Actuator Forces, COM-based external force
 * Action: control signals for the actuators
 */
class Walker2dEnvironment final : public noesis::gym::Environment<float>
{
public:
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

  // Constants
  static constexpr size_t BasePoseDim = 3;
  static constexpr size_t BaseVelocityDim = 3;
  static constexpr size_t JointsDim = 6;
  static constexpr size_t StateDim = BasePoseDim + JointsDim + BaseVelocityDim + JointsDim;
  static constexpr size_t InputDim = JointsDim;

  /*
   * Instantiation
   */

  explicit Walker2dEnvironment(
    size_t control_decimation = 4,
    double discount_factor = 0.995,
    double reset_noise_factor = 5e-3,
    bool enable_logging = false,
    bool visualize = false,
    const std::string &name = "Walker2d",
    const std::string &scope = "/",
    bool verbose = false
  ) :
    Base(
      Walker2dObservations<Scalar>().spec(),
      Walker2dActions<Scalar>().spec(),
      {"total", "foward_velocity", "healthy", "control_cost"},
      name,
      scope,
      verbose
    ),
    resetNoiseFactor_(reset_noise_factor),
    controlDecimation_(control_decimation)
  {
    NWARNING_IF(isVerbose(), "[Walker2dEnv]: New instance at: " << std::hex << this);
    // Simulation
    MujocoSimulationConfig config;
    // Set the default MJCF file to load
    std::string modelFile = mujoco::assets() + "/benchmarks/walker2d.xml";
    NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
    config.model_file = modelFile;
    simulation_ = std::make_unique<MujocoSimulation>(config);
    // Get default state
    defaultState_ << simulation_->state().getGeneralizedCoordinates(), simulation_->state().getGeneralizedVelocities();
    // Default parameters
    setTimeStep(static_cast<Scalar>(simulation_->timestep()) * controlDecimation_);
    setMaxSteps(std::numeric_limits<size_t>::max());
    setDiscountFactor(discount_factor);
    // Visualizer
    if (visualize)
    {
      MujocoVisualizerConfig config;
      config.real_time_factor = 1.0;
      config.use_render_thread = true;
      visualization_ = std::make_unique<MujocoVisualizer>(config);
      visualization_->setSimulation(simulation_.get());
      visualization_->launch();
    }
    // Initialize action
    command_ = Eigen::VectorXd::Zero(InputDim);
    // Define performance metrics
    metrics().push_back("x_position");
    metrics().push_back("x_velocity");
    // TensorBoard Logging
    if (enable_logging)
    {
      logger_ = std::make_unique<noesis::log::TensorBoardLogger>("", "policy", "policy", "", false);
      // Reward/cost terms
      logger_->addLoggingSignal("Rewards/forward_velocity");
      logger_->addLoggingSignal("Rewards/healthy");
      logger_->addLoggingSignal("Rewards/control_cost");
      logger_->addLoggingSignal("Rewards/total");
      logger_->startup();
    }
  }

  ~Walker2dEnvironment() final = default;

  /*
   * Configurations
   */

  /*
   * Properties
   */

  const MujocoSimulation &simulation() const
  { return *simulation_; }

  MujocoSimulation &simulation()
  { return *simulation_; }

  MujocoVisualizer *visualization()
  { return visualization_.get(); }

  /*
   * Operations
   */

  void randomize(int seed) override
  {
    // No simulation domain randomization
    prng_.seed(seed);
  }

  bool initialize(Observations &observations, Terminations &terminations) override
  {
    // Sample new initial state and check if it is within the terminal set.
    sample_initial_state();
    // Initialize joint commands
    command_ = Eigen::VectorXd::Zero(InputDim);
    // Write the new observations resulting from the initialization.
    fill_observations(observations);
    // Check if the initial state sampling resulted in a valid state.
    terminations.back() = check_for_termination();
    // Signal that the initialization is valid.
    // NOTE: Invalid initializations are not possible in this implementation.
    return true;
  }

  bool
  transition(const Actions &actions, Observations &observations, Rewards &rewards, Terminations &terminations) override
  {
    if (!actions.allFinite())
    {
      NWARNING("[" << namescope() << "]: Actions are invalid: " << actions);
      stepIsValid_ = false;
    }
    // Action static transformation
    extract_commands(actions);
    // Get the x-position before the step
    xPositionPrev_ = simulation_->state().getGeneralizedCoordinates()(0);
    // Step the physics simulation
    step_simulation();
    // Compute reward/cost and termination conditions
    if (stepIsValid_)
    {
      compute_rewards(rewards);
      terminations.back() = check_for_termination();
    } else
    {
      rewards.setZero();
      terminations.back() = {0, Termination::Type::InvalidState};
      auto ns = namescope();
      NWARNING("[" << ns << "]: Invalid state detected!");
      NWARNING("[" << ns << "]: q: " << simulation_->state().getGeneralizedCoordinates().transpose());
      NWARNING("[" << ns << "]: u: " << simulation_->state().getGeneralizedVelocities().transpose());
    }
    // Write the new observations resulting from the transition.
    fill_observations(observations);
    // Indicate if mdp transition is valid
    return stepIsValid_;
  }

  void record(Metrics &metrics) override
  {
    const auto x_position = simulation_->state().getGeneralizedCoordinates()(0);
    const auto x_velocity = simulation_->state().getGeneralizedVelocities()(0);
    metrics[0] = static_cast<Scalar>(x_position);
    metrics[1] = static_cast<Scalar>(x_velocity);
  }

  std::string info() const override
  {
    std::stringstream ss;
    ss << "\nq: " << simulation_->state().getGeneralizedCoordinates().transpose();
    ss << "\nu: " << simulation_->state().getGeneralizedVelocities().transpose();
    ss << "\ntau_j: " << simulation_->actuation().getJointEffort().transpose();
    return ss.str();
  }

private:

  inline void sample_initial_state()
  {
    // Constants
    const int Nq = BasePoseDim + JointsDim;
    const int Nu = BaseVelocityDim + JointsDim;
    // Initial state
    Eigen::VectorXd state = Eigen::VectorXd::Zero(Nq + Nu);
    // Sample initial foot position
    // Add uniform noise to default generalized coordinates
    for (size_t j = 0; j < Nq + Nu; ++j) {
      state(j) = defaultState_(j) + prng_.sampleUniform(-resetNoiseFactor_, resetNoiseFactor_);
    }
    simulation_->reset(state.head(Nq), state.tail(Nu));

    if (isVerbose())
    {
      auto ns = namescope();
      NNOTIFY("[" << ns << "]: Sampling new initial state!");
      NINFO("[" << ns << "]: Initial State: q: " << state.head(Nq).transpose());
      NINFO("[" << ns << "]: Initial State: u: " << state.tail(Nu).transpose());
    }
  }

  inline void extract_commands(const Actions &actions)
  {
    // Local references to actions
    using Index = Walker2dActions<Scalar>::Index;
    const auto commands = actions[Index::Ctrl].asFlat();
    // Extract joint commands from actions
    command_ = commands.template cast<double>();
  }

  inline void step_simulation()
  {
    // Advance the physics according to the configured control decimation
    for (size_t i = 0; i < controlDecimation_; i++)
    {
      // Advance the physics simulation by a single dynamics time-step
      stepIsValid_ = simulation_->step(command_);
    }
  }

  inline void compute_rewards(Rewards &rewards)
  {
    // Constituent costs
    Scalar forwardVelcoityCost = 0;
    Scalar healthyReward = 0;
    Scalar controlCost = 0;
    // Initialize reward terms and respective weights
    VectorX weights = VectorX::Ones(4);
    // Set reward weights
    weights(1) = 1.0; // forward velocity
    weights(2) = 1.0; // healthy
    weights(3) = 1e-3; // control cost
    // foward velcoty reward term
    const auto dt = this->time_step();
    const Scalar xPositionNow = simulation_->state().getGeneralizedCoordinates()(0);
    forwardVelcoityCost = (xPositionNow - xPositionPrev_) / dt;
    // healthy reward term
    healthyReward = 1.0;
    // control cost term
    const auto& jointTorques = simulation_->actuation().getJointEffort();
    controlCost = -jointTorques.squaredNorm();

    // Local references
    const auto gamma = this->discount_factor();
    // Fill rewards
    VectorX rew = VectorX::Zero(4);
    rew(1) = forwardVelcoityCost;
    rew(2) = healthyReward;
    rew(3) = controlCost;
    // Total (weighted sum)
    rew(0) = rew.cwiseProduct(weights).sum();
    // Assign the evaluated reward to buffer
    rewards.asFlat()(0) = rew(0);
    // Optional logging of cost/reward terms
    if (logger_)
    {
      logger_->appendScalar("Rewards/forward_velcoity", forwardVelcoityCost);
      logger_->appendScalar("Rewards/healthy", healthyReward);
      logger_->appendScalar("Rewards/control_cost", controlCost);
      logger_->appendScalar("Rewards/total", rew(0));
    }
  }

  inline Termination check_for_termination()
  {
    Termination termination;
    // Retrieve system's state
    const auto& q = simulation_->state().getGeneralizedCoordinates();
    // TC1: base height falls below certain range
    if (q(1) <= 0.8 || q(1) >= 2.0) {
      termination = {static_cast<Scalar>(0), Termination::Type::TerminalState};
    }
    // TC2: orientation of the body
    if (q(2) <= -1.0 || q(2) >= 1.0) {
      termination = {static_cast<Scalar>(0), Termination::Type::TerminalState};
    }
    // Return final termination condition
    return termination;
  }

  inline void fill_observations(Observations &observations)
  {
    // Extract state measurements
    observations_from_kinematics(observations);
    // Check for invalid values
    if (!observations.allFinite())
    {
      NWARNING("[" << namescope() << "]: Observations are invalid: " << observations);
      stepIsValid_ = false;
    }
  }

  inline void observations_from_kinematics(Observations &observations)
  {
    // Local references to internal system state
    const auto &q = simulation_->state().getGeneralizedCoordinates();
    const auto &u = simulation_->state().getGeneralizedVelocities();
    // Local references to observations
    using Index = Walker2dObservations<Scalar>::Index;
    // generalized coordinates besides the x-y positions
    observations[Index::GenCoord].asFlat() = q.tail(BasePoseDim + JointsDim - 1).template cast<Scalar>();
    // generalized velocities
    observations[Index::GenVel].asFlat() = u.cwiseMin(10.0).cwiseMax(-10.0).template cast<Scalar>();
  }


private:
  // Random number generation
  noesis::math::RandomNumberGenerator<Scalar> prng_;
  // Logging
  std::unique_ptr<noesis::log::TensorBoardLogger> logger_{nullptr};
  // Simulation
  std::unique_ptr<MujocoSimulation> simulation_{nullptr};
  std::unique_ptr<MujocoVisualizer> visualization_{nullptr};
  // Dynamics buffers
  Eigen::VectorXd defaultState_{Eigen::VectorXd::Zero(StateDim)};
  Eigen::VectorXd command_{Eigen::VectorXd::Zero(InputDim)};
  // Hyperaparameters
  double resetNoiseFactor_{0.1};
  int controlDecimation_{4};
  // Internals
  bool stepIsValid_{true};
  Scalar xPositionPrev_{0.0};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_WALKER2D_WALKER2D_ENVIRONMENT_HPP_

/* EOF */
