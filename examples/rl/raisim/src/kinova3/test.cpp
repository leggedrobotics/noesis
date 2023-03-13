/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/noesis.hpp>
#include <noesis/gym/core/Vector.hpp>
#include <noesis/rl/function/StochasticPolicy.hpp>
#include <noesis/gym/envs/raisim/common/trajectory.hpp>

// Environment
#include "mdp.hpp"

int main(int argc, char** argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  using namespace noesis::gym;
  
  // Definitions
  using Graph = noesis::core::Graph;
  using Logger = noesis::log::TensorBoardLogger;
  using Environment = noesis::gym::Kinova3Environment;
  using Policy = noesis::function::StochasticPolicy<typename Environment::Scalar>;
  using Termination = noesis::mdp::Termination<typename Environment::Termination>;
  
  // Initialize log paths and other noesis internals
  // NOTE: This creates the process name automatically from argv[0]
  noesis::init(argc, argv);
  
  /*
   * Experiment configuration
   */
  
  // Configurations
  const std::string scope = "Kinova3";
  const bool verbose = false;
  
  // Hyper-parameters
  const double time_step = 0.01;
  const double time_limit = 2.0;
  const double discount_factor = 0.995;
  const double goal_noise_factor = 1.0;
  const double reset_noise_factor = 1.0;
  const double randomization_factor = 1.0;
  const double observations_noise_factor = 1.0;
  const bool use_pid_controller = true;
  const bool use_simulator_pid = false;
  
  // 1. Define a TensorBoard logger
  auto logger = std::make_unique<Logger>("", "Kinova3", "Logger", scope, verbose);
  
  // 2. Create a graph
  auto graph = std::make_unique<Graph>(
    noesis::tf::SessionConfig(),
    "Graph",
    scope,
    verbose);
  
  // 3. Create a vectorized environment
  Environment environment(
    time_step,
    time_limit,
    discount_factor,
    goal_noise_factor,
    reset_noise_factor,
    randomization_factor,
    observations_noise_factor,
    use_pid_controller,
    use_simulator_pid,
    logger.get(),
    true,
    "Environment",
    scope,
    verbose
  );
  
  // 4. Create a generic policy
  Policy policy(
    environment.actions_spec(),
    environment.observations_spec(),
    scope + "/Agent/Observations",
    scope + "/Agent",
    "Policy",
    graph.get()
  );
  
  /*
   * Experiment execution
   */
  
  // Configure all objects based on the hyper-parameters we just loaded.
  environment.configure();
  
  // Set training experiment to load
  const std::string resourceDir = "/home/vassilios/.noesis/proc";
  const std::string experimentDir = "/noesis_rl_train_kinova3_demo/2021-03-18-21-29-21";
  
  // Create a runnable graph from the target Python script and initialize all dependents (i.e. the agent).
  const std::string graphDir = "/graphs/Graph";
  graph->loadFrom(resourceDir + experimentDir + graphDir + "/graph.pb");
  graph->startup();
  policy.initialize();

  // Restore an existing graph from checkpoints.
  graph->restoreFrom(resourceDir + experimentDir + graphDir + "/checkpoints/graph");
  
  // Configure test parameters
  const size_t epnum = 10;
  const int seed = 0;
  size_t resets = 0;
  size_t steps = 0;
  double position_error_sum = 0;
  double orientation_error_sum = 0;
  
  // Initiate video recording
  environment.visualization().startRecording(noesis::logpath() + "/videos", "test");
  
  // Test the policy over a fixed number of episodes
  NNOTIFY("[Test]: Starting run with sampled references ...")
  environment.seed(seed);
  environment.reset();
  for (int t = 0; t < epnum * environment.max_steps(); ++t) {
    policy.mode(environment.observations(), environment.actions());
    environment.step();
    if (static_cast<Termination::Type>(environment.terminations().back().type) == Termination::Type::TerminalState) {
      NWARNING("[Test]: Terminal state detected! Resetting ...")
      environment.reset();
      resets++;
    } else if (static_cast<Termination::Type>(environment.terminations().back().type) == Termination::Type::TimeOut) {
      NNOTIFY("[Test]: Episode timed-out. Resetting ...")
      environment.reset();
    } else {
      position_error_sum += environment.positionError().norm();
      orientation_error_sum += environment.orientationError().norm();
    }
    steps++;
  }
  NINFO("[Test]: Samples test completed.");
  NINFO("[Test]: Total steps: " << steps)
  NINFO("[Test]: Total resets: " << resets)
  NINFO("[Test]: Total success-rate: " << 100.0*static_cast<double>(steps-resets)/static_cast<double>(steps) << "%")
  NINFO("[Test]: Mean pos-error: " << position_error_sum/static_cast<double>(steps) )
  NINFO("[Test]: Mean ori-error: " << orientation_error_sum/static_cast<double>(steps))
  
  // Test the policy over on a single trajectory reference
  NNOTIFY("[Test]: Starting run with trajectory references ...")
  steps = 0;
  resets = 0;
  position_error_sum = 0;
  orientation_error_sum = 0;
  environment.setGoalNoiseFactor(0.0);
  environment.setResetNoiseFactor(0.0);
  environment.reset();
  const double scale = 0.2;
  Trajectory trajectory(environment.goal(), environment.time_step(), 4.0 * environment.time_limit(), scale);
  for (int t = 0; t < epnum * environment.max_steps(); ++t) {
    policy.mode(environment.observations(), environment.actions());
    environment.setGoal(RotationMatrix::Identity(), trajectory.sample());
    environment.step();
    if (static_cast<Termination::Type>(environment.terminations().back().type) == Termination::Type::TerminalState) {
      NWARNING("[Test]: Terminal state detected! Resetting ...")
      environment.reset();
      trajectory.reset();
      resets++;
    } else {
      position_error_sum += environment.positionError().norm();
      orientation_error_sum += environment.orientationError().norm();
    }
    steps++;
  }
  NINFO("[Test]: Trajectory test completed.");
  NINFO("[Test]: Total steps: " << steps)
  NINFO("[Test]: Total resets: " << resets)
  NINFO("[Test]: Total success-rate: " << 100.0*static_cast<double>(steps-resets)/static_cast<double>(steps) << "%")
  NINFO("[Test]: Mean pos-error: " << position_error_sum/static_cast<double>(steps) )
  NINFO("[Test]: Mean ori-error: " << orientation_error_sum/static_cast<double>(steps))
  
  // Terminate video recording
  environment.visualization().stopRecording();
  
  // Success
  return 0;
}

/* EOF */
