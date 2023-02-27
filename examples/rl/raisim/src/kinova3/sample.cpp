/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/noesis.hpp>

// Environment
#include "mdp.hpp"

int main(int argc, char** argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  
  // Definitions
  using Logger = noesis::log::TensorBoardLogger;
  using Environment = noesis::gym::Kinova3Environment;
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
  
  // 1. Create a vectorized environment
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
    nullptr,
    true,
    "Environment",
    scope,
    verbose
  );
  environment.configure();
  const auto& sim = environment.simulation();
  
  /*
   * Introspection
   */
  
  NINFO("[Sample]: Sim: Limits: Max Joint Velocities:\n" << sim.getMaxJointVelocities())
  NINFO("[Sample]: Sim: Limits: Max Joint Torques:\n" << sim.getMaxJointTorques())
  NINFO("[Sample]: Sim: PD: P-gains:\n" << sim.getJointPGains())
  NINFO("[Sample]: Sim: PD: D-gains:\n" << sim.getJointDGains())

  /*
   * Experiment execution
   */
  
  // Configure test parameters
  const size_t epnum = 30;
  const int seed = 0;
  
  // Test the policy over a fixed number of episodes
  NNOTIFY("[Sample]: Starting run ...")
  environment.seed(seed);
  environment.visualization().startRecording(noesis::logpath() + "/videos", "test");
  // 1. Test sampling of initial states
  environment.actions().setZero();
  for (size_t t = 0; t < epnum * environment.max_steps(); ++t) {
    if (t % 200 == 0) { environment.reset(); }
    environment.step();
  }
  // 2. Test joint references
  const auto dt = environment.time_step();
  for (int j = 0; j < 7; ++j) {
    environment.reset();
    environment.actions().setZero();
    for (size_t t = 0; t < 3.0 * environment.max_steps(); ++t) {
      auto action = environment.actions()[0].asFlat();
      action(j) = 1.0 * std::sin(2.0 * M_PI * 0.2 * t * dt);
      environment.step();
    }
  }
  environment.visualization().stopRecording();
  NNOTIFY("[Sample]: Run completed.")

  // Success
  return 0;
}

/* EOF */
