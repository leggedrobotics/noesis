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
#include <noesis/gym/envs/raisim/capler/CaplerEnvironment.hpp>
#include <noesis/gym/envs/raisim/capler/CaplerVisualizer.hpp>

int main(int argc, char** argv)
{
  noesis::init("noesis_gym_capler_example");
  
  // Configurations
  double time_step = 0.01;
  double time_limit = 5.0;
  double discount_factor = 0.995;
  double randomization_factor = 1.0;
  double reset_noise_factor = 0.0;
  double state_noise_factor = 0.0;
  double goal_noise_factor = 0.0;
  bool use_pid_controller = true;
  bool enable_logging = false;
  bool visualize = false;
  bool verbose = true;
  
  // Create a simple single-instance environment
  noesis::gym::CaplerEnvironment env(
    time_step,
    time_limit,
    discount_factor,
    randomization_factor,
    reset_noise_factor,
    state_noise_factor,
    goal_noise_factor,
    use_pid_controller,
    enable_logging,
    visualize, // NOTE: we will use an externally managed visualizer
    "Capler",
    "/Example",
    verbose
  );

  // Create an external visualizer
  noesis::gym::CaplerVisualizer vis(env.simulation());
  vis.launch();

  // Execute demo episode
  env.seed(0);
  env.reset();
  for (size_t t = 0; t < 10*100; ++t) {
    env.actions().setRandom();
    env.step();
  }

  // Print info and internal state of the instance
  NINFO(env);
  
  // Success
  return 0;
}

/* EOF */
