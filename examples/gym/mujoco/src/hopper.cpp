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

// Noesis Environments
#include <noesis/gym/envs/mujoco/hopper/HopperEnvironment.hpp>

int main(int argc, char** argv)
{
  noesis::init("noesis_gym_hopper_example");
  // Enable mujoco
  mujoco::init();

  // Configurations
  int control_decimation = 4;
  double discount_factor = 0.995;
  double reset_noise_factor = 5e-3;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "HopperEnvironment";
  std::string scope = "/Example";
  bool verbose = false;

  noesis::gym::HopperEnvironment env(
    control_decimation,
    discount_factor,
    reset_noise_factor,
    enable_logging, visualize,
    name, scope, verbose);

  // Execute demo episode
  env.seed(0);
  env.reset();
  for (size_t t = 0; t < 10*100; ++t) {
    env.actions().setRandom();
    env.step();
  }

  // Print info and internal state of the instance
  NINFO(env);

  // Disable mujoco
  mujoco::exit();
  // Success
  return 0;
}

/* EOF */
