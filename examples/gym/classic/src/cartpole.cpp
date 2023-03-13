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

// Noesis Environments
#include <noesis/gym/envs/classic/cartpole/CartpoleEnvironment.hpp>
#include <noesis/gym/envs/classic/cartpole/CartpoleVisualizer.hpp>

int main(int argc, char** argv)
{
  noesis::init("noesis_gym_cartpole_example");
  
  // Configurations
  float reset_noise_factor = 1.0f;
  float randomization_factor = 0.0f;
  float discount_factor = 0.99f;
  float time_step = 0.02f;
  float time_limit = 4.0f;
  bool verbose = true;
  
  // Create a simple single-instance environment
  noesis::gym::CartpoleEnvironment env(
    reset_noise_factor,
    randomization_factor,
    discount_factor,
    time_step,
    time_limit,
    "Cartpole",
    "/Example",
    verbose
  );

  // Create an external visualizer
  auto senv = noesis::gym::make_synchronized_wrapper(&env);
  noesis::gym::CartpoleVisualizer vis(senv.get());
  vis.launch();

  // Execute demo episode for 10 seconds
  senv->seed(0);
  senv->reset();
  for (size_t t = 0; t < 10 * static_cast<size_t>(1.0/time_step); ++t) {
    senv->actions().setRandom();
    senv->step();
  }

  // Print info and internal state of the instance
  NINFO(env);
  
  // Success
  return 0;
}

/* EOF */
