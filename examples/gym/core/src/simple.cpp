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

// Example
#include "Environment.hpp"

int main(int argc, char** argv)
{
  noesis::init("noesis_gym_simple_example");
  
  // Create an simple environment
  example::Environment env;

  // Print info
  NNOTIFY(env);

  // Test env operations
  env.reset();
  for (int t = 0; t < 10; ++t) {
    if (env.terminations().back().type != example::Environment::Termination::Type::Unterminated) { break; }
    env.actions()[0].setConstant(t+1);
    env.step();
  }
  NINFO("env:" << env << env.actions() << env.observations() << env.rewards());
  
  // Success
  return 0;
}

/* EOF */
