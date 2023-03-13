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
#include <noesis/gym/core/Synchronizer.hpp>

// Example
#include "Environment.hpp"

int main(int argc, char** argv)
{
  using Termination = example::Environment::Termination;
  
  noesis::init("noesis_gym_synchronized_example");
  
  // Create an simple environment
  example::Environment env;
  
  // Create a synchronization wrapper
  auto senv = noesis::gym::make_synchronized_wrapper(&env);

  // Print info
  NNOTIFY(*senv);

  // Test venv operations
  senv->reset();
  for (int t = 0; t < 10; ++t) {
    if (senv->terminations().back().type != Termination::Type::Unterminated) { break; }
    senv->actions()[0].setConstant(t + 1);
    senv->step();
  }
  NINFO("senv:" << *senv << senv->actions() << senv->observations() << senv->rewards());
  
  // Success
  return 0;
}

/* EOF */
