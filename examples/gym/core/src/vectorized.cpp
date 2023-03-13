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

// Example
#include "Environment.hpp"

int main(int argc, char** argv)
{
  using Termination = example::Environment::Termination;
  
  noesis::init("noesis_gym_vectorized_example");
  
  // Configurations
  size_t batch_size = 3;
  
  // Create a vectorized environment
  auto venv = noesis::gym::make_vectorized<example::Environment>(batch_size, "baz", 0.333, 666, true);

  // Print info
  NNOTIFY(*venv);

  // Test venv operations
  venv->reset();
  for (int t = 0; t < 10; ++t) {
    if (venv->terminations().back().type != Termination::Type::Unterminated) { break; }
    venv->actions()[0].setConstant(t + 1);
    venv->step();
  }
  NINFO("venv:" << *venv << venv->actions() << venv->observations() << venv->rewards());
  
  // Success
  return 0;
}

/* EOF */
