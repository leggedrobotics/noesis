/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_SYSTEM_VERSION_HPP_
#define NOESIS_FRAMEWORK_SYSTEM_VERSION_HPP_

// C/C++
#include <string>

namespace noesis {

std::string cxx_version();

std::string compiler_version();

std::string eigen_version();

std::string tensorflow_version();

} // noesis

#endif // NOESIS_FRAMEWORK_SYSTEM_VERSION_HPP_

/* EOF */
