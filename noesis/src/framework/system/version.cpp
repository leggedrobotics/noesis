/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Header
#include "noesis/framework/system/version.hpp"

// Eigen
#include <Eigen/src/Core/util/Macros.h>

// TensorFlow
#include <tensorflow/core/public/version.h>


#if defined(__GNUC__)
#define CXX_COMPILER_VERSION ("GCC " TF_STR(__GNUC__) "." TF_STR(__GNUC_MINOR__) "." TF_STR(__GNUC_PATCHLEVEL__))
#elif defined(__clang__)
#define CXX_COMPILER_VERSION ("clang " TF_STR(__clang_major__) "." TF_STR(__clang_minor__) "." TF_STR(__clang_patchlevel__))
#else
#define CXX_COMPILER_VERSION ("Unknown")
#endif


namespace noesis {

std::string compiler_version() {
  return std::string(CXX_COMPILER_VERSION);
}

std::string cxx_version() {
  return std::string(TF_STR(__cplusplus));
}

std::string eigen_version() {
  return std::string(TF_STR(EIGEN_WORLD_VERSION) "." TF_STR(EIGEN_MAJOR_VERSION) "." TF_STR(EIGEN_MINOR_VERSION));
}

std::string tensorflow_version() {
  return std::string(TF_VERSION_STRING);
}

} // namespace noesis

/* EOF */
