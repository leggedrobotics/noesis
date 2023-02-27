/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_DATASET_HPP_
#define NOESIS_FRAMEWORK_UTILS_DATASET_HPP_

// C/C++
#include <string>

// Noesis
#include "noesis/framework/core/TensorTuple.hpp"
#include "noesis/framework/utils/filestream.hpp"

namespace noesis {
namespace utils {

template<typename ScalarType_>
bool load_data_from_path(const std::string& path, TensorTuple<ScalarType_>& output) {
  // Check if the path exists
  if (!boost::filesystem::exists(path)) {
    return false;
  }
  // Check if the output buffer already contains data
  // TODO: support for non-empty tensor-tuples?
  if (!output.empty()) {
    return false;
  }
  // Check if the output buffer is batched
  // TODO: support for batched tensor tuples?
  if (output.isBatched()) {
    return false;
  }
  // Iterate over all entities in the specified path
  for (const auto& entry : boost::filesystem::directory_iterator(path)) {
    // Check if the filename extension indicates the entity to be a binary dataset file
    if (entry.path().filename().extension() == ".bin") {
      std::string name = entry.path().stem().string();
      // Load data into buffer
      Eigen::Matrix<ScalarType_, Eigen::Dynamic, Eigen::Dynamic> buffer;
      load_eigen_from_binary<ScalarType_>(make_namescope({path, name + ".bin"}) , buffer);
      // Configure dimensions
      std::vector<size_t> dims;
      dims.push_back(static_cast<size_t>(buffer.rows()));
      dims.push_back(static_cast<size_t>(buffer.cols()));
      output.addTensor(name, dims);
      output.get().back().asEigenMatrix() = buffer;
    }
  }
  // Success
  return true;
}

} // utils
} // noesis

#endif // NOESIS_FRAMEWORK_UTILS_DATASET_HPP_

/* EOF */
