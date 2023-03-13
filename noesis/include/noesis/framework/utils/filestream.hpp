/*!
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_FILE_STREAM_HPP_
#define NOESIS_FRAMEWORK_UTILS_FILE_STREAM_HPP_

// C/C++
#include <iostream>
#include <string>

// Eigen
#include <Eigen/Core>

namespace noesis {
namespace utils {

template<typename ScalarType_>
bool write_eigen_to_binary(const std::string& filename, const Eigen::Matrix<ScalarType_, Eigen::Dynamic, Eigen::Dynamic>& data) {
  std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }
  auto rows = static_cast<size_t>(data.rows());
  auto cols = static_cast<size_t>(data.cols());
  out.write((char*) (&rows), sizeof(size_t));
  out.write((char*) (&cols), sizeof(size_t));
  out.write((char*) data.data(), rows * cols * sizeof(ScalarType_));
  out.close();
  return true;
}

template<typename ScalarType_>
bool load_eigen_from_binary(const std::string& filename, Eigen::Matrix<ScalarType_, Eigen::Dynamic, Eigen::Dynamic>& data) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  size_t rows = 0;
  size_t cols = 0;
  in.read((char*) (&rows), sizeof(size_t));
  in.read((char*) (&cols), sizeof(size_t));
  data.resize(rows, cols);
  in.read((char*) data.data(), rows * cols * sizeof(ScalarType_));
  in.close();
  return true;
}

} // utils
} // noesis

#endif // NOESIS_FRAMEWORK_UTILS_FILE_STREAM_HPP_

/* EOF */
