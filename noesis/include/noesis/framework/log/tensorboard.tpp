/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Header providing template function declarations
#include "noesis/framework/log/tensorboard.hpp"

// TensorFlow
#include <tensorflow/core/lib/histogram/histogram.h>

// Noesis
#include "noesis/framework/utils/png.hpp"

namespace noesis {
namespace log {

/*
 * Template function implementations
 */

template <typename ScalarType_>
bool TensorBoardLogger::appendScalar(const std::string& name, ScalarType_ value) {
  std::unique_lock<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  bool result = true;
  auto& signal = signals_[name];
  // Create the tensorflow::Event
  tensorflow::Event event;
  event.set_wall_time(getWallClockTime());
  event.set_step(signal.getStep());
  auto* summaryValue = event.mutable_summary()->add_value();
  summaryValue->set_tag(name);
  // Add a simple scalar value to the summary value
  summaryValue->set_simple_value(static_cast<float>(value));
  // Write to events file
  result &= signal.append(event);
  checkSignalAppendResult(name, result);
  lock.unlock();
  result &= writeSignal(signal);
  return result;
}

template <typename ScalarType_>
bool TensorBoardLogger::appendVector(const std::string& name, const Vector<ScalarType_>& vector) {
  std::unique_lock<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  bool result = true;
  auto& signal = signals_[name];
  // Create the tensorflow::Event
  tensorflow::Event event;
  event.set_wall_time(getWallClockTime());
  event.set_step(signal.getStep());
  auto* summaryValue = event.mutable_summary()->add_value();
  summaryValue->set_tag(name);
  // TODO @vt: write while also using casting
  // Write to events file
  result &= signal.append(event);
  checkSignalAppendResult(name, result);
  lock.unlock();
  result &= writeSignal(signal);
  return result;
}

template <typename ScalarType_>
bool TensorBoardLogger::appendMatrix(const std::string& name, const Matrix<ScalarType_>& matrix) {
  std::unique_lock<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  bool result = true;
  auto& signal = signals_[name];
  // Create the tensorflow::Event
  tensorflow::Event event;
  event.set_wall_time(getWallClockTime());
  event.set_step(signal.getStep());
  auto* summaryValue = event.mutable_summary()->add_value();
  summaryValue->set_tag(name);
  // TODO @vt: write while also using casting
  // Write to events file
  result &= signal.append(event);
  checkSignalAppendResult(name, result);
  lock.unlock();
  result &= writeSignal(signal);
  return result;
}

template <typename ScalarType_>
bool TensorBoardLogger::appendHistogram(const std::string& name, const std::vector<ScalarType_>& values) {
  // Create a histogram container
  tensorflow::histogram::Histogram hist;
  for (const auto& value : values) {
    hist.Add(static_cast<double>(value));
  }
  std::unique_lock<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  bool result = true;
  auto& signal = signals_[name];
  // Create the tensorflow::Event
  tensorflow::Event event;
  event.set_wall_time(getWallClockTime());
  event.set_step(signal.getStep());
  auto* summaryValue = event.mutable_summary()->add_value();
  summaryValue->set_tag(name);
  // Add a histogram to the summary value
  auto* histProto = summaryValue->mutable_histo();
  hist.EncodeToProto(histProto, false);
  // Write to events file
  result &= signal.append(event);
  checkSignalAppendResult(name, result);
  lock.unlock();
  result &= writeSignal(signal);
  return result;
}

template <typename ScalarType_>
bool TensorBoardLogger::appendImageMatrix(
    const std::string& name,
    const Matrix<ScalarType_>& image_matrix,
    ScalarType_ min,
    ScalarType_ max) {
  // Linear scaling of matrix values
  using MatrixType = Matrix<ScalarType_>;
  MatrixType ones = MatrixType::Ones(image_matrix.rows(), image_matrix.cols());
  MatrixType scaled = (image_matrix - ones * min) / (max - min) * 255.0f;
  auto converted = scaled.template cast<unsigned char>();
  // Convert to encoded string in PNG format
  std::string image_string;
  utils::png_image_to_string(converted, image_string);
  // Append matrix image
  return appendImageString(name, image_string);
}

template <typename ScalarType_>
bool TensorBoardLogger::appendImageMatrix(
    const std::string& name,
    const Matrix<ScalarType_>& image_matrix,
    bool normalize) {
  // Determine the maximum and minimum values of the matrix
  float min = normalize ? image_matrix.minCoeff() : 0.0f;
  float max = normalize ? image_matrix.maxCoeff() : 1.0f;
  return appendImageMatrix(name, image_matrix, min, max);
}

} // namespace log
} // namespace noesis

/* EOF */
