/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Environments
#include "noesis/framework/core/Graph.hpp"

namespace noesis {
namespace core {

template<typename Scalar_>
inline void Graph::run(
  const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
  const std::vector<std::string>& output_tensor_names,
  const std::vector<std::string>& target_node_names,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Local buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  // Run the desired operations
  auto status = session_->Run(inputs, output_tensor_names, target_node_names, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  NFATAL_IF(outputs.size() != outputBuffer.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output argument
  for (size_t k = 0; k<outputs.size(); k++){ outputs[k] = outputBuffer[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  const std::vector<std::string>& output_tensor_names,
  const std::vector<std::string>& target_node_names,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  // Local output buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  auto status = session_->Run(input_tensors, output_tensor_names, target_node_names, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  NFATAL_IF(outputs.size() != outputBuffer.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output tensors
  for (size_t k = 0; k<outputs.size(); k++) { outputs[k] = outputBuffer[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  const std::string& output_tensor_name,
  const std::vector<std::string>& target_node_names,
  Tensor<Scalar_>& output
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  // Local output buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  auto status = session_->Run(input_tensors, {output_tensor_name}, target_node_names, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  NFATAL_IF(outputBuffer.size() != 1,
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output tensor
  output = outputBuffer[0];
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<std::string>& output_tensor_names,
  const std::vector<std::string>& target_node_names,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Local buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  // Run the desired operations
  auto status = session_->Run({}, output_tensor_names, target_node_names, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  NFATAL_IF(outputs.size() != outputBuffer.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output argument
  for (size_t k = 0; k<outputs.size(); k++){ outputs[k] = outputBuffer[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<std::string>& output_tensor_names,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Local buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  // Run the desired operations
  auto status = session_->Run({}, output_tensor_names, {}, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  NFATAL_IF(outputs.size() != outputBuffer.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of output tensors");
  for (size_t k = 0; k<outputs.size(); k++){ outputs[k] = outputBuffer[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::string& output_tensor_name,
  Tensor<Scalar_>& output
) {
  // Local buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  // Run the desired operations
  auto status = session_->Run({}, {output_tensor_name}, {}, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  // Check for errors
  NFATAL_IF(outputBuffer.size() != 1,
    "[" << this->namescope() << "]: Outputs do not match the number of output tensors");
  // Copy local output buffer contents to output tensors
  output = outputBuffer[0];
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  const std::vector<std::string>& operations,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  // Retrieve the target output node pairs
  std::vector<std::string> output_tensor_names;
  for (auto& tensor: outputs) { output_tensor_names.emplace_back(tensor.name()); }
  // Local output buffer
  std::vector<tensorflow::Tensor> output_buffers;
  auto status = session_->Run(input_tensors, output_tensor_names, operations, &output_buffers);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  // Check results and output any error messages
  NFATAL_IF(outputs.size() != output_buffers.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output tensors
  for (size_t k = 0; k<outputs.size(); k++) { outputs[k] = output_buffers[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  const std::string& operation,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  // Retrieve the target output node pairs
  std::vector<std::string> output_tensor_names;
  for (auto& tensor: outputs) { output_tensor_names.emplace_back(tensor.name()); }
  // Local output buffer
  std::vector<tensorflow::Tensor> output_buffers;
  auto status = session_->Run(input_tensors, output_tensor_names, {operation}, &output_buffers);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  // Check results and output any error messages
  NFATAL_IF(outputs.size() != output_buffers.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output tensors
  for (size_t k = 0; k<outputs.size(); k++) { outputs[k] = output_buffers[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  std::vector<Tensor<Scalar_>>& outputs
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  // Retrieve the target output node pairs
  std::vector<std::string> output_tensor_names;
  for (auto& tensor: outputs) { output_tensor_names.emplace_back(tensor.name()); }
  // Local output buffer
  std::vector<tensorflow::Tensor> output_buffers;
  auto status = session_->Run(input_tensors, output_tensor_names, {}, &output_buffers);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  // Check results and output any error messages
  NFATAL_IF(outputs.size() != output_buffers.size(),
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  // Copy local output buffer contents to output tensors
  for (size_t k = 0; k<outputs.size(); k++) { outputs[k] = output_buffers[k]; }
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  Tensor<Scalar_>& output
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  // Retrieve the target output node pairs
  std::vector<std::string> output_tensor_names;
  output_tensor_names.emplace_back(output.name());
  // Local output buffer
  std::vector<tensorflow::Tensor> outputBuffer;
  auto status = session_->Run(input_tensors, output_tensor_names, {}, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
  NFATAL_IF(outputBuffer.size() != 1,
    "[" << this->namescope() << "]: Outputs do not match the number of target nodes");
  output = outputBuffer[0];
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  const std::vector<std::string>& operations
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  auto status = session_->Run(input_tensors, {}, operations, nullptr);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
}

template<typename Scalar_>
inline void Graph::run(
  const std::vector<Tensor<Scalar_>>& inputs,
  const std::string& operation
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  for (auto& tensor: inputs) { input_tensors.emplace_back(tensor); }
  auto status = session_->Run(input_tensors, {}, {operation}, nullptr);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
}

template<typename Scalar_>
inline void Graph::run(
  const Tensor<Scalar_>& input,
  const std::string& operation
) {
  // Unpack inputs into local buffer as named tensorflow tensor
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  input_tensors.emplace_back(input);
  auto status = session_->Run(input_tensors, {}, {operation}, nullptr);
  NFATAL_IF(!status.ok(), "[" << this->namescope() << "]: " << status.ToString());
}

template<typename Scalar_>
inline void Graph::run(
  const TensorTuple<Scalar_>& inputs,
  TensorTuple<Scalar_>& outputs
) {
  this->run(inputs.get(), outputs.get());
  // Re-shape batched dimensions because tensorflow flattens all batches/timesteps
  if (inputs.isBatched() && !inputs.empty()) {
    auto time_size = inputs[0].timeStepCapacity();
    auto batch_size = inputs[0].batches();
    auto& timesteps = inputs[0].timesteps();
    outputs.reshape(time_size, batch_size, timesteps);
  }
}

template<typename Scalar_>
inline void Graph::run(
  const TensorTuple<Scalar_>& inputs,
  Tensor<Scalar_>& output
) {
  this->run(inputs.get(), output);
  if (inputs.isBatched() && !inputs.empty()) {
    auto time_size = inputs[0].timeStepCapacity();
    auto batch_size = inputs[0].batches();
    auto& timesteps = inputs[0].timesteps();
    output.reshapeBatches(time_size, batch_size, timesteps);
  }
}

template<typename Scalar_>
inline void Graph::run(
  const TensorTuple<Scalar_>& inputs,
  std::vector<std::string>& operations
) {
  this->run(inputs.get(), operations);
}

template<typename Scalar_>
inline void Graph::run(
  const TensorTuple<Scalar_>& inputs,
  std::string& operation
) {
  this->run(inputs.get(), operation);
}

template<typename Scalar_>
Scalar_ Graph::readModifiable(const std::string& name) {
  tensorflow::Tensor output;
  run(name + "/value", output);
  return output.scalar<Scalar_>()();
}

template<typename Scalar_>
void Graph::writeModifiable(const std::string& name, Scalar_ value) {
  Tensor<Scalar_> input(name + "/input", {1}, false);
  input[0] = value;
  run(input, name + "/set_op");
}

template<typename Scalar_>
void Graph::writeModifiable(const std::string& name, tensorflow::DataType dtype, Scalar_ value) {
  std::pair<std::string, tensorflow::Tensor> input;
  std::vector<tensorflow::Tensor> output;
  input.first = name + "/input";
  input.second = tensorflow::Tensor(dtype, tensorflow::TensorShape({}));
  input.second.scalar<Scalar_>()() = value;
  run({input}, name + "/set_op");
}

} // namespace core
} // namespace noesis

/* EOF */
