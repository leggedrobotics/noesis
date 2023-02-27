/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Boost
#include <boost/filesystem.hpp>

// Noesis
#include "noesis/framework/system/process.hpp"
#include "noesis/framework/utils/string.hpp"
#include "noesis/framework/log/message.hpp"
#include "noesis/framework/core/Graph.hpp"

namespace noesis {
namespace core {

Graph::Graph(
    tf::SessionConfig session,
    const std::string& name,
    const std::string& scope,
    bool verbose):
  ::noesis::core::Object(name, scope, verbose),
  sessionConfig_(session)
{
}

Graph::Graph(const Config& config):
  Graph(config.session, config.name, config.scope, config.verbose)
{
}

Graph::~Graph()  {
  if (session_) { shutdown(); }
}

void Graph::addDevice(const std::string& name) {
  std::string type = name.substr(0, 3);
  std::string id = name.substr(3, name.size()-3);
  NFATAL_IF(type != "CPU" && type != "GPU" && type != "TPU",
    "[" << namescope() << "]: AddDevice: Invalid type: Must be one of {CPU, GPU, TPU}")
  NINFO("[" << namescope() << "]: Adding device: " << name)
  devices_.push_back(name);
}

void Graph::generateFrom(const std::string& pyscript, const std::string& virtualenv) {
  NFATAL_IF(!boost::filesystem::exists(pyscript),
    "[" << namescope() << "]: GenerateFrom: Filename not found: " << pyscript)
  NFATAL_IF(boost::filesystem::extension(pyscript) != ".py",
    "[" << namescope() << "]: GenerateFrom: Filename is not a Python '*.py' script: " << pyscript)
  NINFO("[" << namescope() << "]: GenerateFrom: Generating graph protobuf from Python script: " << pyscript)
  // NOTE: We use a BASH script to wrap the call so to the python
  // interpreter so that we can execute with the target virtualenv.
  std::string command = noesis::rootpath() + "/noesis/bin/python.sh";
  command += " --script " + pyscript;
  command += " --venv " + virtualenv;
  auto result = std::system(command.c_str());
  NFATAL_IF(result, "[" << namescope() << "]: GenerateFrom: Failed to generate graph from: " << pyscript)
}

void Graph::loadFrom(const std::string& filename) {
  NFATAL_IF(!boost::filesystem::exists(filename),
    "[" << namescope() << "]: LoadFrom: Filename not found: " << filename)
  NFATAL_IF(boost::filesystem::extension(filename) != ".pb",
    "[" << namescope() << "]: LoadFrom: Filename is not a protobuf '*.pb' file: " << filename)
  NINFO("[" << namescope() << "]: LoadFrom: Loading MetaGraphDef from protobuf file: " << filename)
  auto result = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), filename, &metaGraphDef_);
  NFATAL_IF(!result.ok(), "[" << namescope() << "]: LoadFrom: " << result.ToString())
}

void Graph::startup() {
  NINFO("[" << namescope() << "]: Starting-up graph ...")
  NFATAL_IF(!metaGraphDef_.has_graph_def(),
    "[" << namescope() << "]: Failed to startup graph: MetaGraphDef has not been loaded!")
  createSession();
  initialize();
}

void Graph::shutdown() {
  NINFO("[" << namescope() << "]: Shutting-down graph ...")
  if (session_) { TF_CHECK_OK(session_->Close()); }
  metaGraphDef_.release_graph_def();
  session_.reset(nullptr);
}

void Graph::initialize() {
  NFATAL_IF(!session_, "[" << namescope() << "]: Failed to initialize graph: Session is not active!")
  auto status = session_->Run({}, {}, {"Global/global_vars_init_op"}, {});
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
  status = session_->Run({}, {}, {"Global/local_vars_init_op"}, {});
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
  resetGlobalStep();
}

void Graph::restoreFrom(const std::string& path, const std::string& saver) {
  NFATAL_IF(!session_, "[" << namescope() << "]: Failed to restore graph: Session is not active!")
  NFATAL_IF(!boost::filesystem::exists(path + ".index"), "[" << namescope() << "]: RestoreFrom: Checkpoint does not exist: " << path)
  tensorflow::Tensor checkpointsFilename(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointsFilename.scalar<std::string>()() = path;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  std::vector<std::string> target_tensor_names;
  if (saver.empty()) {
    inputs.emplace_back(metaGraphDef_.saver_def().filename_tensor_name(), checkpointsFilename);
    target_tensor_names.push_back(metaGraphDef_.saver_def().restore_op_name());
  } else {
    std::string filename_tensor_name = saver + "/filename_in";
    std::string restore_op_name = saver + "/restore_op";
    inputs.emplace_back(filename_tensor_name, checkpointsFilename);
    target_tensor_names.push_back(restore_op_name);
  }
  auto status = session_->Run(inputs, {}, target_tensor_names, {});
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::saveTo(const std::string& path) {
  NFATAL_IF(!session_, "[" << namescope() << "]: Failed to save graph: Session is not active!")
  tensorflow::Tensor checkpointsFilename(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointsFilename.scalar<std::string>()() = path;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  inputs.emplace_back(metaGraphDef_.saver_def().filename_tensor_name(), checkpointsFilename);
  std::vector<std::string> output_tensor_names;
  output_tensor_names.push_back(metaGraphDef_.saver_def().save_tensor_name());
  auto status = session_->Run(inputs, output_tensor_names, {}, {});
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::resetGlobalStep() {
  auto status = session_->Run({}, {}, {"Global/global_step/reset_op"}, nullptr);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::incrementGlobalStep() {
  auto status = session_->Run({}, {}, {"Global/global_step/increment_op"}, nullptr);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(
  const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
  const std::vector<std::string>& output_tensor_names,
  const std::vector<std::string>& target_node_names,
  std::vector<tensorflow::Tensor>& outputs) {
  auto status = session_->Run(inputs, output_tensor_names, target_node_names, &outputs);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(
  const std::vector<std::string>& output_tensor_names,
  const std::vector<std::string>& target_node_names,
  std::vector<tensorflow::Tensor>& outputs
) {
  auto status = session_->Run({}, output_tensor_names, target_node_names, &outputs);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(const std::vector<std::string>& output_tensor_names, std::vector<tensorflow::Tensor>& outputs) {
  auto status = session_->Run({}, output_tensor_names, {}, &outputs);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(const std::string& output_tensor_name, tensorflow::Tensor& output) {
  std::vector<tensorflow::Tensor> outputBuffer;
  auto status = session_->Run({}, {output_tensor_name}, {}, &outputBuffer);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
  NFATAL_IF(outputBuffer.size()!=1, "[" << namescope() << "]: Output does not match the number of output tensors")
  output = outputBuffer[0];
}

void Graph::run(
  const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
  const std::vector<std::string>& target_node_names
) {
  auto status = session_->Run(inputs, {}, target_node_names, nullptr);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(
  const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
  const std::string& target_node_name
) {
  auto status = session_->Run(inputs, {}, {target_node_name}, nullptr);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(const std::vector<std::string>& target_node_names) {
  auto status = session_->Run({}, {}, target_node_names, nullptr);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::run(const std::string& target_node_name) {
  auto status = session_->Run({}, {}, {target_node_name}, nullptr);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString())
}

void Graph::createSession() {
  NINFO("[" << namescope() << "]: Session: Launching TensorFlow session ...")
  // Set the session configurations
  tensorflow::SessionOptions options;
  // NOTE: See the following link for details:
  // https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/core/protobuf/config.proto
  tensorflow::ConfigProto* config = &options.config;
  // CPU device configurations
  (*config->mutable_device_count())["CPU"] = 1;
  // Target device configurations
  if (tf::filter_gpu_names(devices_).empty()) {
    NINFO("[" << namescope() << "]: Session: Allocating resources only on host CPU (/device:CPU:0)")
    (*config->mutable_device_count())["GPU"] = 0;
    config->mutable_gpu_options()->set_visible_device_list("");
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
  } else {
    size_t deviceCount = 0;
    std::string gpuIdList;
    std::string gpuDeviceList;
    auto gpus = tf::filter_gpu_names(devices_);
    for (const auto& name: gpus) {
      gpuIdList += name.substr(3, name.size()-3) + std::string(",");
      gpuDeviceList += std::string("/device:GPU:") + std::to_string(deviceCount++) + std::string(",");
    }
    gpuIdList.pop_back();
    gpuDeviceList.pop_back();
    NINFO("[" << namescope() << "]: Session: GPU Names: " << utils::vector_to_string(gpus))
    NINFO("[" << namescope() << "]: Session: GPU CUDA Ids: " << gpuIdList)
    NINFO("[" << namescope() << "]: Session: GPU TF Devices: " << gpuDeviceList)
    // We limit the visibility of the devices to be mapped by CUDA and TensorFlow
    setenv("CUDA_VISIBLE_DEVICES", gpuIdList.c_str(), 1);
    // Set the device list into the session configurations
    config->mutable_gpu_options()->set_visible_device_list(gpuIdList);
    config->mutable_gpu_options()->set_allow_growth(true);
  }
  // General device placement configurations
  config->set_allow_soft_placement(sessionConfig_.allow_soft_placement);
  config->set_log_device_placement(sessionConfig_.log_device_placement);
  // Session configurations
  config->set_use_per_session_threads(sessionConfig_.use_per_session_threads);
  config->set_isolate_session_state(sessionConfig_.isolate_session_state);
  // CPU thread-pool configurations
  config->set_inter_op_parallelism_threads(sessionConfig_.inter_op_parallelism_threads);
  config->set_intra_op_parallelism_threads(sessionConfig_.intra_op_parallelism_threads);
  // Create the tensorflow session
  options.env = tensorflow::Env::Default();
  tensorflow::Session* session = nullptr;
  auto status = tensorflow::NewSession(options, &session);
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: Session: " << status.ToString())
  session_.reset(session);
  // Create the graph within the new session
  status = session_->Create(metaGraphDef_.graph_def());
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: Session: " << status.ToString())
}

} // namespace core
} // namespace noesis

/* EOF */
