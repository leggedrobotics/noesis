/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_SYSTEM_TENSORFLOW_HPP_
#define NOESIS_FRAMEWORK_SYSTEM_TENSORFLOW_HPP_

// C/C++
// TODO

// Boost
// TODO

// TensorFlow
#include <tensorflow/core/public/session.h>

namespace noesis {
namespace tf {

/*
 * Tensor helper functions
 */

static inline auto dimensions_from_tensor(const tensorflow::Tensor& tensor) {
  std::vector<size_t> dims;
  for (int i = tensor.dims(); i > -1; i--) {
    dims.push_back(static_cast<size_t>(tensor.dim_size(i)));
  }
  return dims;
}

/*
 * Computation device helper functions
 */

// TODO @vt: This needs fixing
static inline std::vector<std::string> available_devices() {
  std::vector<std::string> devices;
  tensorflow::Session* session = nullptr;
  auto status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
  TF_CHECK_OK(status);
  if (status.ok()) {
    std::vector<::tensorflow::DeviceAttributes> response;
    TF_CHECK_OK(session->ListDevices(&response));
    for (const auto& device: response) {
      const auto& fullName = device.name();
      auto name = fullName.substr(fullName.rfind("/device:"));
      devices.push_back(name);
    }
  }
  TF_CHECK_OK(session->Close());
  return devices;
}

static inline std::vector<std::string> filter_gpu_devices(const std::vector<std::string>& devices) {
  std::vector<std::string> ids;
  for (const auto& device: devices) {
    // Search for the GPU tag
    auto pos = device.find("GPU:");
    if (pos != std::string::npos) {
      auto id = device.substr(pos+4);
      ids.push_back(id);
    }
  }
  return ids;
}

static inline std::vector<std::string> filter_gpu_names(const std::vector<std::string>& names) {
  std::vector<std::string> result;
  for (const auto& name: names) {
    if (name.find("GPU") != std::string::npos) {
      result.push_back(name);
    }
  }
  return result;
}

static inline void check_device_identifier(std::string& device) {
  // NOTE: at the moment only a single GPU is supported
  // See link for details: https://github.com/tensorflow/tensorflow/issues/9201
  if (device == "CPU" || device == "CPU0") {
    device = "/device:CPU:0";
  } else if (device == "GPU" || device == "GPU0") {
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    device = "/device:GPU:0";
  } else if (device == "GPU1") {
    setenv("CUDA_VISIBLE_DEVICES", "1", 1);
    device = "/device:GPU:0";
  } else if (device == "GPU2") {
    setenv("CUDA_VISIBLE_DEVICES", "2", 1);
    device = "/device:GPU:0";
  }
}

/*
 * Session configuration helper
 */

struct SessionConfig {
  int32_t inter_op_parallelism_threads{0};
  int32_t intra_op_parallelism_threads{0};
  bool use_per_session_threads{false};
  bool isolate_session_state{true};
  bool allow_soft_placement{false};
  bool log_device_placement{false};
};

} // tf
} // noesis

#endif // NOESIS_FRAMEWORK_SYSTEM_TENSORFLOW_HPP_

/* EOF */
