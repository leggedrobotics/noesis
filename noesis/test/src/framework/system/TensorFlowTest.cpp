/*!
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/system/tensorflow.hpp>
#include <noesis/framework/utils/string.hpp>

namespace noesis {
namespace tests {

TEST(TensorFlowTest, CreateSession) {
  tensorflow::Session* session = nullptr;
  tensorflow::Status status = NewSession(tensorflow::SessionOptions(), &session);
  EXPECT_TRUE(status.ok());
  std::cout << "Status string: " << status.ToString() << std::endl;
}

TEST(TensorFlowTest, ListAvailableDevices) {
  auto devices = tf::available_devices();
  EXPECT_EQ("/device:CPU:0", devices[0]);
  std::cout << "Devices: " << utils::vector_to_string(devices) << std::endl;
}

TEST(TensorFlowTest, FilterdDeviceLists) {
  std::vector<std::string> devices = {"/device:CPU:0", "/device:GPU:0", "/device:GPU:1"};
  auto filtered = tf::filter_gpu_devices(devices);
  EXPECT_EQ(2, filtered.size());
  EXPECT_EQ((std::vector<std::string>{"0","1"}), filtered);
  std::cout << "Filtered devices: " << utils::vector_to_string(filtered) << std::endl;
}

} // namespace tests
} // namespace noesis

/* EOF */
