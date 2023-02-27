/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
//#include <memory>

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/core/TensorsSpec.hpp>

namespace noesis {
namespace tests {

template <typename T, typename F>
struct Dummy {};

/*
 * Tests
 */

TEST(TensorsSpecTest, Streaming) {
  std::stringstream ss;
  noesis::TensorsSpec spec;
  ss << spec;
  std::cout << "Spec before: " << ss.str() << std::endl;
  EXPECT_EQ("{}", ss.str());
  spec = {{"command", {1,2}}, {"motion", {3,4}}};
  ss.str(std::string());
  ss << spec;
  std::cout << "Spec after: " << ss.str() << std::endl;
  auto specStr = ss.str();
  EXPECT_EQ("{ 'command': [1, 2], 'motion': [3, 4] }", specStr);
}

} // namespace tests
} // namespace noesis

/* EOF */
