/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// Base classes
#include <noesis/framework/math/random.hpp>

namespace noesis {
namespace tests {

/**
 * Define Test Fixture
 */

template<typename ScalarType_>
class RandomNumberGeneratorTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  using ScalarType = ScalarType_;
  // We use default constructor/destructor
  RandomNumberGeneratorTest() = default;
  ~RandomNumberGeneratorTest() = default;
};

// Test over the supported arithmetic types
using ScalarTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(RandomNumberGeneratorTest, ScalarTypes);

/*
 * Tests
 */

TYPED_TEST(RandomNumberGeneratorTest, MultiInstance) {
  using ScalarType = typename RandomNumberGeneratorTest_MultiInstance_Test::ScalarType;
  
  static constexpr size_t num_of_instances = 3;
  static constexpr size_t num_of_samples = 20;
  
  // Create a generator
  std::vector<noesis::math::RandomNumberGenerator<ScalarType>> generators;
  generators.resize(num_of_instances);

  // Create a data buffer
  Eigen::Matrix<ScalarType, num_of_samples, num_of_instances> samples;
  samples.setZero();
  
  NNOTIFY("Starting multi-instance test ...");
  for (size_t t = 0; t < num_of_samples; ++t) {
    for (size_t instance = 0; instance < num_of_instances; ++instance) {
      samples(t, instance) = generators[instance].sampleUnitUniform();
    }
  }
  NINFO("Samples:\n" << samples);

  // Check that all samples match
  for (size_t t = 0; t < num_of_samples; ++t) {
    for (size_t instance = 1; instance < num_of_instances; ++instance) {
      EXPECT_TRUE(samples(t, 0) == samples(t, instance));
    }
  }
}

} // namespace tests
} // namespace noesis

/* EOF */
