/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/math/TensorTupleStatistic.hpp>

namespace noesis {
namespace tests {

/*
 * Define Test Fixture
 */
 
template<typename ScalarType_>
class TensorTupleStatisticTest : public ::testing::Test
{
protected:
  using ScalarType = ScalarType_;
  TensorTupleStatisticTest() = default;
  ~TensorTupleStatisticTest() = default;
};

// Test over the supported arithmetic types
using TestTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(TensorTupleStatisticTest, TestTypes);

/*
 * Tests
 */

TYPED_TEST(TensorTupleStatisticTest, RollingStatisticsFromBatches) {
  using ScalarType = typename TensorTupleStatisticTest_RollingStatisticsFromBatches_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Define tuple
  TensorsSpec spec = {{"x",{3,4}},{"y",{3,4}}};
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-3.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(7.0);
  
  // Create samples
  TensorTuple<ScalarType> samples("samples", spec, 1, 12);
  for (size_t k = 0; k < samples.size(); ++k) {
    samples[k].setRandomNormal(prng, min, max);
  }
  
  // Create tensor statistics and initialize w/ initial samples
  TensorTupleStatistic<ScalarType> tstats("tstat", "test", true);
  tstats.configure(spec);
  NINFO("Initial Mean: " << tstats.mean());
  NINFO("Initial Standard Deviation: " << tstats.stddev());
  NINFO("Initial Count: " << tstats.count());
  
  // Generate a sequence of samples and update the tensor statistics iteratively
  for (size_t i = 0; i < 100000; ++i) {
    for (size_t k = 0; k < samples.size(); ++k) {
      samples[k].setRandomNormal(prng, min, max);
    }
    tstats.update(samples);
  }
  NINFO("Mean: " << tstats.mean());
  NINFO("Standard Deviation: " << tstats.stddev());
  NINFO("Count: " << tstats.count());
  
  // Test convergence of mean and variance
  auto& mean = tstats.mean();
  auto stddev = tstats.stddev();
  for (size_t k = 0; k < mean.size(); ++k) {
    samples[k].setRandomNormal(prng, min, max);
    for (size_t i = 0; i < mean[k].size(); ++i) {
      EXPECT_NEAR(2.0, mean[k][i], 1e-2);
      EXPECT_NEAR(1.667, stddev[k][i], 1e-2);
    }
  }
  
  // Normalize last generated samples
  NINFO("Samples: " << samples);
  tstats.normalize(samples);
  NINFO("Normalized Samples: " << samples);
}

TYPED_TEST(TensorTupleStatisticTest, RollingStatisticsFromSamples) {
  using ScalarType = typename TensorTupleStatisticTest_RollingStatisticsFromSamples_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Define tuple
  TensorsSpec spec = {{"x",{3,4}},{"y",{3,4}}};
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-3.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(7.0);
  
  // Create samples
  TensorTuple<ScalarType> samples("samples", spec, 1, 1);
  for (size_t k = 0; k < samples.size(); ++k) {
    samples[k].setRandomNormal(prng, min, max);
  }
  
  // Create tensor statistics and initialize w/ initial samples
  TensorTupleStatistic<ScalarType> tstats("tstat", "test", true);
  tstats.configure(spec);
  NINFO("Initial Mean: " << tstats.mean());
  NINFO("Initial Standard Deviation: " << tstats.stddev());
  NINFO("Initial Count: " << tstats.count());
  
  // Generate a sequence of samples and update the tensor statistics iteratively
  for (size_t i = 0; i < 100000; ++i) {
    for (size_t k = 0; k < samples.size(); ++k) {
      samples[k].setRandomNormal(prng, min, max);
    }
    tstats.update(samples);
  }
  NINFO("Mean: " << tstats.mean());
  NINFO("Standard Deviation: " << tstats.stddev());
  NINFO("Count: " << tstats.count());
  
  // Test convergence of mean and variance
  auto& mean = tstats.mean();
  auto stddev = tstats.stddev();
  for (size_t k = 0; k < mean.size(); ++k) {
    samples[k].setRandomNormal(prng, min, max);
    for (size_t i = 0; i < mean[k].size(); ++i) {
      EXPECT_NEAR(2.0, mean[k][i], 5e-2);
      EXPECT_NEAR(1.667, stddev[k][i], 5e-2);
    }
  }
  
  // Normalize last generated samples
  NINFO("Samples: " << samples);
  tstats.normalize(samples);
  NINFO("Normalized Samples: " << samples);
}

} // namespace tests
} // namespace noesis

/* EOF */
