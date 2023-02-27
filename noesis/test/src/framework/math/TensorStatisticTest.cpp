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

// Noesis
#include <noesis/framework/math/TensorStatistic.hpp>

namespace noesis {
namespace tests {

/*
 * Define Test Fixture
 */
 
template<typename ScalarType_>
class TensorStatisticTest : public ::testing::Test
{
protected:
  using ScalarType = ScalarType_;
  TensorStatisticTest() = default;
  ~TensorStatisticTest() = default;
};

// Test over the supported arithmetic types
using TestTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(TensorStatisticTest, TestTypes);

/*
 * Tests
 */

TYPED_TEST(TensorStatisticTest, RollingStatisticsFromBatches) {
  using ScalarType = typename TensorStatisticTest_RollingStatisticsFromBatches_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-3.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(7.0);
  
  // Create sample
  Tensor<ScalarType> samples("samples", {3,4,3,4}, true);
  samples.setRandomNormal(prng, min, max);
  
  // Create tensor statistics and initialize w/ initial samples
  TensorStatistic<ScalarType> tstat("tstat", "test", true);
  tstat.configure({3,4});
  NINFO("Initial Count: " << tstat.count());
  NINFO("Initial Mean: " << tstat.mean());
  NINFO("Initial Standard Deviation: " << tstat.stddev());
  
  // Generate a sequence of samples and update the tensor statistics iteratively
  for (size_t i = 0; i < 100000; ++i) {
    samples.setRandomNormal(prng, min, max);
    tstat.update(samples);
  }
  NINFO("Count: " << tstat.count());
  NINFO("Mean: " << tstat.mean());
  NINFO("Standard Deviation: " << tstat.stddev());
  
  // Test convergence of mean and variance
  auto& mean = tstat.mean();
  auto stddev = tstat.stddev();
  for (size_t k = 0; k < mean.size(); ++k) {
    EXPECT_NEAR(2.0, mean[k], 1e-2);
    EXPECT_NEAR(1.667, stddev[k], 1e-2);
  }
  
  // Normalize last generated samples
  NINFO("Samples: " << samples);
  tstat.normalize(samples);
  NINFO("Normalized Samples: " << samples);
}

TYPED_TEST(TensorStatisticTest, RollingStatisticsFromSamples) {
  using ScalarType = typename TensorStatisticTest_RollingStatisticsFromSamples_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-3.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(7.0);
  
  // Create sample
  Tensor<ScalarType> samples("samples", {3,4,1,1}, true);
  samples.setRandomNormal(prng, min, max);
  
  // Create tensor statistics and initialize w/ initial samples
  TensorStatistic<ScalarType> tstat("tstat", "test", true);
  tstat.configure({3,4});
  NINFO("Initial Count: " << tstat.count());
  NINFO("Initial Mean: " << tstat.mean());
  NINFO("Initial Standard Deviation: " << tstat.stddev());
  
  // Generate a sequence of samples and update the tensor statistics iteratively
  for (size_t i = 0; i < 1000000; ++i) {
    samples.setRandomNormal(prng, min, max);
    tstat.update(samples);
  }
  NINFO("Count: " << tstat.count());
  NINFO("Mean: " << tstat.mean());
  NINFO("Standard Deviation: " << tstat.stddev());
  
  // Test convergence of mean and variance
  auto& mean = tstat.mean();
  auto stddev = tstat.stddev();
  for (size_t k = 0; k < mean.size(); ++k) {
    EXPECT_NEAR(2.0, mean[k], 1e-2);
    EXPECT_NEAR(1.667, stddev[k], 1e-2);
  }
  
  // Normalize last generated samples
  NINFO("Samples: " << samples);
  tstat.normalize(samples);
  NINFO("Normalized Samples: " << samples);
}

} // namespace tests
} // namespace noesis

/* EOF */
