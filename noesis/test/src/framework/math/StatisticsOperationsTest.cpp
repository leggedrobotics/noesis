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
#include <noesis/framework/math/statistics.hpp>

namespace noesis {
namespace tests {

/**
 * Define Test Fixture
 */

template<typename ScalarType_>
class StatisticsOperationsTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  using ScalarType = ScalarType_;
  // We use default constructor/destructor
  StatisticsOperationsTest() = default;
  ~StatisticsOperationsTest() = default;
};

// Test over the supported arithmetic types
using ScalarTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(StatisticsOperationsTest, ScalarTypes);

/*
 * Test coefficient-wise statistics computations
 */

TYPED_TEST(StatisticsOperationsTest, MatrixNormalize) {
  using ScalarType = typename StatisticsOperationsTest_MatrixNormalize_Test::ScalarType;
  // Create sample
  Eigen::MatrixXd matrix(3, 4);
  matrix.setRandom();
  matrix *= 10.0;
  NWARNING("matrix:\n" << matrix);
  // Normalize
  auto moments = math::cwise_normalize(matrix);
  NINFO("matrix mean: " << moments.first);
  NINFO("matrix std: " << moments.second);
  NWARNING("matrix normalized:\n" << matrix);
}


TYPED_TEST(StatisticsOperationsTest, MeanOfScalarTensor) {
  using ScalarType = typename StatisticsOperationsTest_MeanOfScalarTensor_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  // Set bounds
  Tensor<ScalarType> min("min", {1}, false);
  min.setConstant(-5.0);
  Tensor<ScalarType> max("max", {1}, false);
  max.setConstant(3.0);
  
  // Create sample
  Tensor<ScalarType> tensor("scalar", {1}, false);
  tensor.setRandomUniform(prng, min, max);
  auto mean = math::cwise_mean(tensor);
  auto eigenMean = tensor.asEigenMatrix().mean();
  NINFO("scalar mean: " << mean);
  NINFO("scalar mean using Eigen: " << eigenMean);
  EXPECT_EQ(mean, eigenMean);
  EXPECT_EQ(tensor[0], mean);
}

TYPED_TEST(StatisticsOperationsTest, MeanOfRowVectorTensor) {
  using ScalarType = typename StatisticsOperationsTest_MeanOfRowVectorTensor_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  constexpr size_t size = 10000;
  
  // Set bounds
  Tensor<ScalarType> min("min", {1, size}, false);
  min.setConstant(-5.0);
  Tensor<ScalarType> max("max", {1, size}, false);
  max.setConstant(3.0);
  
  // Create sample
  Tensor<ScalarType> tensor("vector", {1, size}, false);
  tensor.setRandomUniform(prng, min, max);
  auto mean = math::cwise_mean(tensor);
  auto eigenMean = tensor.asEigenMatrix().mean();
  NINFO("row vector mean: " << mean);
  NINFO("row vector mean using Eigen: " << eigenMean);
  EXPECT_EQ(mean, eigenMean);
  EXPECT_NEAR(-1.0, mean, 1.0e-2);
}

TYPED_TEST(StatisticsOperationsTest, MeanOfColumnVectorTensor) {
  using ScalarType = typename StatisticsOperationsTest_MeanOfColumnVectorTensor_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  constexpr size_t size = 10000;
  
  // Set bounds
  Tensor<ScalarType> min("min", {size, 1}, false);
  min.setConstant(-5.0);
  Tensor<ScalarType> max("max", {size, 1}, false);
  max.setConstant(3.0);
  
  // Create sample
  Tensor<ScalarType> tensor("vector", {size, 1}, false);
  tensor.setRandomUniform(prng, min, max);
  auto mean = math::cwise_mean(tensor);
  auto eigenMean = tensor.asEigenMatrix().mean();
  NINFO("column vector mean: " << mean);
  NINFO("column vector mean using Eigen: " << eigenMean);
  EXPECT_EQ(mean, eigenMean);
  EXPECT_NEAR(-1.0, mean, 1.0e-2);
}

TYPED_TEST(StatisticsOperationsTest, MeanOfTensorMatrix) {
  using ScalarType = typename StatisticsOperationsTest_MeanOfTensorMatrix_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  constexpr size_t size = 1000;
  
  // Set bounds
  Tensor<ScalarType> min("min", {size, size}, false);
  min.setConstant(-5.0);
  Tensor<ScalarType> max("max", {size, size}, false);
  max.setConstant(3.0);
  
  // Create sample
  Tensor<ScalarType> tensor("matrix", {size, size}, false);
  tensor.setRandomUniform(prng, min, max);
  auto mean = math::cwise_mean(tensor);
  auto eigenMean = tensor.asEigenMatrix().mean();
  NINFO("matrix mean: " << mean);
  NINFO("matrix mean using Eigen: " << eigenMean);
  EXPECT_EQ(mean, eigenMean);
  EXPECT_NEAR(-1.0, mean, 1.0e-2);
}

TYPED_TEST(StatisticsOperationsTest, VarianceOfScalarTensor) {
  using ScalarType = typename StatisticsOperationsTest_VarianceOfScalarTensor_Test::ScalarType;
  // Create sample
  Tensor<ScalarType> tensor("scalar", {1}, false);
  tensor.setConstant(M_PI);
  NWARNING("scalar:" << tensor);
  // Normalize
  auto variance = math::cwise_variance(tensor);
  NWARNING("scalar variance: " << variance);
  EXPECT_EQ(0, variance);
}

TYPED_TEST(StatisticsOperationsTest, NormalizeScalarTensor) {
  using ScalarType = typename StatisticsOperationsTest_NormalizeScalarTensor_Test::ScalarType;
  // Create sample
  Tensor<ScalarType> tensor("scalar", {1}, false);
  tensor.setConstant(1);
  NWARNING("scalar:" << tensor);
  // Normalize
  math::cwise_normalize(tensor);
  NWARNING("scalar normalized: " << tensor);
  EXPECT_EQ(1, tensor[0]);
}

TYPED_TEST(StatisticsOperationsTest, NormalizeVectorTensor) {
  using ScalarType = typename StatisticsOperationsTest_NormalizeVectorTensor_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  constexpr size_t size = 10;
  // Set bounds
  Tensor<ScalarType> min("min", {size}, false);
  min.setConstant(-5.0);
  Tensor<ScalarType> max("max", {size}, false);
  max.setConstant(3.0);
  // Create sample
  Tensor<ScalarType> tensor("vector", {size}, false);
  tensor.setRandomUniform(prng, min, max);
  NWARNING("vector:" << tensor);
  // Normalize
  auto moments = math::cwise_normalize(tensor);
  NINFO("vector mean: " << moments.first);
  NINFO("vector std: " << moments.second);
  NWARNING("vector normalized: " << tensor);
}


/*
 * Test sequence-wise statistics computations
 */

TYPED_TEST(StatisticsOperationsTest, TensorSequenceMean) {
  using ScalarType = typename StatisticsOperationsTest_TensorSequenceMean_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-10.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(10.0);
  
  // Create sample
  Tensor<ScalarType> tensor("samples", {3,4,3,4}, true);
  tensor.setRandomUniform(prng, min, max);
  NINFO("Samples: " << tensor);
  
  // Compute mean over batches
  auto mean = noesis::math::sequence_mean(tensor);
  NINFO("Sequence Mean: " << mean);
  
  // Check values
  for (size_t b = 0; b < tensor.batches(); ++b) {
    for (size_t k = 0; k < mean(b).size(); ++k) {
      ScalarType sum = 0;
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        sum += tensor(t,b)[k];
      }
      EXPECT_EQ(sum/tensor.timesteps()[b], mean(0,b)[k]);
    }
  }
}

/*
 * Test batch-wise statistics computations
 */

TYPED_TEST(StatisticsOperationsTest, TensorBatchMean) {
  using ScalarType = typename StatisticsOperationsTest_TensorBatchMean_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-10.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(10.0);
  
  // Create sample
  Tensor<ScalarType> tensor("samples", {3,4,3,4}, true);
  tensor.setRandomUniform(prng, min, max);
  NINFO("Samples: " << tensor);
  
  // Compute mean over batches
  auto mean = noesis::math::batch_mean(tensor);
  NINFO("Batch Mean: " << mean);
  
  // Check values
  for (size_t k = 0; k < mean.size(); ++k) {
    ScalarType sum = 0;
    for (size_t b = 0; b < tensor.batches(); ++b) {
      sum += tensor(b)[k];
    }
    EXPECT_EQ(sum/tensor.batches(), mean[k]);
  }
}

/*
 * Test datum-wise statistics computations
 */

TYPED_TEST(StatisticsOperationsTest, TensorMean) {
  using ScalarType = typename StatisticsOperationsTest_TensorMean_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-10.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(10.0);
  
  // Create sample
  Tensor<ScalarType> tensor("samples", {3,4,3,4}, true);
  tensor.setRandomUniform(prng, min, max);
  NINFO("Samples: " << tensor);
  
  // Compute total mean
  auto mean = noesis::math::mean(tensor);
  NINFO("Total Mean: " << mean);
  
  // Check values
  for (size_t k = 0; k < mean.size(); ++k) {
    ScalarType sum = 0;
    for (size_t b = 0; b < tensor.batches(); ++b) {
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        sum += tensor(t,b)[k];
      }
    }
    EXPECT_EQ(sum/(tensor.totalTimeSteps()), mean[k]);
  }
}

TYPED_TEST(StatisticsOperationsTest, TensorVariance) {
  using ScalarType = typename StatisticsOperationsTest_TensorVariance_Test::ScalarType;
  math::RandomNumberGenerator<ScalarType> prng;
  
  // Set bounds
  Tensor<ScalarType> min("min", {3,4}, false);
  min.setConstant(-10.0);
  Tensor<ScalarType> max("max", {3,4}, false);
  max.setConstant(10.0);
  
  // Create sample
  Tensor<ScalarType> tensor("samples", {3,4,3,4}, true);
  tensor.setRandomUniform(prng, min, max);
  NINFO("Samples: " << tensor);
  
  // Compute total variance
  auto variance = noesis::math::variance(tensor);
  NINFO("Total Variance: " << variance);
  
  // Check values
  // TODO
}

} // namespace tests
} // namespace noesis

/* EOF */
