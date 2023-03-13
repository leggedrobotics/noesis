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
#include <noesis/framework/core/Tensor.hpp>

// Tests
#include "utils/helpers.hpp"

namespace noesis {
namespace tests {

template<typename ScalarType_>
class TensorMapTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  using ScalarType = ScalarType_;
  using MatrixType = Eigen::Matrix<ScalarType, -1, -1>;
  using TensorType = noesis::Tensor<ScalarType>;
  // We use default constructor/destructor
  TensorMapTest() = default;
  ~TensorMapTest() = default;
protected:
  static noesis::math::RandomNumberGenerator<ScalarType> prng;
};

// Initialize the static RandomNumberGenerator member
template<typename ScalarType_>
noesis::math::RandomNumberGenerator<ScalarType_> TensorMapTest<ScalarType_>::prng;

// Test over the supported arithmetic types
using ScalarTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(TensorMapTest, ScalarTypes);

/*
 * Tests
 */

TYPED_TEST(TensorMapTest, Creation) {
  using TensorType = typename TensorMapTest_Creation_Test::TensorType;
  NNOTIFY("Testing creating maps to tensors ...");
  
  TensorType tensor("tensor", {5, 6, 3, 4}, true);
  tensor.setConstant(1.0);
  
  auto datumMap = tensor(0, 0);
  NINFO("Datum map:\n" << datumMap);
  EXPECT_EQ("tensor", datumMap.name());
  EXPECT_EQ((std::vector<size_t>{5, 6}), datumMap.dimensions());
  EXPECT_EQ(0, datumMap.batches());
  EXPECT_EQ(30, datumMap.size());
  EXPECT_EQ(0, datumMap.totalTimeSteps());
  EXPECT_FALSE(datumMap.isBatched());
  
  auto trajMap = tensor(0);
  NINFO("Trajectory map:\n" << trajMap);
  EXPECT_EQ("tensor", trajMap.name());
  EXPECT_EQ((std::vector<size_t>{5, 6, 3}), trajMap.dimensions());
  EXPECT_EQ(1, trajMap.batches());
  EXPECT_EQ(90, trajMap.size());
  EXPECT_EQ(3, trajMap.totalTimeSteps());
  EXPECT_TRUE(trajMap.isBatched());
  
  auto blockMap = tensor.batch_block(1, 2);
  NINFO("Block map:\n" << blockMap);
  EXPECT_EQ("tensor", blockMap.name());
  EXPECT_EQ((std::vector<size_t>{5, 6, 3, 2}), blockMap.dimensions());
  EXPECT_EQ(2, blockMap.batches());
  EXPECT_EQ(180, blockMap.size());
  EXPECT_EQ(6, blockMap.totalTimeSteps());
  EXPECT_TRUE(blockMap.isBatched());
  
  NINFO("Datum map:\n" << tensor(0, 0));
  NINFO("Trajectory map:\n" << tensor(0));
  NINFO("Block map:\n" << tensor.batch_block(1, 2));
}

TYPED_TEST(TensorMapTest, TimeStepOccupancy) {
  using TensorType = typename TensorMapTest_TimeStepOccupancy_Test::TensorType;
  NNOTIFY("Testing resetting ...");
  
  TensorType tensor("tensor", {5, 6, 3, 4}, true);
  tensor.setConstant(1.0);
  
  auto datumMap = tensor(0, 0);
  {
    std::vector<size_t> timesteps(datumMap.timesteps(), datumMap.timesteps() + datumMap.batches());
    EXPECT_EQ((std::vector<size_t>{}), timesteps);
  }
  datumMap.clearTimeSteps();
  {
    std::vector<size_t> timesteps(datumMap.timesteps(), datumMap.timesteps() + datumMap.batches());
    EXPECT_EQ((std::vector<size_t>{}), timesteps);
  }
  datumMap.fillTimeSteps();
  {
    std::vector<size_t> timesteps(datumMap.timesteps(), datumMap.timesteps() + datumMap.batches());
    EXPECT_EQ((std::vector<size_t>{}), timesteps);
  }
  
  auto trajMap = tensor(0);
  {
    std::vector<size_t> timesteps(trajMap.timesteps(), trajMap.timesteps() + trajMap.batches());
    EXPECT_EQ((std::vector<size_t>{3}), timesteps);
  }
  trajMap.clearTimeSteps();
  {
    std::vector<size_t> timesteps(trajMap.timesteps(), trajMap.timesteps() + trajMap.batches());
    EXPECT_EQ((std::vector<size_t>{0}), timesteps);
  }
  trajMap.fillTimeSteps();
  {
    std::vector<size_t> timesteps(trajMap.timesteps(), trajMap.timesteps() + trajMap.batches());
    EXPECT_EQ((std::vector<size_t>{3}), timesteps);
  }
  
  auto blockMap = tensor.batch_block(1, 2);
  {
    std::vector<size_t> timesteps(blockMap.timesteps(), blockMap.timesteps() + blockMap.batches());
    EXPECT_EQ((std::vector<size_t>{3,3}), timesteps);
  }
  blockMap.clearTimeSteps();
  {
    std::vector<size_t> timesteps(blockMap.timesteps(), blockMap.timesteps() + blockMap.batches());
    EXPECT_EQ((std::vector<size_t>{0,0}), timesteps);
  }
  blockMap.fillTimeSteps();
  {
    std::vector<size_t> timesteps(blockMap.timesteps(), blockMap.timesteps() + blockMap.batches());
    EXPECT_EQ((std::vector<size_t>{3,3}), timesteps);
  }
  
}

TYPED_TEST(TensorMapTest, Randomize) {
  using TensorType = typename TensorMapTest_Randomize_Test::TensorType;
  
  TensorType tensor("tensor", {5, 6, 3, 4}, true);
  tensor.setZero();
  NINFO(tensor);
  
  auto datumMap = tensor(0, 0);
  auto trajMap = tensor(0);
  auto blockMap = tensor.batch_block(1, 2);
  
  datumMap.setRandomUnitUniform(this->prng);
  NINFO(tensor);
  NINFO(datumMap);
  
  trajMap.setRandomUnitUniform(this->prng);
  NINFO(tensor);
  NINFO(trajMap);
  
  blockMap.setRandomUnitUniform(this->prng);
  NINFO(tensor);
  NINFO(blockMap);
}

TYPED_TEST(TensorMapTest, MapToMap) {
  using TensorType = typename TensorMapTest_MapToMap_Test::TensorType;
  
  TensorType tensor("tensor", {5, 6, 3, 4}, true);
  tensor.setConstant(1.0);
  
  // TODO: can we have these?
//  auto datumMap = tensor(0, 0);
//  auto trajMap = tensor(0);
//  auto trajDatumMap = trajMap(0, 0);
//  isEqual(trajDatumMap, datumMap);
}

TYPED_TEST(TensorMapTest, Assignment) {
  using ScalarType = typename TensorMapTest_Assignment_Test::ScalarType;
  using MatrixType = typename TensorMapTest_Assignment_Test::MatrixType;
  using TensorType = typename TensorMapTest_Assignment_Test::TensorType;
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.setConstant(1.0);
  TensorType tensorAlt("tensor_alt", {4,3,2,1}, true);
  tensorAlt.setConstant(2.0);
  TensorType tensorOther("tensor_other", {4,3}, false);
  tensorOther.setConstant(3.0);
  
  tensor(0, 0) = tensorOther;
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(3, mat.data()[k]);
    }
  }
  
  tensor(1) = tensorAlt(0);
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 1).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(2, mat.data()[k]);
    }
  }
  {
    MatrixType mat = tensor(1, 1).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(2, mat.data()[k]);
    }
  }
}

TYPED_TEST(TensorMapTest, MapArithmetic) {
  using MatrixType = typename TensorMapTest_MapArithmetic_Test::MatrixType;
  using TensorType = typename TensorMapTest_MapArithmetic_Test::TensorType;
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.setConstant(0.0);
  tensor(0, 0).setConstant(1.0);
  tensor(1, 0).setConstant(2.0);
  tensor(0, 1).setConstant(3.0);
  tensor(1, 1).setConstant(4.0);
  tensor(0, 2).setConstant(5.0);
  tensor(1, 2).setConstant(6.0);
  NINFO(tensor);
  
  tensor(0, 0) += tensor(0, 1);
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(4, mat.data()[k]);
    }
  }
  
  tensor(1, 0) -= tensor(1, 1);
  NINFO(tensor);
  {
    MatrixType mat = tensor(1, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(-2, mat.data()[k]);
    }
  }
  
  tensor(0, 2) *= tensor(0, 1);
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 2).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(15, mat.data()[k]);
    }
  }
  
  tensor(1, 2) /= tensor(1, 1);
  NINFO(tensor);
  {
    MatrixType mat = tensor(1, 2).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(1.5, mat.data()[k]);
    }
  }
}

TYPED_TEST(TensorMapTest, TensorArithmetic) {
  using ScalarType = typename TensorMapTest_TensorArithmetic_Test::ScalarType;
  using MatrixType = typename TensorMapTest_TensorArithmetic_Test::MatrixType;
  using TensorType = typename TensorMapTest_TensorArithmetic_Test::TensorType;
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.setConstant(1.0);
  tensor.setConstant(0.0);
  tensor(0, 0).setConstant(1.0);
  tensor(1, 0).setConstant(2.0);
  tensor(0, 1).setConstant(3.0);
  tensor(1, 1).setConstant(4.0);
  tensor(0, 2).setConstant(5.0);
  tensor(1, 2).setConstant(6.0);
  NINFO(tensor);
  
  TensorType tensorOther("tensor_other", {4,3}, false);
  tensorOther.setConstant(1.3);
  NINFO(tensorOther);
  
  tensor(0, 0) += tensorOther;
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(static_cast<ScalarType>(2.3), mat.data()[k]);
    }
  }
  
  tensor(1, 0) -= tensorOther;
  NINFO(tensor);
  {
    MatrixType mat = tensor(1, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_NEAR(static_cast<ScalarType>(0.7), mat.data()[k], 1.0e-6);
    }
  }
  
  tensor(0, 2) *= tensorOther;
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 2).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(static_cast<ScalarType>(6.5), mat.data()[k]);
    }
  }

  tensor(1, 2) /= tensorOther;
  NINFO(tensor);
  {
    MatrixType mat = tensor(1, 2).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_NEAR(static_cast<ScalarType>(6.0/1.3), mat.data()[k], 1.0e-6);
    }
  }
}

TYPED_TEST(TensorMapTest, ScalarArithmetic) {
  using ScalarType = typename TensorMapTest_ScalarArithmetic_Test::ScalarType;
  using MatrixType = typename TensorMapTest_ScalarArithmetic_Test::MatrixType;
  using TensorType = typename TensorMapTest_ScalarArithmetic_Test::TensorType;
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.setConstant(1.0);
  tensor.setConstant(0.0);
  tensor(0, 0).setConstant(1.0);
  tensor(1, 0).setConstant(2.0);
  tensor(0, 1).setConstant(3.0);
  tensor(1, 1).setConstant(4.0);
  tensor(0, 2).setConstant(5.0);
  tensor(1, 2).setConstant(6.0);
  NINFO(tensor);
  
  tensor(0, 0) += 1.3;
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(static_cast<ScalarType>(2.3), mat.data()[k]);
    }
  }
  
  tensor(1, 0) -= 1.3;
  NINFO(tensor);
  {
    MatrixType mat = tensor(1, 0).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_NEAR(static_cast<ScalarType>(0.7), mat.data()[k], 1.0e-6);
    }
  }
  
  tensor(0, 2) *= 1.3;
  NINFO(tensor);
  {
    MatrixType mat = tensor(0, 2).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_EQ(static_cast<ScalarType>(6.5), mat.data()[k]);
    }
  }
  
  tensor(1, 2) /= 1.3;
  NINFO(tensor);
  {
    MatrixType mat = tensor(1, 2).asEigenMatrix();
    for (size_t k = 0; k < mat.size(); ++k) {
      EXPECT_NEAR(static_cast<ScalarType>(6.0/1.3), mat.data()[k], 1.0e-6);
    }
  }
}

TYPED_TEST(TensorMapTest, EigenMapsToFilled) {
  using TensorType = typename TensorMapTest_EigenMapsToFilled_Test::TensorType;
  NNOTIFY("Testing mapping to eigen matrices and tensors ...");
  
  TensorType tensor("tensor", {5, 6, 3, 2}, true);
  tensor.setConstant(1.0);
  
  auto datumMap = tensor(0, 0);
  NINFO("Datum map:\n" << datumMap);
  
  auto trajMap = tensor(0);
  NINFO("Trajectory map:\n" << trajMap);
  
  auto datumMatMap = datumMap.asEigenMatrix();
  NINFO("As an Eigen::Matrix:\n" << datumMatMap);
  EXPECT_EQ(5, datumMatMap.rows());
  EXPECT_EQ(6, datumMatMap.cols());
  EXPECT_EQ(30, datumMatMap.sum());
  
  auto datumTenMap = datumMap.template asEigenTensor<2>();
  NINFO("As an Eigen::Tensor:\n" << datumTenMap);
  
  auto trajMatMap = trajMap.asEigenMatrix();
  NINFO("As an Eigen::Matrix:\n" << trajMatMap);
  EXPECT_EQ(5, trajMatMap.rows());
  EXPECT_EQ(18, trajMatMap.cols());
  EXPECT_EQ(90, trajMatMap.sum());
  
  auto trajTenMap = trajMap.template asEigenTensor<3>();
  NINFO("As an Eigen::Tensor:\n" << trajTenMap);
}

// TODO @vt,jh: Make asEigenMatrix return map to only valid samples?
TYPED_TEST(TensorMapTest, EigenMapsToPartiallyFilled) {
  using TensorType = typename TensorMapTest_EigenMapsToPartiallyFilled_Test::TensorType;
  NNOTIFY("Testing mapping to eigen matrices ...");
  // Create partially filled batched tensor
  TensorType tensor("tensor", {3, 2, 10, 4}, true);
  for (size_t b = 0; b < tensor.batches(); ++b) {
    for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
      tensor(t,b).setConstant(b*10+t+1);
    }
    tensor.popBackTimeSteps(b, b+1);
  }
  NINFO("Tensor:" << tensor);
  // Check each batch individualy
  for (size_t b = 0; b < tensor.batches(); ++b) {
    NWARNING("Batch '" << b << "':\n" << tensor(b).asEigenMatrix());
  }
}

} // namespace tests
} // namespace noesis

/* EOF */
