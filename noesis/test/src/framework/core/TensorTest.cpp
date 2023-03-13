/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Joonho Lee
 * @email     junja94@gmail.com
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

/*
 * Test Fixture
 */

template<typename ScalarType_>
class TensorTest : public ::testing::Test
{
protected:
  // Aliases
  using ScalarType = ScalarType_;
  using TensorType = Tensor<ScalarType>;
  // We use default constructor/destructor
  TensorTest() = default;
  ~TensorTest() = default;
protected:
  static noesis::math::RandomNumberGenerator<ScalarType> prng;
};

// Initialize the static RandomNumberGenerator member
template<typename ScalarType_>
noesis::math::RandomNumberGenerator<ScalarType_> TensorTest<ScalarType_>::prng;


// Test over the supported arithmetic types
using ScalarTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(TensorTest, ScalarTypes);

/*
 * Tests
 */

TYPED_TEST(TensorTest, Creation) {
  using TensorType = typename TensorTest_Creation_Test::TensorType;
  NNOTIFY("Testing creation ...");
  
  TensorType emptyUnnamedTensor;
  NINFO(emptyUnnamedTensor);
  auto dims = emptyUnnamedTensor.dimensions();
  EXPECT_EQ(0, emptyUnnamedTensor.size());
  EXPECT_EQ(std::vector<size_t>{0}, dims);
  EXPECT_EQ("", emptyUnnamedTensor.name());
  EXPECT_EQ((std::vector<size_t>{}), emptyUnnamedTensor.capacities());
  
  TensorType emptyTensor("empty_tensor");
  NINFO(emptyTensor);
  dims = emptyTensor.dimensions();
  EXPECT_EQ(0, emptyTensor.size());
  EXPECT_EQ("empty_tensor", emptyTensor.name());
  EXPECT_EQ((std::vector<size_t>{0}), dims);
  EXPECT_EQ((std::vector<size_t>{}), emptyTensor.capacities());
  
  TensorType scalarTensor("scalar_tensor", {}, false);
  NINFO(scalarTensor);
  dims = scalarTensor.dimensions();
  EXPECT_EQ(1, scalarTensor.size());
  EXPECT_EQ((std::vector<size_t>{}), dims);
  EXPECT_EQ((std::vector<size_t>{}), scalarTensor.capacities());
  
  TensorType simpleTensor("simple_tensor", {4,3,2,1}, false);
  NINFO(simpleTensor);
  dims = simpleTensor.dimensions();
  EXPECT_EQ(24, simpleTensor.size());
  EXPECT_EQ((std::vector<size_t>{4,3,2,1}), dims);
  EXPECT_EQ((std::vector<size_t>{}), simpleTensor.capacities());
  
  TensorType batchedTensor("batched_tensor", {4,3,2,3}, true);
  NINFO(batchedTensor);
  dims = batchedTensor.dimensions();
  auto timesteps = batchedTensor.timesteps();
  EXPECT_EQ(72, batchedTensor.size());
  EXPECT_EQ((std::vector<size_t>{4,3,2,3}), dims);
  EXPECT_EQ((std::vector<size_t>{2,2,2}), timesteps);
  EXPECT_EQ(2, batchedTensor.timeStepCapacity());
  EXPECT_EQ(3, batchedTensor.batches());
  EXPECT_EQ((std::vector<size_t>{2,3}), batchedTensor.capacities());
  EXPECT_FALSE(batchedTensor.empty());
  EXPECT_TRUE(batchedTensor.isBatched());
}

TYPED_TEST(TensorTest, Resetting) {
  using TensorType = typename TensorTest_Resetting_Test::TensorType;
  NNOTIFY("Testing resetting ...");
  
  TensorType tensor("tensor", {4,4,3,2}, true);
  NINFO(tensor);
  NINFO("Dimensions after construction: " << utils::vector_to_string(tensor.dimensions()));
  NINFO("Capacities after construction: " << utils::vector_to_string(tensor.capacities()));
  EXPECT_EQ((std::vector<size_t>{4, 4, 3, 2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{3, 2}), tensor.capacities());
  
  tensor.reset();
  NINFO(tensor);
  NINFO("Dimensions after reset(): " << utils::vector_to_string(tensor.dimensions()));
  NINFO("Capacities after reset(): " << utils::vector_to_string(tensor.capacities()));
  EXPECT_EQ((std::vector<size_t>{0}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), tensor.capacities());
  EXPECT_TRUE(tensor.empty());
  EXPECT_FALSE(tensor.isBatched());
}

TYPED_TEST(TensorTest, Clearing) {
  using TensorType = typename TensorTest_Clearing_Test::TensorType;
  NNOTIFY("Testing clearing batched tensor ...");
  TensorType tensor("tensor", {4,4,3,2}, true);
  NINFO(tensor);
  NINFO("Dimensions after construction: " << utils::vector_to_string(tensor.dimensions()));
  NINFO("Capacities after construction: " << utils::vector_to_string(tensor.capacities()));
  EXPECT_EQ((std::vector<size_t>{4, 4, 3, 2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{3, 2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{3, 3}), tensor.timesteps());
  EXPECT_EQ(2, tensor.batches());
  
  tensor.clear();
  NINFO(tensor);
  NINFO("Dimensions after clear(): " << utils::vector_to_string(tensor.dimensions()));
  NINFO("Capacities after clear(): " << utils::vector_to_string(tensor.capacities()));
  EXPECT_EQ((std::vector<size_t>{4, 4, 3, 0}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{3, 2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{}), tensor.timesteps());
  EXPECT_EQ(0, tensor.batches());
  EXPECT_TRUE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
}

TYPED_TEST(TensorTest, Filling) {
  using TensorType = typename TensorTest_Filling_Test::TensorType;
  NNOTIFY("Testing filling batched tensor ...");
  
  TensorType tensor("tensor", {4,4,3,2}, true);
  tensor.clear();
  NINFO(tensor);
  NINFO("Dimensions after construction and clear(): " << utils::vector_to_string(tensor.dimensions()));
  NINFO("Capacities after construction and clear(): " << utils::vector_to_string(tensor.capacities()));
  EXPECT_EQ((std::vector<size_t>{4, 4, 3, 0}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{3, 2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{}), tensor.timesteps());
  EXPECT_EQ(0, tensor.batches());
  EXPECT_EQ(0, tensor.totalTimeSteps());
  EXPECT_EQ(0, tensor.maxTimeSteps());
  EXPECT_EQ(0, tensor.minTimeSteps());
  
  tensor.fill();
  NINFO(tensor);
  NINFO("Dimensions after fill(): " << utils::vector_to_string(tensor.dimensions()));
  NINFO("Capacities after fill(): " << utils::vector_to_string(tensor.capacities()));
  EXPECT_EQ((std::vector<size_t>{4, 4, 3, 2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{3, 2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{3, 3}), tensor.timesteps());
  EXPECT_EQ(2, tensor.batches());
  EXPECT_EQ(6, tensor.totalTimeSteps());
  EXPECT_EQ(3, tensor.maxTimeSteps());
  EXPECT_EQ(3, tensor.minTimeSteps());
}

TYPED_TEST(TensorTest, Resizing) {
  using TensorType = typename TensorTest_Resizing_Test::TensorType;
  NNOTIFY("Testing resizing operations ...");
  
  TensorType tensor("tensor");
  NINFO("Before:\n" << tensor);
  
  size_t newdim0 = (size_t) this->prng.sampleIntegerUniform(2, 4);
  size_t newdim1 = (size_t) this->prng.sampleIntegerUniform(1, 4);
  size_t newdim2 = 4 + (size_t) this->prng.sampleIntegerUniform(1, 4);
  NINFO("Calling resize(" << utils::vector_to_string(std::vector<size_t>{newdim0, newdim1, newdim2}) << ") and setConstant(1.0)");
  tensor.resize({newdim0, newdim1, newdim2}, true);
  tensor.setConstant(1.0);
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{newdim0, newdim1, newdim2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{newdim1, newdim2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>(newdim2, newdim1)), tensor.timesteps());
  EXPECT_EQ(newdim2, tensor.batches());
  EXPECT_EQ(newdim2*newdim1, tensor.totalTimeSteps());
  EXPECT_EQ(newdim1, tensor.maxTimeSteps());
  EXPECT_EQ(newdim1, tensor.minTimeSteps());
  
  newdim0 = 5;
  newdim1 = 2;
  NINFO("Calling resize(" << utils::vector_to_string(std::vector<size_t>{newdim0, newdim1}) << ") and setConstant(1.0)");
  tensor.resize({newdim0, newdim1}, true);
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{1, newdim0, newdim1}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{newdim0, newdim1}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>(newdim1, newdim0)), tensor.timesteps());
  EXPECT_EQ(newdim1, tensor.batches());
  EXPECT_EQ(newdim1*newdim0, tensor.totalTimeSteps());
  EXPECT_EQ(newdim0, tensor.maxTimeSteps());
  EXPECT_EQ(newdim0, tensor.minTimeSteps());
}

TYPED_TEST(TensorTest, ResizingBatches) {
  using TensorType = typename TensorTest_ResizingBatches_Test::TensorType;
  NNOTIFY("Testing resizing batches operations ...");
  
  TensorType tensor("tensor");
  size_t newdim0 = (size_t) this->prng.sampleIntegerUniform(2, 4);
  size_t newdim1 = (size_t) this->prng.sampleIntegerUniform(1, 4);
  size_t newdim2 = 4 + (size_t) this->prng.sampleIntegerUniform(4, 7);
  tensor.resize({newdim0, newdim1, newdim2}, true);
  tensor.setConstant(1.0);
  NINFO("Before:\n" << tensor);
  
  auto newnewdim2 = newdim2 - (size_t)this->prng.sampleIntegerUniform(1, 3);
  NINFO("Calling resizeBatches(" << newnewdim2 << "):");
  tensor.resizeBatches(newnewdim2);
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{newdim0, newdim1, newnewdim2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{newdim1, newdim2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>(newnewdim2, newdim1)), tensor.timesteps());
  EXPECT_EQ(newnewdim2, tensor.batches());
  EXPECT_EQ(newnewdim2*newdim1, tensor.totalTimeSteps());
  EXPECT_EQ(newdim1, tensor.maxTimeSteps());
  EXPECT_EQ(newdim1, tensor.minTimeSteps());
}

TYPED_TEST(TensorTest, ReshapingBatches) {
  using TensorType = typename TensorTest_ReshapingBatches_Test::TensorType;
  NNOTIFY("Testing reshaping batches operations ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  TensorType tensorAlt("tensor", {4,3,2,3}, true);
  this->prng.seed(0);
  tensor.setRandomUnitUniform(this->prng);
  this->prng.seed(0);
  tensorAlt.setRandomUnitUniform(this->prng);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(tensor[k], tensorAlt[k]);
  }
  NINFO("Before:\n" << tensor);
  
  tensor.reshapeBatches(6, 1);
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,6,1}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{6, 1}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>(1, 6)), tensor.timesteps());
  EXPECT_EQ(1, tensor.batches());
  EXPECT_EQ(6, tensor.totalTimeSteps());
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(tensor[k], tensorAlt[k]);
  }
}

TYPED_TEST(TensorTest, Reshaping) {
  using TensorType = typename TensorTest_Reshaping_Test::TensorType;
  NNOTIFY("Testing reshaping operations ...");
  
  TensorType tensor("tensor");
  tensor.resize({4,3,2,10}, true);
  tensor.setConstant(1.0);
  tensor.resizeBatches(5);
  NINFO("Before:\n" << tensor);
  
  // Keeps batch info.
  tensor.reshape({2,6,2,10}, true);
  NINFO("Calling reshape(" << utils::vector_to_string(std::vector<size_t>{2,6,2,10}) << ", true)");
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{2,6,2,5}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,2,2,2,2}), tensor.timesteps());
  EXPECT_EQ(10, tensor.batchCapacity());
  EXPECT_EQ(10, tensor.totalTimeSteps());
  EXPECT_TRUE(tensor.isBatched());
  
  // Loses batch info. and assumes to be a full Tensor
  tensor.reshape({6,2,4,5}, true);
  NINFO("Calling reshape(" << utils::vector_to_string(std::vector<size_t>{6,2,4,5}) << ", true)");
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{6,2,4,5}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{4,4,4,4,4}), tensor.timesteps());
  EXPECT_EQ(5, tensor.batchCapacity());
  EXPECT_EQ(20, tensor.totalTimeSteps());
  EXPECT_TRUE(tensor.isBatched());
  
  tensor.resizeBatches(3);
  NINFO("Before:\n" << tensor);
  
  // Reshape into an unbatched Tensor
  tensor.reshape({6,4,2,5}, false);
  NINFO("Calling reshape(" << utils::vector_to_string(std::vector<size_t>{6,4,2,5}) << ", false)");
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{6,4,2,5}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), tensor.timesteps());
  EXPECT_EQ(0, tensor.batchCapacity());
  EXPECT_EQ(0, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.isBatched());
  
  // Reshape into a batched Tensor
  tensor.reshape({6,4,2,5}, true);
  NINFO("Calling reshape(" << utils::vector_to_string(std::vector<size_t>{6,4,2,5}) << ", true)");
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{6,4,2,5}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,2,2,2,2}), tensor.timesteps());
  EXPECT_EQ(5, tensor.batchCapacity());
  EXPECT_EQ(10, tensor.totalTimeSteps());
  EXPECT_TRUE(tensor.isBatched());
}

TYPED_TEST(TensorTest, Reserving) {
  using TensorType = typename TensorTest_Reserving_Test::TensorType;
  NNOTIFY("Testing reserving operations ...");
  
  TensorType tensor("tensor");
  tensor.reserve({4,3,2,3});
  NINFO("Reserved:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,0}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{}), tensor.timesteps());
  EXPECT_EQ(0, tensor.batches());
  EXPECT_EQ(0, tensor.totalTimeSteps());
  EXPECT_TRUE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  
  tensor.fill();
  tensor.setConstant(1.0);
  NINFO("Filled:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,3}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>(3,2)), tensor.timesteps());
  EXPECT_EQ(3, tensor.batches());
  EXPECT_EQ(6, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  
  tensor.reserve(2, 2);
  NINFO("Reduced:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,2}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{2,2}), tensor.timesteps());
  EXPECT_EQ(2, tensor.batches());
  EXPECT_EQ(4, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  
  tensor.reserve(4);
  NINFO("Extended:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,4}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{2,2}), tensor.timesteps());
  EXPECT_EQ(2, tensor.batches());
  EXPECT_EQ(4, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
}

TYPED_TEST(TensorTest, PushBackBatch) {
  using TensorType = typename TensorTest_PushBackBatch_Test::TensorType;
  NNOTIFY("Testing push-back batch operations ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.clear();
  NINFO(tensor);
  
  TensorType tensorOther("tensor_other", {4,3,2,1}, true);
  tensorOther.setConstant(2.0);
  NINFO(tensorOther);
  
  tensor.pushBack(tensorOther);
  NINFO("After push-back:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,1}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{2}), tensor.timesteps());
  EXPECT_EQ(1, tensor.batches());
  EXPECT_EQ(2, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
}

TYPED_TEST(TensorTest, PushBackTensor) {
  using TensorType = typename TensorTest_PushBackTensor_Test::TensorType;
  NNOTIFY("Testing push-back tensor operations ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.clear();
  NINFO(tensor);
  
  TensorType tensorOther("tensor_other", {4,3}, false);
  tensorOther.setConstant(2.0);
  NINFO(tensorOther);
  
  tensor.pushBack(0, tensorOther);
  NINFO("After push-back:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,1}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{1}), tensor.timesteps());
  EXPECT_EQ(1, tensor.batches());
  EXPECT_EQ(1, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  
  
  tensor.pushBack(0, tensorOther);
  NINFO("After push-back:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,1}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{2}), tensor.timesteps());
  EXPECT_EQ(1, tensor.batches());
  EXPECT_EQ(2, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  
  tensor.pushBack(0, tensorOther);
  NINFO("After push-back:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,3,1}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{3,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{3}), tensor.timesteps());
  EXPECT_EQ(1, tensor.batches());
  EXPECT_EQ(3, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
}

TYPED_TEST(TensorTest, PopBackBatch) {
  using TensorType = typename TensorTest_PopBackBatch_Test::TensorType;
  NNOTIFY("Testing pop-back batch operations ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.setConstant(1.0);
  NINFO(tensor);
  
  tensor.popBackBatch(tensor.batches()-1);
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,2,2}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{2,2}), tensor.timesteps());
  EXPECT_EQ(2, tensor.batches());
  EXPECT_EQ(4, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
}

TYPED_TEST(TensorTest, PopBackTimeSteps) {
  using TensorType = typename TensorTest_PopBackTimeSteps_Test::TensorType;
  NNOTIFY("Testing pop-back batch operations ...");
  
  TensorType tensor("tensor", {4,3,10,3}, true);
  tensor(0).setConstant(1.0);
  tensor(1).setConstant(2.0);
  tensor(2).setConstant(3.0);
  NINFO(tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,10,3}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{10,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{10,10,10}), tensor.timesteps());
  EXPECT_EQ(3, tensor.batches());
  EXPECT_EQ(30, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  
  auto steps_to_clear = tensor.timeStepCapacity();
  tensor.popBackTimeSteps(1, steps_to_clear);
  NINFO("After:\n" << tensor);
  EXPECT_EQ((std::vector<size_t>{4,3,10,3}), tensor.dimensions());
  EXPECT_EQ((std::vector<size_t>{10,3}), tensor.capacities());
  EXPECT_EQ((std::vector<size_t>{10,0,10}), tensor.timesteps());
  EXPECT_EQ(3, tensor.batches());
  EXPECT_EQ(20, tensor.totalTimeSteps());
  EXPECT_FALSE(tensor.empty());
  EXPECT_TRUE(tensor.isBatched());
  auto batch = tensor(1);
  for (size_t k = 0; k < batch.size(); ++k) {
    EXPECT_EQ(0, batch[k]);
  }
}

TYPED_TEST(TensorTest, FlattenedBatches) {
  using TensorType = typename TensorTest_FlattenedBatches_Test::TensorType;
  NNOTIFY("Testing flattened batches operations ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor.setConstant(2.0);
  NINFO(tensor);
  
  TensorType tensorOther("tensor_other");
  NINFO(tensorOther);
  
  tensorOther = tensor.getFlattenedBatches();
  NINFO("After:\n" << tensorOther);
  EXPECT_EQ((std::vector<size_t>{4,3,6,1}), tensorOther.dimensions());
  EXPECT_EQ((std::vector<size_t>{6,1}), tensorOther.capacities());
  EXPECT_EQ((std::vector<size_t>{6}), tensorOther.timesteps());
  EXPECT_EQ(1, tensorOther.batches());
  EXPECT_EQ(6, tensorOther.totalTimeSteps());
  EXPECT_FALSE(tensorOther.empty());
  EXPECT_TRUE(tensorOther.isBatched());
}

TYPED_TEST(TensorTest, Cloning) {
  using TensorType = typename TensorTest_Cloning_Test::TensorType;
  NNOTIFY("Testing cloning ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor(1).setConstant(1);
  tensor(2).setConstant(2);
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt.clone(tensor);
  NINFO("Destination after:\n" << tensorAlt);
  EXPECT_EQ(tensorAlt.name(), tensor.name());
  EXPECT_EQ(tensorAlt.dimensions(), tensor.dimensions());
  EXPECT_EQ(tensorAlt.capacities(), tensor.capacities());
  EXPECT_EQ(tensorAlt.timesteps(), tensor.timesteps());
  EXPECT_EQ(tensorAlt.batches(), tensor.batches());
  EXPECT_EQ(tensorAlt.totalTimeSteps(), tensor.totalTimeSteps());
  EXPECT_FALSE(tensorAlt.hasSameStorageWith(tensor));
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, Copying) {
  using TensorType = typename TensorTest_Copying_Test::TensorType;
  NNOTIFY("Testing copying ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor(1).setConstant(1);
  tensor(2).setConstant(2);
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt.copy(tensor);
  NINFO("Destination after:\n" << tensorAlt);
  EXPECT_EQ("tensor_alt", tensorAlt.name());
  EXPECT_EQ(tensorAlt.dimensions(), tensor.dimensions());
  EXPECT_EQ(tensorAlt.capacities(), tensor.capacities());
  EXPECT_EQ(tensorAlt.timesteps(), tensor.timesteps());
  EXPECT_EQ(tensorAlt.batches(), tensor.batches());
  EXPECT_EQ(tensorAlt.totalTimeSteps(), tensor.totalTimeSteps());
  EXPECT_FALSE(tensorAlt.hasSameStorageWith(tensor));
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, AssignmentFromTensor) {
  using TensorType = typename TensorTest_AssignmentFromTensor_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from Tensor ...");
  
  TensorType tensor("tensor", {4,3,2,3}, false);
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt = tensor;
  NINFO("Destination after:\n" << tensorAlt);
  EXPECT_EQ("tensor_alt", tensorAlt.name());
  EXPECT_EQ(tensor.dimensions(), tensorAlt.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), tensorAlt.capacities());
  EXPECT_EQ(tensor.isBatched(), tensorAlt.isBatched());
  EXPECT_FALSE(tensorAlt.isBatched());
  EXPECT_EQ(tensor.batches(), tensorAlt.batches());
  EXPECT_EQ(0, tensorAlt.batches());
  EXPECT_TRUE(tensorAlt.timesteps().empty());
  EXPECT_EQ(tensor.timesteps(), tensorAlt.timesteps());
  EXPECT_EQ(tensor.totalTimeSteps(), tensorAlt.totalTimeSteps());
  EXPECT_EQ(72, tensorAlt.sizeOfDatum());
  EXPECT_EQ(72, tensorAlt.sizeOfBatch());
  EXPECT_EQ(72, tensorAlt.size());
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, AssignmentFromTensorBatched) {
  using TensorType = typename TensorTest_AssignmentFromTensorBatched_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from Tensor ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt = tensor;
  NINFO("Destination after:\n" << tensorAlt);
  EXPECT_EQ("tensor_alt", tensorAlt.name());
  EXPECT_EQ(tensor.dimensions(), tensorAlt.dimensions());
  EXPECT_EQ(tensor.capacities(), tensorAlt.capacities());
  EXPECT_EQ(tensor.isBatched(), tensorAlt.isBatched());
  EXPECT_TRUE(tensorAlt.isBatched());
  EXPECT_EQ(tensor.batches(), tensorAlt.batches());
  EXPECT_EQ(3, tensorAlt.batches());
  EXPECT_EQ((std::vector<size_t>{2,2,2}), tensorAlt.timesteps());
  EXPECT_EQ(tensor.timesteps(), tensorAlt.timesteps());
  EXPECT_EQ(tensor.maxTimeSteps(), tensorAlt.maxTimeSteps());
  EXPECT_EQ(tensor.totalTimeSteps(), tensorAlt.totalTimeSteps());
  EXPECT_EQ(12, tensorAlt.sizeOfDatum());
  EXPECT_EQ(24, tensorAlt.sizeOfBatch());
  EXPECT_EQ(72, tensorAlt.size());
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, AssignmentFromTensorMapDatum) {
  using TensorType = typename TensorTest_AssignmentFromTensorMapDatum_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from TensorMap ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor(0,0).setConstant(0);
  tensor(1,0).setConstant(1);
  tensor(0,1).setConstant(2);
  tensor(1,1).setConstant(3);
  tensor(0,2).setConstant(4);
  tensor(1,2).setConstant(5);
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt = tensor(0,0);
  NINFO("Destination after:\n" << tensorAlt);
  EXPECT_EQ("tensor_alt", tensorAlt.name());
  EXPECT_EQ((std::vector<size_t>{4,3}), tensorAlt.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), tensorAlt.capacities());
  EXPECT_FALSE(tensorAlt.isBatched());
  EXPECT_EQ(0, tensorAlt.batches());
  EXPECT_EQ((std::vector<size_t>{}), tensorAlt.timesteps());
  EXPECT_EQ(0, tensorAlt.maxTimeSteps());
  EXPECT_EQ(0, tensorAlt.totalTimeSteps());
  EXPECT_EQ(12, tensorAlt.sizeOfDatum());
  EXPECT_EQ(12, tensorAlt.sizeOfBatch());
  EXPECT_EQ(12, tensorAlt.size());
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, AssignmentFromTensorMapBatch) {
  using TensorType = typename TensorTest_AssignmentFromTensorMapBatch_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from TensorMap ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  tensor(0,0).setConstant(0);
  tensor(1,0).setConstant(1);
  tensor(0,1).setConstant(2);
  tensor(1,1).setConstant(3);
  tensor(0,2).setConstant(4);
  tensor(1,2).setConstant(5);
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt = tensor(0);
  NINFO("Destination after:\n" << tensorAlt);
  EXPECT_EQ("tensor_alt", tensorAlt.name());
  EXPECT_EQ((std::vector<size_t>{4,3,2,1}), tensorAlt.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,1}), tensorAlt.capacities());
  EXPECT_TRUE(tensorAlt.isBatched());
  EXPECT_EQ(1, tensorAlt.batches());
  EXPECT_EQ((std::vector<size_t>{2}), tensorAlt.timesteps());
  EXPECT_EQ(tensor.maxTimeSteps(), tensorAlt.maxTimeSteps());
  EXPECT_EQ(2, tensorAlt.totalTimeSteps());
  EXPECT_EQ(12, tensorAlt.sizeOfDatum());
  EXPECT_EQ(24, tensorAlt.sizeOfBatch());
  EXPECT_EQ(24, tensorAlt.size());
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, AssignmentFromTensorBatchBlock) {
  using TensorType = typename TensorTest_AssignmentFromTensorBatchBlock_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from TensorMap ...");
  
  TensorType tensor("tensor", {4,3,2,3}, true);
  for (size_t b = 0; b < tensor.batches(); ++b) {
    for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
      tensor(t,b).setConstant(b*2+t);
    }
  }
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt = tensor.batch_block(0,2);
  NINFO("Destination after:\n" << tensorAlt);
  
  EXPECT_EQ("tensor_alt", tensorAlt.name());
  EXPECT_EQ((std::vector<size_t>{4,3,2,2}), tensorAlt.dimensions());
  EXPECT_EQ((std::vector<size_t>{2,2}), tensorAlt.capacities());
  EXPECT_TRUE(tensorAlt.isBatched());
  EXPECT_EQ(2, tensorAlt.batches());
  EXPECT_EQ((std::vector<size_t>{2,2}), tensorAlt.timesteps());
  EXPECT_EQ(tensor.maxTimeSteps(), tensorAlt.maxTimeSteps());
  EXPECT_EQ(4, tensorAlt.totalTimeSteps());
  EXPECT_EQ(12, tensorAlt.sizeOfDatum());
  EXPECT_EQ(24, tensorAlt.sizeOfBatch());
  EXPECT_EQ(48, tensorAlt.size());
  EXPECT_FALSE(tensorAlt.empty());
}

TYPED_TEST(TensorTest, AssignmentFromTensorBlock) {
  using TensorType = typename TensorTest_AssignmentFromTensorBlock_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from TensorMap ...");
  
  TensorType tensor("tensor", {4,3,5,3}, true);
  for (size_t b = 0; b < tensor.batches(); ++b) {
    for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
      tensor(t,b).setConstant(b*5+t);
    }
  }
  NINFO("Source:\n" << tensor);
  
  TensorType tensorAlt("tensor_alt");
  NINFO("Destination before:\n" << tensorAlt);
  
  tensorAlt = tensor.block(0,6);
  NINFO("Destination after:\n" << tensorAlt);
}

TYPED_TEST(TensorTest, AssignmentFromTensorFlowTensor) {
  using TensorType = typename TensorTest_AssignmentFromTensorFlowTensor_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from tensorflow::Tensor ...");
//  TODO
}

TYPED_TEST(TensorTest, AssignmentFromTensorFlowTensorBatched) {
  using TensorType = typename TensorTest_AssignmentFromTensorFlowTensorBatched_Test::TensorType;
  NNOTIFY("Testing assignment Tensor from tensorflow::Tensor ...");
// TODO
}

TYPED_TEST(TensorTest, Randomizing) {
  using TensorType = typename TensorTest_Randomizing_Test::TensorType;
  NNOTIFY("Testing randomizing Tensor ...");
  // TODO
}

TYPED_TEST(TensorTest, ValueChecks) {
  using ScalarType = typename TensorTest_ValueChecks_Test::ScalarType;
  using TensorType = typename TensorTest_ValueChecks_Test::TensorType;
  NNOTIFY("Testing hasNaN(), hasInf() operations ...");
  
  TensorType tensor("tensor", {5,6,3,2}, true);
  tensor.setZero();
  NINFO("Initial:\n" << tensor);
  EXPECT_FALSE(tensor.hasNaN());
  EXPECT_FALSE(tensor.hasInf());
  
  tensor[0] = std::numeric_limits<ScalarType>::infinity();
  NINFO("With Inf:\n" << tensor);
  EXPECT_TRUE(tensor.hasInf());
  
  tensor[0] = std::numeric_limits<ScalarType>::quiet_NaN();
  NINFO("With NaN:\n" << tensor);
  EXPECT_TRUE(tensor.hasNaN());
}

TYPED_TEST(TensorTest, ScalarArithmetic) {
  using TensorType = typename TensorTest_ScalarArithmetic_Test::TensorType;
  NNOTIFY("Testing {+,-,*,/} scalar arithmetic operators ...");
  
  TensorType tensor("tensor", {5, 6, 3, 2}, true);
  auto maxIdx = tensor.size();
  tensor.setConstant(3.0);
  
  tensor *= 2.0;
  NINFO("tensor *= 2:\n" << tensor);
  EXPECT_EQ(6.0, tensor[this->prng.sampleIntegerUniform(0, maxIdx)]);
  
  tensor -= 1.0;
  NINFO("tensor -= 1:\n" << tensor);
  EXPECT_EQ(5.0, tensor[this->prng.sampleIntegerUniform(0, maxIdx)]);
  
  tensor += 2.0;
  NINFO("tensor += 2:\n" << tensor);
  EXPECT_EQ(7.0, tensor[this->prng.sampleIntegerUniform(0, maxIdx)]);
  
  tensor /= 7.0;
  NINFO("tensor /= 7:\n" << tensor);
  EXPECT_EQ(1.0, tensor[this->prng.sampleIntegerUniform(0, maxIdx)]);
  
}

TYPED_TEST(TensorTest, TensorArithmetic) {
  using TensorType = typename TensorTest_TensorArithmetic_Test::TensorType;
  NNOTIFY("Testing {+,-,*,/} tensor arithmetic operators ...");
  
  TensorType tensor("tensor", {5, 6, 3, 2}, true);
  tensor.setConstant(5.0);
  
  TensorType tensorAlt("tensor_alt", {5, 6, 3, 2}, true);
  tensorAlt.setConstant(2.0);
  
  tensor += tensorAlt;
  NINFO("tensor += tensorAlt:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(7, tensor[k]);
  }
  
  tensor -= tensorAlt;
  NINFO("tensor -= tensorAlt:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(5, tensor[k]);
  }
  
  tensor *= tensorAlt;
  NINFO("tensor *= tensorAlt:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(10, tensor[k]);
  }
  
  tensor /= tensorAlt;
  NINFO("tensor /= tensorAlt:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(5, tensor[k]);
  }
}

TYPED_TEST(TensorTest, TensorBinaryArithmetic) {
  using ScalarType = typename TensorTest_TensorBinaryArithmetic_Test::ScalarType;
  using TensorType = typename TensorTest_TensorBinaryArithmetic_Test::TensorType;
  NNOTIFY("Testing {+,-,*,/} tensor binary arithmetic operators ...");
  
  TensorType tensor("tensor", {5, 6, 3, 2}, true);
  tensor.setConstant(0.0);
  
  TensorType tensorAlt("tensor_alt", {5, 6, 3, 2}, true);
  tensorAlt.setConstant(2.0);
  
  TensorType tensorOther("tensor_other", {5, 6, 3, 2}, true);
  tensorOther.setConstant(3.0);
  
  tensor = tensorAlt + tensorOther;
  NINFO("tensor = tensorAlt + :\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(5, tensor[k]);
  }
  
  tensor = tensorAlt - tensorOther;
  NINFO("tensor = tensorAlt - tensorOther:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(-1, tensor[k]);
  }
  
  tensor = tensorAlt * tensorOther;
  NINFO("tensor = tensorAlt * tensorOther:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(6, tensor[k]);
  }
  
  tensor = tensorOther / tensorAlt;
  NINFO("tensor = tensorOther / tensorAlt:\n" << tensor);
  for (size_t k = 0; k < tensor.size(); ++k) {
    EXPECT_EQ(static_cast<ScalarType>(1.5), tensor[k]);
  }
}

TYPED_TEST(TensorTest, EigenMap) {
  using TensorType = typename TensorTest_EigenMap_Test::TensorType;
  NNOTIFY("Testing mapping to eigen matrices and tensors ...");
  
  TensorType tensor("tensor", {5, 6, 3, 2}, true);
  tensor.setConstant(1.0);
  
  auto matMap = tensor.asEigenMatrix();
  NINFO("As an Eigen::Matrix:\n" << matMap);
  EXPECT_EQ(5, matMap.rows());
  EXPECT_EQ(36, matMap.cols());
  EXPECT_EQ(180, matMap.sum());
  
  auto tenMap = tensor.template asEigenTensor<4>();
  NINFO("As an Eigen::Tensor:\n" << tenMap);
}

TYPED_TEST(TensorTest, SharedStorage) {
  using TensorType = typename TensorTest_SharedStorage_Test::TensorType;
  NNOTIFY("Testing shared storage checking ...");
  
  TensorType tensor("tensor", {5, 6, 3, 2}, true);
  tensor.setConstant(1.0);
  TensorType other("other");
  
  other.clone(tensor);
  NINFO("After cloning:" << other);
  EXPECT_FALSE(other.hasSameStorageWith(tensor));
  
  other.copy(tensor);
  NINFO("After copying:" << other);
  EXPECT_FALSE(other.hasSameStorageWith(tensor));
  
  other = tensor;
  NINFO("After assignment:" << other);
  EXPECT_TRUE(other.hasSameStorageWith(tensor));
}

TYPED_TEST(TensorTest, Shuffling) {
  using TensorType = typename TensorTest_Shuffling_Test::TensorType;
  NNOTIFY("Testing shuffling operations ...");
  
  // Create an example tensor
  constexpr size_t T = 10;
  constexpr size_t B = 3;
  TensorType tensor("tensor", {3, 2, 2, T, B}, true);
  for (size_t b = 0; b < tensor.batches(); ++b) {
    for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
      tensor(t,b).setConstant(b*T+t);
    }
  }
  NINFO("BEFORE:" << tensor);
  
  // Shuffle indeces
  Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(T*B, 0, T*B);
  NINFO("Indices: " << indices.transpose());
  std::shuffle(indices.data(), indices.data() + indices.size(), this->prng.generator());
  NINFO("Shuffled Indices: " << indices.transpose());
  
  // Shuffle data
  tensor.shuffle(indices);
  NINFO("AFTER:" << tensor);
}

} // namespace tests
} // namespace noesis

/* EOF */
