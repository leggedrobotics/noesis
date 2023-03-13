/*!
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

template<typename ScalarType_>
class TensorAllocatorTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  using ScalarType = ScalarType_;
  using TensorType = noesis::Tensor<ScalarType>;
  // We use default constructor/destructor
  TensorAllocatorTest() = default;
  ~TensorAllocatorTest() = default;
protected:
  static noesis::math::RandomNumberGenerator<ScalarType> prng;
};

// Initialize the static RandomNumberGenerator member
template<typename ScalarType_>
noesis::math::RandomNumberGenerator<ScalarType_> TensorAllocatorTest<ScalarType_>::prng;

// Test over the supported arithmetic types
using ScalarTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(TensorAllocatorTest, ScalarTypes);

/*
 * Tests
 */

TYPED_TEST(TensorAllocatorTest, SpecialCases) {
  using ScalarType = typename TensorAllocatorTest_SpecialCases_Test::ScalarType;
  using TensorType = typename TensorAllocatorTest_SpecialCases_Test::TensorType;
  DNNOTIFY("Testing Allocating (Special cases)...");
  
  tensorflow::Tensor tensor;
  NINFO("Underlying tensor" << tensor.DebugString())
  TensorAllocator<ScalarType > allocator(&tensor);
  NINFO("Allocator info" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{0}), allocator.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(0, allocator.timeStepCapacity());
  EXPECT_EQ(0, allocator.batchCapacity());
  EXPECT_EQ(0, allocator.sizeOfDatum());
  EXPECT_EQ(0, allocator.sizeOfBatch());
  EXPECT_EQ(0, allocator.size());
  
  allocator.allocateScalar();
  NINFO("Allocator info after allocateScalar()" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{}), allocator.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(0, allocator.timeStepCapacity());
  EXPECT_EQ(0, allocator.batchCapacity());
  EXPECT_EQ(1, allocator.sizeOfDatum());
  EXPECT_EQ(1, allocator.sizeOfBatch());
  EXPECT_EQ(1, allocator.size());
  EXPECT_EQ(0, tensor.template flat<ScalarType >().data()[0]);
  EXPECT_FALSE(allocator.isBatched());
  
  allocator.allocateEmpty();
  NINFO("Allocator info after allocateEmpty()" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{0}), allocator.dimensions());
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(0, allocator.timeStepCapacity());
  EXPECT_EQ(0, allocator.batchCapacity());
  EXPECT_EQ(0, allocator.sizeOfDatum());
  EXPECT_EQ(0, allocator.sizeOfBatch());
  EXPECT_EQ(0, allocator.size());
  EXPECT_FALSE(allocator.isBatched());
  
}

TYPED_TEST(TensorAllocatorTest, AllocateSimple) {
  using ScalarType = typename TensorAllocatorTest_AllocateSimple_Test::ScalarType;
  using TensorType = typename TensorAllocatorTest_AllocateSimple_Test::TensorType;
  DNNOTIFY("Testing AllocateSimple ...");
  
  tensorflow::Tensor tensor;
  TensorAllocator<ScalarType > allocator(&tensor);
  
  allocator.allocateSimple({5,1,6,2});
  NINFO("Allocator info after allocateSimple({5,1,6,2})" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{5,1,6,2}), allocator.dimensions());
  EXPECT_EQ(2, tensor.dim_size(0));
  EXPECT_EQ(6, tensor.dim_size(1));
  EXPECT_EQ(1, tensor.dim_size(2));
  EXPECT_EQ(5, tensor.dim_size(3));
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(0, allocator.timeStepCapacity());
  EXPECT_EQ(0, allocator.batchCapacity());
  EXPECT_EQ(60, allocator.sizeOfDatum());
  EXPECT_EQ(60, allocator.sizeOfBatch());
  EXPECT_EQ(60, allocator.size());
  EXPECT_FALSE(allocator.isBatched());
  for (size_t i = 0; i<allocator.size(); i++){
    EXPECT_EQ(0.0, tensor.template flat<ScalarType >().data()[i]);
  }
  
  for (size_t i = 0; i<allocator.size(); i++){
    tensor.template flat<ScalarType >().data()[i] = i;
  }
  NINFO("tensor filled with data\n" << (tensor.template tensor<ScalarType,4>()));
  
  allocator.allocateSimple({5,1,3,4});
  NINFO("Allocator info after allocateSimple({5,1,3,4})" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{5,1,3,4}), allocator.dimensions());
  EXPECT_EQ(4, tensor.dim_size(0));
  EXPECT_EQ(3, tensor.dim_size(1));
  EXPECT_EQ(1, tensor.dim_size(2));
  EXPECT_EQ(5, tensor.dim_size(3));
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(0, allocator.timeStepCapacity());
  EXPECT_EQ(0, allocator.batchCapacity());
  EXPECT_EQ(60, allocator.sizeOfDatum());
  EXPECT_EQ(60, allocator.sizeOfBatch());
  EXPECT_EQ(60, allocator.size());
  EXPECT_FALSE(allocator.isBatched());
  NINFO("tensor data\n" << (tensor.template tensor<ScalarType,4>()));
  for (size_t i = 0; i<allocator.size(); i++){
    EXPECT_EQ(i, tensor.template flat<ScalarType >().data()[i]);
  }
  
  allocator.allocateSimple({2,3});
  NINFO("Allocator info after allocateSimple({2,3})" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{2,3}), allocator.dimensions());
  EXPECT_EQ(3, tensor.dim_size(0));
  EXPECT_EQ(2, tensor.dim_size(1));
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(0, allocator.timeStepCapacity());
  EXPECT_EQ(0, allocator.batchCapacity());
  EXPECT_EQ(6, allocator.sizeOfDatum());
  EXPECT_EQ(6, allocator.sizeOfBatch());
  EXPECT_EQ(6, allocator.size());
  EXPECT_FALSE(allocator.isBatched());
  NINFO("tensor data\n" << (tensor.template tensor<ScalarType,2>()));
  
  for (size_t i = 0; i<allocator.size(); i++){
    EXPECT_EQ(0.0, tensor.template flat<ScalarType >().data()[i]);
  }
}

TYPED_TEST(TensorAllocatorTest, AllocateBatched) {
  using ScalarType = typename TensorAllocatorTest_AllocateBatched_Test::ScalarType;
  using TensorType = typename TensorAllocatorTest_AllocateBatched_Test::TensorType;
  DNNOTIFY("Testing AllocateBatched ...");
  
  tensorflow::Tensor tensor;
  TensorAllocator<ScalarType > allocator(&tensor);
  
  // new_capacity < new_dimensions.back()
  allocator.allocateBatched({1,2,3,4}, 2);
  NINFO("Allocator info after allocateBatched({1,2,3,4}, 2)" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{1,2,3,4}), allocator.dimensions());
  EXPECT_EQ(12, tensor.dim_size(0));
  EXPECT_EQ(2, tensor.dim_size(1));
  EXPECT_EQ(1, tensor.dim_size(2));
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(3, allocator.timeStepCapacity());
  EXPECT_EQ(4, allocator.batchCapacity());
  EXPECT_EQ(2, allocator.sizeOfDatum());
  EXPECT_EQ(2 * 3, allocator.sizeOfBatch());
  EXPECT_EQ(2 * 3 * 4, allocator.size());
  EXPECT_TRUE(allocator.isBatched());
  for (size_t i = 0; i<allocator.size(); i++){
    EXPECT_EQ(0.0, tensor.template flat<ScalarType >().data()[i]);
  }
  
  // new_capacity = 0
  allocator.allocateEmpty();
  allocator.allocateBatched({1,2,3,4}, 0);
  NINFO("Allocator info after allocateBatched({1,2,3,4}, 0)" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{1,2,3,4}), allocator.dimensions());
  EXPECT_EQ(12, tensor.dim_size(0));
  EXPECT_EQ(2, tensor.dim_size(1));
  EXPECT_EQ(1, tensor.dim_size(2));
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(3, allocator.timeStepCapacity());
  EXPECT_EQ(4, allocator.batchCapacity());
  EXPECT_EQ(2, allocator.sizeOfDatum());
  EXPECT_EQ(2 * 3, allocator.sizeOfBatch());
  EXPECT_EQ(2 * 3 * 4, allocator.size());
  EXPECT_TRUE(allocator.isBatched());
  for (size_t i = 0; i<allocator.size(); i++){
    EXPECT_EQ(0.0, tensor.template flat<ScalarType >().data()[i]);
  }
  
  // new_capacity > new_dimensions.back()
  allocator.allocateEmpty();
  allocator.allocateBatched({1,2,3,4}, 8);
  NINFO("Allocator info after allocateBatched({1,2,3,4}, 8)" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{1,2,3,4}), allocator.dimensions());
  EXPECT_EQ(24, tensor.dim_size(0));
  EXPECT_EQ(2, tensor.dim_size(1));
  EXPECT_EQ(1, tensor.dim_size(2));
  EXPECT_EQ((std::vector<size_t>{}), allocator.timesteps());
  EXPECT_EQ(3, allocator.timeStepCapacity());
  EXPECT_EQ(8, allocator.batchCapacity());
  EXPECT_EQ(2, allocator.sizeOfDatum());
  EXPECT_EQ(2 * 3, allocator.sizeOfBatch());
  EXPECT_EQ(2 * 3 * 4, allocator.size());
  EXPECT_TRUE(allocator.isBatched());
  for (size_t i = 0; i<allocator.size(); i++){
    EXPECT_EQ(0.0, tensor.template flat<ScalarType >().data()[i]);
  }
  
  for (size_t i = 0; i<tensor.NumElements(); i++){
    tensor.template flat<ScalarType >().data()[i] = i % 6;
  }
  
  DNINFO("Filled tensor\n" << (tensor.template tensor<ScalarType,3>()));
  // conservative resize
  allocator.allocateBatched({1,2,2,2}, 0);
  NINFO("Allocator info after allocateBatched({1,2,2,2}, 0)" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{1,2,2,2}), allocator.dimensions());
  EXPECT_EQ(16, tensor.dim_size(0));
  EXPECT_EQ(2, tensor.dim_size(1));
  EXPECT_EQ(1, tensor.dim_size(2));
  EXPECT_EQ(8, allocator.batchCapacity());
  EXPECT_EQ(2, allocator.sizeOfDatum());
  EXPECT_EQ(2 * 2, allocator.sizeOfBatch());
  EXPECT_EQ(2 * 2 * 2, allocator.size());
  
  DNINFO("Filled tensor\n" << (tensor.template tensor<ScalarType,3>()));
  size_t position = 0;
  for (size_t i=0; i<2; i++){
    for (size_t j=0; j<4; j++){
      EXPECT_EQ(j, tensor.template flat<ScalarType >().data()[position++]);
    }
  }
  
  // non conservative resize
  allocator.allocateBatched({4,3,2,1}, 0);
  NINFO("Allocator info after allocateSimple({4,3,2,1}, 0)" << allocator.info());
  EXPECT_EQ((std::vector<size_t>{4,3,2,1}), allocator.dimensions());
  EXPECT_EQ(2, tensor.dim_size(0));
  EXPECT_EQ(3, tensor.dim_size(1));
  EXPECT_EQ(4, tensor.dim_size(2));
  EXPECT_EQ(1, allocator.batchCapacity());
  EXPECT_EQ(12, allocator.sizeOfDatum());
  EXPECT_EQ(12 * 2, allocator.sizeOfBatch());
  EXPECT_EQ(12 * 2 * 1, allocator.size());
}

} // namespace tests
} // namespace noesis

/* EOF */
