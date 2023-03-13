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
#include <noesis/framework/core/TensorTuple.hpp>

namespace noesis {
namespace tests {

/**
 * Define Test Fixture
 */

template<typename ScalarType_>
class TensorTupleTest : public ::testing::Test
{
protected:

  // Declare fixture aliases
  using ScalarType = ScalarType_;
  using TensorType = noesis::Tensor<ScalarType>;
  using TensorTupleType = noesis::TensorTuple<ScalarType>;
  using DimensionsVectorType = typename TensorTupleType::DimensionsVectorType;
  TensorTupleTest() = default;
  ~TensorTupleTest() = default;
protected:
  static noesis::math::RandomNumberGenerator<ScalarType> prng;
};

// Initialize the static RandomNumberGenerator member
template<typename ScalarType_>
noesis::math::RandomNumberGenerator<ScalarType_> TensorTupleTest<ScalarType_>::prng;

// Test over the supported arithmetic types
using ScalarTypes = ::testing::Types<float, double>;

// Declare the typed-test
TYPED_TEST_CASE(TensorTupleTest, ScalarTypes);

/*
 * Tests
 */

TYPED_TEST(TensorTupleTest, Creation) {
  using TensorTupleType = typename TensorTupleTest_Creation_Test::TensorTupleType;
  NNOTIFY("Testing creation ...");
  
  // Fully specified tuple from TensorsSpec
  {
    TensorsSpec spec = { {"elem0", {1,3}}, {"elem1", {4,5,6}} };
    TensorTupleType tuple("tuple", spec, 3, 2);
    NINFO(tuple);
    EXPECT_EQ(2, tuple.size());
    EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
    EXPECT_EQ("tuple", tuple.scope());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{1,3,3,2},{4,5,6,3,2}}), tuple.dimensions());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{1,3},{4,5,6}}), tuple.datumDimensions());
    EXPECT_EQ(3, tuple.timesteps());
    EXPECT_EQ(2, tuple.batches());
    EXPECT_FALSE(tuple.empty());
    EXPECT_TRUE(tuple.isBatched());
  }
  
  // Fully specified tuple
  {
    TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{1,3},{4,5,6}}, 3, 2);
    NINFO(tuple);
    EXPECT_EQ(2, tuple.size());
    EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
    EXPECT_EQ("tuple", tuple.scope());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{1,3,3,2},{4,5,6,3,2}}), tuple.dimensions());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{1,3},{4,5,6}}), tuple.datumDimensions());
    EXPECT_EQ(3, tuple.timesteps());
    EXPECT_EQ(2, tuple.batches());
    EXPECT_FALSE(tuple.empty());
    EXPECT_TRUE(tuple.isBatched());
  }
  
  // Unnamed tuple
  {
    TensorTupleType tuple({{1,3},{4,5,6}}, 3, 2);
    NINFO(tuple);
    EXPECT_EQ(2, tuple.size());
    EXPECT_EQ((std::vector<std::string>{"",""}), tuple.names());
    EXPECT_EQ("", tuple.scope());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{1,3,3,2},{4,5,6,3,2}}), tuple.dimensions());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{1,3},{4,5,6}}), tuple.datumDimensions());
    EXPECT_EQ(3, tuple.timesteps());
    EXPECT_EQ(2, tuple.batches());
    EXPECT_FALSE(tuple.empty());
    EXPECT_TRUE(tuple.isBatched());
  }
  
  // Unspecified tuple
  {
    TensorTupleType tuple("tuple", 3, 2);
    NINFO(tuple);
    EXPECT_EQ(0, tuple.size());
    EXPECT_EQ((std::vector<std::string>{}), tuple.names());
    EXPECT_EQ("tuple", tuple.scope());
    EXPECT_EQ((std::vector<std::vector<size_t>>{}), tuple.dimensions());
    EXPECT_EQ(3, tuple.timesteps());
    EXPECT_EQ(2, tuple.batches());
    EXPECT_TRUE(tuple.empty());
    EXPECT_TRUE(tuple.isBatched());
  }
  
  // Scoped empty tuple
  {
    TensorTupleType tuple("tuple");
    NINFO(tuple);
    EXPECT_EQ(0, tuple.size());
    EXPECT_EQ((std::vector<std::string>{}), tuple.names());
    EXPECT_EQ("tuple", tuple.scope());
    EXPECT_EQ((std::vector<std::vector<size_t>>{}), tuple.dimensions());
    EXPECT_EQ(0, tuple.timesteps());
    EXPECT_EQ(0, tuple.batches());
    EXPECT_TRUE(tuple.empty());
    EXPECT_FALSE(tuple.isBatched());
  }
  
  // Empty tuple
  {
    TensorTupleType tuple;
    NINFO(tuple);
    EXPECT_EQ(0, tuple.size());
    EXPECT_EQ((std::vector<std::string>{}), tuple.names());
    EXPECT_EQ("", tuple.scope());
    EXPECT_EQ((std::vector<std::vector<size_t>>{}), tuple.dimensions());
    EXPECT_EQ(0, tuple.timesteps());
    EXPECT_EQ(0, tuple.batches());
    EXPECT_TRUE(tuple.empty());
    EXPECT_FALSE(tuple.isBatched());
  }
}

TYPED_TEST(TensorTupleTest, GetSpec) {
  using TensorTupleType = typename TensorTupleTest_GetSpec_Test::TensorTupleType;
  NNOTIFY("Testing getting TensorsSpec ...");
  
  TensorsSpec spec = { {"elem0", {4,3}}, {"elem1", {4,5,6}} };
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  NINFO(tuple);
  EXPECT_EQ(spec, tuple.spec());
}

TYPED_TEST(TensorTupleTest, SetFromSpec) {
  using TensorTupleType = typename TensorTupleTest_SetFromSpec_Test::TensorTupleType;
  NNOTIFY("Testing setting from TensorsSpec ...");
  
  TensorsSpec spec = { {"elem0", {4,3}}, {"elem1", {4,5,6}} };
  TensorTupleType tuple("tuple", 3, 2);
  NINFO("Before: " << tuple);
  tuple.setFromSpec(spec);
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3,3,2},{4,5,6,3,2}}), tuple.dimensions());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(3, tuple.timesteps());
  EXPECT_EQ(2, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
}

TYPED_TEST(TensorTupleTest, AddingTensor) {
  using TensorTupleType = typename TensorTupleTest_AddingTensor_Test::TensorTupleType;
  NNOTIFY("Testing adding tensor ...");
  
  TensorTupleType tuple("tuple", 3, 2);
  NINFO("Before: " << tuple);
  tuple.addTensor("elem0", {4,3});
  tuple.addTensor("elem1", {4,5,6});
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3,3,2},{4,5,6,3,2}}), tuple.dimensions());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(3, tuple.timesteps());
  EXPECT_EQ(2, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
}

TYPED_TEST(TensorTupleTest, Resetting) {
  using TensorTupleType = typename TensorTupleTest_Resetting_Test::TensorTupleType;
  NNOTIFY("Testing setting from TensorsSpec ...");
  
  TensorsSpec spec = { {"elem0", {4,3}}, {"elem1", {4,5,6}} };
  TensorTupleType tuple("tuple", spec, 3, 2);
  NINFO("Before: " << tuple);
  tuple.reset();
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(3, tuple.timesteps());
  EXPECT_EQ(2, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{0}), tuple[k].dimensions());
    EXPECT_EQ((std::vector<size_t>{}), tuple[k].capacities());
    EXPECT_TRUE(tuple[k].empty());
    EXPECT_FALSE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Clearing) {
  using TensorTupleType = typename TensorTupleTest_Clearing_Test::TensorTupleType;
  NNOTIFY("Testing setting from TensorsSpec ...");
  
  TensorsSpec spec = { {"elem0", {4,3}}, {"elem1", {4,5,6}} };
  TensorTupleType tuple("tuple", spec, 3, 2);
  NINFO("Before: " << tuple);
  tuple.clear();
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(3, tuple.timesteps());
  EXPECT_EQ(2, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{4,3,3,0}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,6,3,0}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{3,2}), tuple[k].capacities());
    EXPECT_TRUE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Filling) {
  using TensorTupleType = typename TensorTupleTest_Filling_Test::TensorTupleType;
  NNOTIFY("Testing setting from TensorsSpec ...");
  
  TensorsSpec spec = { {"elem0", {4,3}}, {"elem1", {4,5,6}} };
  TensorTupleType tuple("tuple", spec, 3, 2);
  NINFO("Before: " << tuple);
  tuple.clear();
  tuple.fill();
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(3, tuple.timesteps());
  EXPECT_EQ(2, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{4,3,3,2}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,6,3,2}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{3,2}), tuple[k].capacities());
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Resizing) {
  using TensorTupleType = typename TensorTupleTest_Resizing_Test::TensorTupleType;
  NNOTIFY("Testing resizing ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  NINFO("Before: " << tuple);
  
  tuple.resize({{8,2},{4,5,2}}, 1, 1);
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{8,2},{4,5,2}}), tuple.datumDimensions());
  EXPECT_EQ(1, tuple.timesteps());
  EXPECT_EQ(1, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{8,2,1,1}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,2,1,1}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{1,1}), tuple[k].capacities());
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
  
  tuple.resize(5, 6);
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{8,2},{4,5,2}}), tuple.datumDimensions());
  EXPECT_EQ(5, tuple.timesteps());
  EXPECT_EQ(6, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{8,2,5,6}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,2,5,6}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{5,6}), tuple[k].capacities());
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
  
  tuple.resize(3);
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{8,2},{4,5,2}}), tuple.datumDimensions());
  EXPECT_EQ(5, tuple.timesteps());
  EXPECT_EQ(3, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{8,2,5,3}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,2,5,3}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{5,3}), tuple[k].capacities());
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Reserving) {
  using TensorTupleType = typename TensorTupleTest_Reserving_Test::TensorTupleType;
  NNOTIFY("Testing reserving ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  NINFO("Before: " << tuple);
  
  tuple.reserve(10, 5);
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(10, tuple.timesteps());
  EXPECT_EQ(5, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{4,3,10,2}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,6,10,2}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{10,5}), tuple[k].capacities());
    EXPECT_EQ(3, tuple[k].timesteps()[0]);
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Shrinking) {
  using TensorTupleType = typename TensorTupleTest_Shrinking_Test::TensorTupleType;
  NNOTIFY("Testing shrinking ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  tuple.reserve(6);
  NINFO("Before: " << tuple);
  
  tuple.shrink();
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(3, tuple.timesteps());
  EXPECT_EQ(2, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{4,3,3,2}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,6,3,2}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{3,2}), tuple[k].capacities());
    EXPECT_EQ(3, tuple[k].timesteps()[0]);
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Reshaping) {
  using TensorTupleType = typename TensorTupleTest_Reshaping_Test::TensorTupleType;
  NNOTIFY("Testing reshaping ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  NINFO("Before: " << tuple);
  
  tuple.reshape(6, 1);
  NINFO("After: " << tuple);
  EXPECT_EQ(2, tuple.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), tuple.names());
  EXPECT_EQ("tuple", tuple.scope());
  EXPECT_EQ((std::vector<std::vector<size_t>>{{4,3},{4,5,6}}), tuple.datumDimensions());
  EXPECT_EQ(6, tuple.timesteps());
  EXPECT_EQ(1, tuple.batches());
  EXPECT_FALSE(tuple.empty());
  EXPECT_TRUE(tuple.isBatched());
  EXPECT_EQ((std::vector<size_t>{4,3,6,1}), tuple[0].dimensions());
  EXPECT_EQ((std::vector<size_t>{4,5,6,6,1}), tuple[1].dimensions());
  for (size_t k = 0; k < tuple.size(); ++k) {
    EXPECT_EQ((std::vector<size_t>{6,1}), tuple[k].capacities());
    EXPECT_EQ(6, tuple[k].timesteps()[0]);
    EXPECT_FALSE(tuple[k].empty());
    EXPECT_TRUE(tuple[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Cloning) {
  using TensorTupleType = typename TensorTupleTest_Cloning_Test::TensorTupleType;
  NNOTIFY("Testing cloning ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  TensorTupleType other("other");
  NINFO("Before: " << other);
  
  other.clone(tuple);
  NINFO("After: " << other);
  EXPECT_FALSE(other.hasSameStorageWith(tuple));
  EXPECT_EQ(tuple.size(), other.size());
  EXPECT_EQ(tuple.names(), other.names());
  EXPECT_EQ("other", other.scope());
  EXPECT_EQ(tuple.datumDimensions(), other.datumDimensions());
  EXPECT_EQ(other.timesteps(), other.timesteps());
  EXPECT_EQ(other.batches(), other.batches());
  EXPECT_FALSE(other.empty());
  EXPECT_TRUE(other.isBatched());
  EXPECT_EQ(other[0].dimensions(), tuple[0].dimensions());
  EXPECT_EQ(other[1].timesteps(), tuple[1].timesteps());
  for (size_t k = 0; k < other.size(); ++k) {
    EXPECT_EQ(other[k].capacities(), tuple[k].capacities());
    EXPECT_EQ(other[k].timesteps()[0], tuple[k].timesteps()[0]);
    EXPECT_FALSE(other[k].empty());
    EXPECT_TRUE(other[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Copying) {
  using TensorTupleType = typename TensorTupleTest_Copying_Test::TensorTupleType;
  NNOTIFY("Testing copying ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  TensorTupleType other("other");
  NINFO("Before: " << other);
  
  other.copy(tuple);
  NINFO("After: " << other);
  EXPECT_FALSE(other.hasSameStorageWith(tuple));
  EXPECT_EQ(tuple.size(), other.size());
  EXPECT_EQ((std::vector<std::string>{"", ""}), other.names());
  EXPECT_EQ("other", other.scope());
  EXPECT_EQ(tuple.datumDimensions(), other.datumDimensions());
  EXPECT_EQ(other.timesteps(), other.timesteps());
  EXPECT_EQ(other.batches(), other.batches());
  EXPECT_FALSE(other.empty());
  EXPECT_TRUE(other.isBatched());
  EXPECT_EQ(other[0].dimensions(), tuple[0].dimensions());
  EXPECT_EQ(other[1].timesteps(), tuple[1].timesteps());
  for (size_t k = 0; k < other.size(); ++k) {
    EXPECT_EQ(other[k].capacities(), tuple[k].capacities());
    EXPECT_EQ(other[k].timesteps()[0], tuple[k].timesteps()[0]);
    EXPECT_FALSE(other[k].empty());
    EXPECT_TRUE(other[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Mimicking) {
  using TensorTupleType = typename TensorTupleTest_Mimicking_Test::TensorTupleType;
  NNOTIFY("Testing mimicking ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  TensorTupleType other("other");
  NINFO("Before: " << other);
  
  other.mimic(tuple);
  NINFO("After: " << other);
  EXPECT_FALSE(other.hasSameStorageWith(tuple));
  EXPECT_EQ(tuple.size(), other.size());
  EXPECT_EQ((std::vector<std::string>{"tuple/elem0", "tuple/elem1"}), other.names());
  EXPECT_EQ("tuple", other.scope());
  EXPECT_EQ(tuple.datumDimensions(), other.datumDimensions());
  EXPECT_EQ(other.timesteps(), other.timesteps());
  EXPECT_EQ(other.batches(), other.batches());
  EXPECT_FALSE(other.empty());
  EXPECT_TRUE(other.isBatched());
  EXPECT_EQ(other[0].dimensions(), tuple[0].dimensions());
  EXPECT_EQ(other[1].timesteps(), tuple[1].timesteps());
  for (size_t k = 0; k < other.size(); ++k) {
    EXPECT_EQ(other[k].capacities(), tuple[k].capacities());
    EXPECT_EQ(other[k].timesteps()[0], tuple[k].timesteps()[0]);
    EXPECT_FALSE(other[k].empty());
    EXPECT_TRUE(other[k].isBatched());
  }
}

TYPED_TEST(TensorTupleTest, Assignment) {
  using TensorTupleType = typename TensorTupleTest_Assignment_Test::TensorTupleType;
  NNOTIFY("Testing assignment ...");
  
  TensorTupleType tuple("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  tuple.setConstant(1.0);
  
  TensorTupleType other("tuple", {"elem0", "elem1"}, {{4,3},{4,5,6}}, 3, 2);
  other.setConstant(2.0);
  NINFO("Before: " << other);
  
  other = tuple;
  NINFO("After: " << other);
  EXPECT_TRUE(other.hasSameStorageWith(tuple));
  for (size_t k = 0; k < other.size(); ++k) {
    for (size_t i = 0; i < other[k].size(); ++i) {
      EXPECT_EQ(1.0, other[k][i]);
    }
  }
}

} // namespace tests
} // namespace noesis

/* EOF */
