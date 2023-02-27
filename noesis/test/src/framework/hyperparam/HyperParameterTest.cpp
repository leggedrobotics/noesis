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
#include <noesis/framework/hyperparam/HyperParameter.hpp>

namespace noesis {
namespace tests {

/**
 * Define Test Fixture
 */

class HyperParameterTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  HyperParameterTest() = default;
  ~HyperParameterTest() = default;
};

template <typename ValueType_>
std::enable_if_t<std::is_same<ValueType_, std::vector<bool>>::value, std::string>
print_vector(const ValueType_& vec) {
  std::string out = "[";
  for (auto element: vec) {
    out += std::to_string(element) + ", ";
  }
  if (!vec.empty()) {
    out.pop_back();
    out.pop_back();
  }
  out += "]";
  return out;
}

template <typename ValueType_>
std::enable_if_t<noesis::hyperparam::internal::is_supported_numeric_vector<ValueType_>::value, std::string>
print_vector(const ValueType_& vec) {
  std::string out = "[";
  for (auto element: vec) {
    out += std::to_string(element) + ", ";
  }
  if (!vec.empty()) {
    out.pop_back();
    out.pop_back();
  }
  out += "]";
  return out;
}

template <typename ValueType_>
std::enable_if_t<std::is_same<ValueType_, std::vector<std::string>>::value, std::string>
print_vector(const ValueType_& vec) {
  std::string out = "[";
  for (auto element: vec) {
    out += element + ", ";
  }
  if (!vec.empty()) {
    out.pop_back();
    out.pop_back();
  }
  out += "]";
  return out;
}

/*
 * Tests
 */

TEST_F(HyperParameterTest, Bool) {
  noesis::hyperparam::HyperParameter<bool> param(true, "test/param");
  NINFO(param);
  ASSERT_EQ(true, param);
  param = false;
  ASSERT_EQ(false, param);
  bool value = param;
  NINFO("Value: " << value);
  ASSERT_EQ(false, value);
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, Integer) {
  noesis::hyperparam::HyperParameter<int> param(42, "test/param", {0, 1000});
  NINFO(param);
  ASSERT_EQ(42, param);
  param = 37;
  ASSERT_EQ(37, param);
  int value = param;
  NINFO("Value: " << value);
  ASSERT_EQ(37, value);
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, Float) {
  noesis::hyperparam::HyperParameter<float> param(42.0, "test/param", {0.0, 1000.0});
  NINFO(param);
  ASSERT_EQ(42.0, param);
  param = 37.0;
  ASSERT_EQ(37.0, param);
  float value = param;
  NINFO("Value: " << value);
  ASSERT_EQ(37.0, value);
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, Double) {
  noesis::hyperparam::HyperParameter<double> param(42.0, "test/param", {0.0, 1000.0});
  NINFO(param);
  ASSERT_EQ(42.0, param);
  param = 37.0;
  ASSERT_EQ(37.0, param);
  double value = param;
  NINFO("Value: " << value);
  ASSERT_EQ(37.0, value);
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, String) {
  noesis::hyperparam::HyperParameter<std::string> param("Value0", "test/param", {"Value0", "Value1", "Value2"});
  NINFO(param);
  std::string value = param;
  ASSERT_EQ(std::string("Value0"), value);
  param = "Value1";
  value = param;
  ASSERT_EQ(std::string("Value1"), value);
  NINFO("Value: " << value);
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, BoolVector) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<std::vector<bool>> param({false, false, false}, "test/param");
  NINFO(param);
  param = {false, true, false};
  NINFO(param);
  std::vector<bool> values = param;
  NINFO("Value: " << print_vector(values));
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, IntegerVector) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<std::vector<int>> param({1,2,3}, "test/param");
  NINFO(param);
  param = {42, 37, 57};
  NINFO(param);
  std::vector<int> values = param;
  NINFO("Value: " << print_vector(values));
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, FloatVector) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<std::vector<float>> param({1.0,2.0,3.0}, "test/param");
  NINFO(param);
  param = {42.0, 37.0, 57.0};
  NINFO(param);
  std::vector<float> values = param;
  NINFO("Value: " << print_vector(values));
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, DoubleVector) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<std::vector<double>> param({1.0,2.0,3.0}, "test/param");
  NINFO(param);
  param = {42.0, 37.0, 57.0};
  NINFO(param);
  std::vector<double> values = param;
  NINFO("Value: " << print_vector(values));
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, StringVector) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<std::vector<std::string>> param({"Value0", "Value1", "Value2"}, "test/param");
  NINFO(param);
  param = {"Value2", "Value5", "Value8"};
  NINFO(param);
  std::vector<std::string> values = param;
  NINFO("Value: " << print_vector(values));
  NINFO("XML output:\n" << param.toXmlStr());
  NINFO("Simplified XML output:\n" << param.toXmlStr(true));
}

TEST_F(HyperParameterTest, MinMaxRange) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<double> param(42.0, "test/param", {0.0, 1000.0});
  NINFO(param);
  ASSERT_DEATH({param = -1.0;}, "");
  ASSERT_DEATH({param = 1024.0;}, "");
}

TEST_F(HyperParameterTest, DiscreteRange) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameter<double> param(0.0, "test/param", {0.0, 100.0, 1000.0});
  NINFO(param);
  param = 100.0;
  ASSERT_EQ(100.0, param);
  ASSERT_DEATH({param = -1.0;}, "");
  ASSERT_DEATH({param = 1024.0;}, "");
}

} // namespace tests
} // namespace noesis

/* EOF */
