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
#include <noesis/framework/system/process.hpp>
#include <noesis/framework/hyperparam/HyperParameterTuple.hpp>

namespace noesis {
namespace tests {

/**
 * Define Test Fixture
 */

class HyperParameterTupleTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  HyperParameterTupleTest() = default;
  ~HyperParameterTupleTest() = default;
};

/*
 * Tests
 */

TEST_F(HyperParameterTupleTest, Empty) {
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  NINFO(tuple);
  ASSERT_EQ("test", tuple.getScope());
  ASSERT_EQ("tuple", tuple.getName());
  ASSERT_EQ("custom", tuple.getCategory());
  ASSERT_EQ("test/custom/tuple", tuple.namescope());
  NINFO("XML output:\n" << tuple.toXmlStr());
  NINFO("Simplified XML output:\n" << tuple.toXmlStr(true));
}

TEST_F(HyperParameterTupleTest, Adding) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  tuple.addParameter<bool>(true, "sBool");
  tuple.addParameter<int>(1, "sInt", {0, 100});
  tuple.addParameter<float>(2.0f, "sFloat", {0.0f, 100.0f});
  tuple.addParameter<double>(3.0, "sDouble", {0.0, 100.0});
  tuple.addParameter<std::string>("val0", "sString", {"val0", "val1", "val2"});
  tuple.addParameter<std::vector<bool>>({true, true}, "vBool");
  tuple.addParameter<std::vector<int>>({2,3}, "vInt", {{1, 100}});
  tuple.addParameter<std::vector<float>>({2.0f,3.0f}, "vFloat", {{1, 100}});
  tuple.addParameter<std::vector<double>>({2.0, 3.0}, "vDouble", {{1, 100}});
  tuple.addParameter<std::vector<std::string>>({"val0", "val1"}, "vString", {{"val0", "val1", "val2"}});
  NINFO(tuple);
  ASSERT_EQ(10, tuple.size());
  ASSERT_DEATH(tuple.addParameter<int>(2, "sInt");, "");
  NINFO("XML output:\n" << tuple.toXmlStr());
  NINFO("Simplified XML output:\n" << tuple.toXmlStr(true));
}

TEST_F(HyperParameterTupleTest, Getting) {
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  tuple.addParameter<bool>(true, "sBool");
  tuple.addParameter<int>(1, "sInt", {0, 100});
  tuple.addParameter<float>(2.0f, "sFloat", {0.0f, 100.0f});
  tuple.addParameter<double>(3.0, "sDouble", {0.0, 100.0});
  tuple.addParameter<std::string>("val0", "sString", {"val0", "val1", "val2"});
  tuple.addParameter<std::vector<bool>>({true, true}, "vBool");
  tuple.addParameter<std::vector<int>>({2,3}, "vInt", {{1, 100}});
  tuple.addParameter<std::vector<float>>({2.0f,3.0f}, "vFloat", {{1, 100}});
  tuple.addParameter<std::vector<double>>({2.0,3.0}, "vDouble", {{1, 100}});
  tuple.addParameter<std::vector<std::string>>({"val0", "val1"}, "vString", {{"val0", "val1", "val2"}});
  NINFO(tuple);
  ASSERT_EQ(true, tuple.getParameterValue<bool>("sBool"));
  ASSERT_EQ(1, tuple.getParameterValue<int>("sInt"));
  ASSERT_EQ(2.0f, tuple.getParameterValue<float>("sFloat"));
  ASSERT_EQ(3.0, tuple.getParameterValue<double>("sDouble"));
  ASSERT_EQ("val0", tuple.getParameterValue<std::string>("sString"));
  ASSERT_EQ(std::vector<bool>({true, true}), tuple.getParameterValue<std::vector<bool>>("vBool"));
  ASSERT_EQ(std::vector<int>({2,3}), tuple.getParameterValue<std::vector<int>>("vInt"));
  ASSERT_EQ(std::vector<float>({2.0f,3.0f}), tuple.getParameterValue<std::vector<float>>("vFloat"));
  ASSERT_EQ(std::vector<double>({2.0,3.0}), tuple.getParameterValue<std::vector<double>>("vDouble"));
  ASSERT_EQ(std::vector<std::string>({"val0", "val1"}), tuple.getParameterValue<std::vector<std::string>>("vString"));
  NINFO("XML output:\n" << tuple.toXmlStr());
  NINFO("Simplified XML output:\n" << tuple.toXmlStr(true));
}

TEST_F(HyperParameterTupleTest, Setting) {
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  tuple.addParameter<bool>(true, "sBool");
  tuple.addParameter<int>(1, "sInt", {0, 100});
  tuple.addParameter<float>(2.0f, "sFloat", {0.0f, 100.0f});
  tuple.addParameter<double>(3.0, "sDouble", {0.0, 100.0});
  tuple.addParameter<std::string>("val0", "sString", {"val0", "val1", "val2"});
  tuple.addParameter<std::vector<bool>>({true, true}, "vBool");
  tuple.addParameter<std::vector<int>>({2,3}, "vInt", {{1, 100}});
  tuple.addParameter<std::vector<float>>({2.0f,3.0f}, "vFloat", {{1.0f, 100.0f}});
  tuple.addParameter<std::vector<double>>({2.0,3.0}, "vDouble", {{1.0, 100.0}});
  tuple.addParameter<std::vector<std::string>>({"val0", "val1"}, "vString", {{"val0", "val1", "val2"}});
  NINFO(tuple);
  tuple.setParameterValue<bool>("sBool", false);
  tuple.setParameterValue<int>("sInt", 42);
  tuple.setParameterValue<float>("sFloat", 42.0f);
  tuple.setParameterValue<double>("sDouble", 42.0);
  tuple.setParameterValue<std::string>("sString", "val2");
  tuple.setParameterValue<std::vector<bool>>("vBool", {false, false});
  tuple.setParameterValue<std::vector<int>>("vInt", {42, 37});
  tuple.setParameterValue<std::vector<float>>("vFloat", {42.0f, 37.0f});
  tuple.setParameterValue<std::vector<double>>("vDouble", {42.0, 37.0});
  tuple.setParameterValue<std::vector<std::string>>("vString", {"val2", "val1"});
  ASSERT_EQ(false, tuple.getParameterValue<bool>("sBool"));
  ASSERT_EQ(42, tuple.getParameterValue<int>("sInt"));
  ASSERT_EQ(42.0f, tuple.getParameterValue<float>("sFloat"));
  ASSERT_EQ(42.0, tuple.getParameterValue<double>("sDouble"));
  ASSERT_EQ("val2", tuple.getParameterValue<std::string>("sString"));
  ASSERT_EQ(std::vector<bool>({false, false}), tuple.getParameterValue<std::vector<bool>>("vBool"));
  ASSERT_EQ(std::vector<int>({42, 37}), tuple.getParameterValue<std::vector<int>>("vInt"));
  ASSERT_EQ(std::vector<float>({42.0f, 37.0f}), tuple.getParameterValue<std::vector<float>>("vFloat"));
  ASSERT_EQ(std::vector<double>({42.0, 37.0}), tuple.getParameterValue<std::vector<double>>("vDouble"));
  ASSERT_EQ(std::vector<std::string>({"val2", "val1"}), tuple.getParameterValue<std::vector<std::string>>("vString"));
  NINFO("XML output:\n" << tuple.toXmlStr());
  NINFO("Simplified XML output:\n" << tuple.toXmlStr(true));
}

TEST_F(HyperParameterTupleTest, SetFromXml) {
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  tuple.addParameter<bool>(false, "sBool");
  tuple.addParameter<int>(42, "sInt", {0, 100});
  tuple.addParameter<float>(42.0f, "sFloat", {0.0f, 100.0f});
  tuple.addParameter<double>(42.0, "sDouble", {0.0, 100.0});
  tuple.addParameter<std::string>("val2", "sString", {"val0", "val1", "val2"});
  tuple.addParameter<std::vector<bool>>({false, true}, "vBool");
  tuple.addParameter<std::vector<int>>({6,7}, "vInt", {{1, 100}});
  tuple.addParameter<std::vector<float>>({6.0f,7.0f}, "vFloat", {{1.0f, 100.0f}});
  tuple.addParameter<std::vector<double>>({6.0,7.0}, "vDouble", {{1.0, 100.0}});
  tuple.addParameter<std::vector<std::string>>({"val1", "val0"}, "vString", {{"val0", "val1", "val2"}});
  NINFO(tuple);
  TiXmlDocument document(noesis::rootpath() + "/noesis/test/src/hyperparam/parameters.xml");
  document.LoadFile(TiXmlEncoding::TIXML_ENCODING_UTF8);
  TiXmlHandle handle(&document);
  tuple.fromXml(*handle.FirstChildElement().ToElement());
  ASSERT_EQ(true, tuple.getParameterValue<bool>("sBool"));
  ASSERT_EQ(1, tuple.getParameterValue<int>("sInt"));
  ASSERT_EQ(2.0f, tuple.getParameterValue<float>("sFloat"));
  ASSERT_EQ(3.0, tuple.getParameterValue<double>("sDouble"));
  ASSERT_EQ("val0", tuple.getParameterValue<std::string>("sString"));
  ASSERT_EQ(std::vector<bool>({true, true}), tuple.getParameterValue<std::vector<bool>>("vBool"));
  ASSERT_EQ(std::vector<int>({2,3}), tuple.getParameterValue<std::vector<int>>("vInt"));
  ASSERT_EQ(std::vector<float>({2.0f,3.0f}), tuple.getParameterValue<std::vector<float>>("vFloat"));
  ASSERT_EQ(std::vector<double>({2.0,3.0}), tuple.getParameterValue<std::vector<double>>("vDouble"));
  ASSERT_EQ(std::vector<std::string>({"val0", "val1"}), tuple.getParameterValue<std::vector<std::string>>("vString"));
  NINFO(tuple);
  NINFO("XML output:\n" << tuple.toXmlStr());
  NINFO("Simplified XML output:\n" << tuple.toXmlStr(true));
}

TEST_F(HyperParameterTupleTest, CreateFromXml) {
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  TiXmlDocument document(noesis::rootpath() + "/noesis/test/src/hyperparam/parameters.xml");
  document.LoadFile(TiXmlEncoding::TIXML_ENCODING_UTF8);
  TiXmlHandle handle(&document);
  tuple.createParametersFromXml(*handle.FirstChildElement().ToElement());
  NINFO(tuple);
  ASSERT_EQ(true, tuple.getParameterValue<bool>("sBool"));
  ASSERT_EQ(1, tuple.getParameterValue<int>("sInt"));
  ASSERT_EQ(2.0f, tuple.getParameterValue<float>("sFloat"));
  ASSERT_EQ(3.0, tuple.getParameterValue<double>("sDouble"));
  ASSERT_EQ("val0", tuple.getParameterValue<std::string>("sString"));
  ASSERT_EQ(std::vector<bool>({true, true}), tuple.getParameterValue<std::vector<bool>>("vBool"));
  ASSERT_EQ(std::vector<int>({2,3}), tuple.getParameterValue<std::vector<int>>("vInt"));
  ASSERT_EQ(std::vector<float>({2.0f,3.0f}), tuple.getParameterValue<std::vector<float>>("vFloat"));
  ASSERT_EQ(std::vector<double>({2.0,3.0}), tuple.getParameterValue<std::vector<double>>("vDouble"));
  ASSERT_EQ(std::vector<std::string>({"val0", "val1"}), tuple.getParameterValue<std::vector<std::string>>("vString"));
  NINFO("XML output:\n" << tuple.toXmlStr());
  NINFO("Simplified XML output:\n" << tuple.toXmlStr(true));
}

} // namespace tests
} // namespace noesis

/* EOF */
