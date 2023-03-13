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
#include <noesis/framework/hyperparam/hyper_parameters.hpp>

namespace noesis {
namespace tests {

/**
 * Define Test Fixture
 */

class HyperParameterManagerTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  HyperParameterManagerTest() = default;
  ~HyperParameterManagerTest() = default;
};

/*
 * Tests
 */

TEST_F(HyperParameterManagerTest, Basic) {
  hyperparam::HyperParameter<bool> myBool(true, "TestAgent/myBool");
  hyperparam::HyperParameter<int> myInt(42, "TestAgent/MyFunc/myInt", {0, 100});
  hyperparam::HyperParameter<std::string> myString("Value0", "TestAgent/MyFunc/myString", {"Value0", "Value1", "Value2"});
  hyperparam::manager->addParameter(myBool);
  hyperparam::manager->addParameter(myInt);
  hyperparam::manager->addParameter(myString);
  hyperparam::manager->printParameters();
  hyperparam::manager->setParameterValue<bool>("TestAgent/myBool", true);
  hyperparam::manager->setParameterValue<int>("TestAgent/MyFunc/myInt", 47);
  hyperparam::manager->setParameterValue<std::string>("TestAgent/MyFunc/myString", "Value2");
  hyperparam::manager->printParameters();
  NINFO("\n" << myBool.toXmlStr());
  NINFO("\n" << myInt.toXmlStr());
  NINFO("\n" << myString.toXmlStr());
  NINFO("\n" << myBool.toXmlStr(true));
  NINFO("\n" << myInt.toXmlStr(true));
  NINFO("\n" << myString.toXmlStr(true));
  NNOTIFY("Manager XML export: ");
  TiXmlPrinter printerFull;
  printerFull.SetIndent("  ");
  TiXmlElement parameter_description_full("ExampleParametersFull");
  hyperparam::manager->saveParametersToXmlElement(&parameter_description_full);
  parameter_description_full.Accept(&printerFull);
  NINFO("FULL:\n" << printerFull.Str());
  TiXmlPrinter printerSimple;
  printerSimple.SetIndent("  ");
  TiXmlElement parameter_description_simple("ExampleParametersSimple");
  hyperparam::manager->saveParametersToXmlElement(&parameter_description_simple, true);
  parameter_description_simple.Accept(&printerSimple);
  NINFO("SIMPLE:\n" << printerSimple.Str());
  hyperparam::manager->removeAllParameters();
}

TEST_F(HyperParameterManagerTest, AddingRemovingTuple) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::hyperparam::HyperParameterTuple tuple("test", "tuple", "custom");
  tuple.addParameter<int>(1, "sInt", {0, 100});
  tuple.addParameter<std::vector<int>>({2,3}, "vInt", {{1, 100}});
  NINFO(tuple);
  NINFO("Manager before adding tuple:");
  hyperparam::manager->printParameters();
  NINFO("Manager after adding tuple:");
  hyperparam::manager->addParameterTuple(tuple);
  hyperparam::manager->printParameters();
  NINFO("Manager after removing tuple:");
  hyperparam::manager->removeParameterTuple(tuple);
  hyperparam::manager->printParameters();
  hyperparam::manager->removeAllParameters();
}

TEST_F(HyperParameterManagerTest, AddingMultipleInstances) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  hyperparam::HyperParameter<int> myInt(42, "TestAgent/MyFunc/myInt", {0, 100});
  hyperparam::HyperParameter<int> myIntAlt(39, "TestAgent/MyFunc/myInt");
  NINFO(myInt);
  NINFO(myIntAlt);
  ASSERT_EQ(42, myInt.getValue<int>());
  ASSERT_EQ((std::vector<int>{0, 100}), myInt.range());
  ASSERT_EQ(39, myIntAlt.getValue<int>());
  ASSERT_EQ((std::vector<int>{}), myIntAlt.range());
  hyperparam::manager->addParameter(myInt);
  NINFO("Manager before adding extra instance of the same parameter:");
  hyperparam::manager->printParameters();
  hyperparam::manager->addParameter(myIntAlt);
  NINFO("Manager after adding extra instance of the same parameter:");
  hyperparam::manager->printParameters();
  ASSERT_EQ(42, myInt.getValue<int>());
  ASSERT_EQ((std::vector<int>{0, 100}), myInt.range());
  ASSERT_EQ(42, myIntAlt.getValue<int>());
  ASSERT_EQ((std::vector<int>{0, 100}), myIntAlt.range());
  ASSERT_EQ(myInt.getValuePtr<int>().get(), myIntAlt.getValuePtr<int>().get());
  hyperparam::manager->removeAllParameters();
}

TEST_F(HyperParameterManagerTest, RemovingMultipleInstances) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  hyperparam::HyperParameter<int> myInt(42, "TestAgent/MyFunc/myInt", {0, 100});
  hyperparam::HyperParameter<int> myIntAlt(39, "TestAgent/MyFunc/myInt");
  NINFO(myInt);
  NINFO(myIntAlt);
  hyperparam::manager->addParameter(myInt);
  hyperparam::manager->addParameter(myIntAlt);
  NINFO("Managed parameters before first removal:");
  hyperparam::manager->printParameters();
  ASSERT_EQ(myInt.getValuePtr<int>().get(), myIntAlt.getValuePtr<int>().get());
  EXPECT_TRUE(hyperparam::manager->exists(myInt));
  EXPECT_TRUE(hyperparam::manager->exists(myIntAlt));
  hyperparam::manager->removeParameter(myInt);
  NINFO("Managed parameters after first removal:");
  hyperparam::manager->printParameters();
  EXPECT_FALSE(hyperparam::manager->exists(myInt));
  EXPECT_FALSE(hyperparam::manager->exists(myIntAlt));
  ASSERT_EQ(myInt.getValuePtr<int>().get(), myIntAlt.getValuePtr<int>().get());
  hyperparam::manager->removeAllParameters();
}

TEST_F(HyperParameterManagerTest, RetrievingAccessorToManaged) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  hyperparam::HyperParameter<int> myInt(42, "TestAgent/MyFunc/myInt", {0, 100});
  NINFO(myInt);
  hyperparam::manager->addParameter(myInt);
  hyperparam::manager->printParameters();
  EXPECT_TRUE(hyperparam::manager->exists(myInt));
  auto myIntAlt = hyperparam::manager->getParameter<int>("TestAgent/MyFunc/myInt");
  NINFO(myIntAlt);
  EXPECT_TRUE(hyperparam::manager->exists(myIntAlt));
  ASSERT_EQ(myInt.getValuePtr<int>().get(), myIntAlt.getValuePtr<int>().get());
  hyperparam::manager->removeAllParameters();
}

} // namespace tests
} // namespace noesis

/* EOF */
