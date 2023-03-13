/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
//#include <memory>

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/utils/string.hpp>

namespace noesis {
namespace tests {

template <typename T, typename F>
struct Dummy {};

/*
 * Tests
 */

TEST(StringUtilsTest, MakeNameScope) {
  auto namescope = utils::make_namescope({"dog", "cat", "mouse"});
  std::cout << "Scope: " << namescope << std::endl;
  EXPECT_EQ("dog/cat/mouse", namescope);
}

TEST(StringUtilsTest, RemoveNameScope) {
  auto name = utils::remove_namescope({"dog/cat/mouse"});
  std::cout << "Name: " << name << std::endl;
  EXPECT_EQ("mouse", name);
}

TEST(StringUtilsTest, VecToString) {
  std::vector<int> ints = {1,2,3,4};
  std::vector<float> floats = {0.0, 10.0};
  std::vector<std::string> strings = {"dog", "cat", "mouse"};
  std::vector<Eigen::VectorXd> vectors = {
    Eigen::VectorXd::Constant(2, 1.0),
    Eigen::VectorXd::Constant(3, 2.0),
    Eigen::VectorXd::Constant(2, 3.0)
  };
  auto intsStr = utils::vector_to_string(ints, "[]");
  auto floatsStr = utils::vector_to_string(floats, "(]");
  auto stringsStr = utils::vector_to_string(strings);
  auto vectorStr = utils::vector_to_string(vectors);
  std::cout << "Integers: " << stringsStr << std::endl;
  std::cout << "Floats: " << floatsStr << std::endl;
  std::cout << "Strings: " << intsStr << std::endl;
  std::cout << "Vectors: " << vectorStr << std::endl;
  EXPECT_EQ("[1, 2, 3, 4]", intsStr);
  EXPECT_EQ("(0.000000, 10.000000]", floatsStr);
  EXPECT_EQ("{dog, cat, mouse}", stringsStr);
}

TEST(StringUtilsTest, TypenameToString) {
  using Type = Dummy<int,float>;
  auto raw = utils::type_name<Type>().raw;
  auto scoped = utils::type_name<Type>().scoped;
  auto name = utils::type_name<Type>().name;
  std::cout << "raw: " << raw << std::endl;
  std::cout << "scoped_name: " << scoped << std::endl;
  std::cout << "name: " << name << std::endl;
  EXPECT_EQ("noesis::tests::Dummy<int, float>", raw);
  EXPECT_EQ("noesis::tests::Dummy", scoped);
  EXPECT_EQ("Dummy", name);
}

} // namespace tests
} // namespace noesis

/* EOF */
