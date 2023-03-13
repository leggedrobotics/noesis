/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_STRING_HPP_
#define NOESIS_FRAMEWORK_UTILS_STRING_HPP_

// C/C++
#include <iostream>
#include <sstream>
#include <vector>

// Boost
#include <boost/type_index.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

// Eigen
#include <Eigen/Core>

namespace noesis {
namespace utils {

static inline std::string make_namescope(std::vector<std::string> names) {
//  boost::filesystem::path path = "/";
  boost::filesystem::path path;
  if (!names.empty()) { for (const auto& name: names) { path /= name; }}
  return path.remove_trailing_separator().string();
}

static inline std::string remove_namescope(std::string namescope) {
  boost::filesystem::path scoped_name = namescope;
  std::vector<std::string> scopes;
  for (auto const& subscope : scoped_name) { scopes.push_back(subscope.string()); }
  std::string name = scopes.back();
  return name;
}

static inline std::vector<std::string> to_scope_vector(std::string namescope) {
  boost::filesystem::path scoped_name = namescope;
  std::vector<std::string> scopes;
  for (auto const& scope : scoped_name) { scopes.push_back(scope.string()); }
  return scopes;
}

template <typename Type_>
static inline std::enable_if_t<std::is_same<Type_, std::string>::value, std::string>
vector_to_string(const std::vector<Type_>& vector, std::string brackets = "{}") {
  std::string out;
  out += brackets.front();
  for (auto& element: vector) { out += element + ", "; }
  if (!vector.empty()) {
    out.pop_back();
    out.pop_back();
  }
  out += brackets.back();
  return out;
}

template <typename Type_>
static inline std::enable_if_t<std::is_scalar<Type_>::value, std::string>
vector_to_string(const std::vector<Type_>& vector, std::string brackets = "{}") {
  std::string out;
  out += brackets.front();
  for (auto& element: vector) { out += std::to_string(element) + ", "; }
  if (!vector.empty()) {
    out.pop_back();
    out.pop_back();
  }
  out += brackets.back();
  return out;
}

template<typename Scalar_, int Rows_, int Cols_>
static inline std::enable_if_t<std::is_class<Eigen::Matrix<Scalar_, Rows_, Cols_>>::value, std::string>
vector_to_string(const std::vector<Eigen::Matrix<Scalar_, Rows_, Cols_>>& vectors) {
  std::stringstream ss;
  for (int i = 0; i < vectors.size(); ++i) { ss << "\n[" << i << "]:\n" << vectors[i]; }
  return ss.str();
}

template<typename Scalar_, int Rows_, int Cols_, class Allocator_>
static inline std::enable_if_t<std::is_class<Eigen::Matrix<Scalar_, Rows_, Cols_>>::value, std::string>
vector_to_string(const std::vector<Eigen::Matrix<Scalar_, Rows_, Cols_>, Allocator_>& vectors) {
  std::stringstream ss;
  for (size_t i = 0; i < vectors.size(); ++i) { ss << "\n[" << i << "]:\n" << vectors[i]; }
  return ss.str();
}

template <typename T>
struct type_name {
  type_name() {
    raw = boost::typeindex::type_id_with_cvr<T>().pretty_name();
    scoped = raw;
    if (scoped.back() == '>') {
      auto template_start_pos = scoped.find_first_of('<');
      scoped.erase(template_start_pos);
    }
    size_t position = scoped.rfind("::");
    if (position == std::string::npos) {
      position = 0;
    } else {
      position += 2;
    }
    name = scoped.substr(position);
  }
  std::string raw;
  std::string scoped;
  std::string name;
};

template <typename T>
static inline std::string typename_to_string(const T& t) {
  std::string name = boost::typeindex::type_id_runtime(t).pretty_name();
  return name;
}

template <typename Scalar>
typename std::enable_if<std::is_scalar<Scalar>::value>::type
append_back(const std::vector<Scalar>& input, std::vector<Scalar>& output) {
  output.insert(output.end(), input.begin(), input.end());
}

} // utils
} //noesis

#endif // NOESIS_FRAMEWORK_UTILS_STRING_HPP_

/* EOF */
