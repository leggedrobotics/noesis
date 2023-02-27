/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_CORE_TENSORS_SPEC_HPP_
#define NOESIS_FRAMEWORK_CORE_TENSORS_SPEC_HPP_

// C/C++
#include <iostream>
#include <vector>

// Noesis
#include "noesis/framework/utils/string.hpp"

namespace noesis {

/*!
 * @brief The input/output specification provides name-to-dimensions mappings for parameterized functions.
 */
using TensorsSpec = std::vector<std::pair<std::string, std::vector<size_t>>>;

static inline std::vector<std::string> names_from_spec(const TensorsSpec& spec) {
  std::vector<std::string> names;
  for (auto& elem: spec) {
    names.push_back(elem.first);
  }
  return names;
}

static inline std::vector<std::vector<size_t>> dimensions_from_spec(const TensorsSpec& spec) {
  std::vector<std::vector<size_t>> dims;
  for (auto& elem: spec) {
    dims.push_back(elem.second);
  }
  return dims;
}

static inline std::vector<size_t> batched_dimensions_from_spec(const std::pair<std::string, std::vector<size_t>>& spec) {
  auto newDims = spec.second;
  newDims.insert(newDims.end(), {1, 1});
  return newDims;
}

} // namespace noesis

/*!
 * @brief Custom stream operator for TensorsSpec objects.
 * @param os The target output stream to direct output.
 * @param spec the input-output specification to be written.
 * @return The augmented output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const noesis::TensorsSpec &spec) {
  std::string out = "{";
  for (auto& field: spec) {
    out += " '" + field.first + "': " + noesis::utils::vector_to_string(field.second, "[]") + ",";
  }
  if (!spec.empty()) {
    out.pop_back();
    out += " ";
  }
  out += "}";
  return os << out;
}

#endif // NOESIS_FRAMEWORK_CORE_TENSORS_SPEC_HPP_

/* EOF */
