/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_LOG_METRIC_HPP_
#define NOESIS_FRAMEWORK_LOG_METRIC_HPP_

// C/C++
#include <string>
#include <vector>

// Eigen
#include <Eigen/Core>

// Noesis
#include "noesis/framework/log/tensorboard.hpp"

namespace noesis {
namespace log {

/*!
 * @brief Defines a container for holding collections of scalar metrics as name-value pairs.
 * @note Data is stored using std and Eigen vectors instead of vector of std::pair for performance reasons.
 * @tparam ScalarType_ The scalar type used for the metric values of the metric collection.
 */
template<typename ScalarType_>
class Metrics
{
public:
  
  /*
   * Types
   */
  
  //! @brief Each value is of the specified scalar type.
  using Scalar = ScalarType_;
  
  //! @brief Names is simply a vector of std::string.
  using Names = std::vector<std::string>;
  
  //! @brief Values are dynamically-sized Eigen (column) vectors.
  using Values = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  
  /*
   * Instantiation
   */
  
  //! @note Default construction is enabled for this class.
  Metrics() = default;
  
  //! @note Move and move-assignment construction is permissible for this class.
  Metrics(Metrics&& other) noexcept = default;
  Metrics& operator=(Metrics&& other) noexcept = default;
  
  ///! @note Copy and copy-assignment construction is permissible for this class.
  Metrics(const Metrics& other) = default;
  Metrics& operator=(const Metrics& other) = default;
  
  //! @brief Constructor using a vector of strings to configure metrics.
  //! @note Default construction is enabled for this class.
  explicit Metrics(const Names& names) {
    for (const auto& name: names) {
      this->push_back(name);
    }
  }
  
  //! @note Destructor is defaulted. Nothing fancy needed here.
  ~Metrics() = default;
  
  /*
   * Properties
   */
  
  //! @brief Convenience function for retrieving the size of the metric collection.
  size_t size() const { return names_.size(); }
  
  //! @brief Convenience function for creating appropriately size vector of zeros.
  const Names& names() const { return names_; }
  
  //! @brief Convenience function for creating appropriately size vector of zeros.
  const Values& values() const { return values_; }
  
  //! @brief Convenience function for creating appropriately size vector of zeros.
  Values& values() { return values_; }
  
  //! @brief Convenience function for creating appropriately size vector of zeros.
  Values zeros() const { return Values::Zero(values_.size()); }
  
  /*
   * Operations
   */
  
  //! @brief Convenience function for clearing the instance of all name-value pairs.
  void clear() {
    keys_.clear();
    names_.clear();
    values_.resize(0);
  }
  
  //! @brief Convenience function for initializing the held values with zeros.
  void reset() {
    values_.setZero();
  }
  
  //! @brief Convenience function for appending a new name-value pair to the metric collection.
  void push_back(std::string name, Scalar value=0) {
    NFATAL_IF(keys_.count(name) > 0, "Failed to add '" << name << "' to metrics: Name already exists.");
    values_.conservativeResize(values_.size()+1);
    values_(values_.size()-1) = value;
    names_.emplace_back(name);
    keys_.insert({name, values_.size()-1});
  }
  
  //! @brief Convenience read-only access operator for direct access to individual value elements.
  Scalar operator[](size_t index) const { return values_(index); }
  
  //! @brief Convenience read-write access operator for direct access to individual value elements.
  Scalar& operator[](size_t index) { return values_(index); }
  
  //! @brief Convenience read-only access operator for direct access to individual value elements.
  Scalar operator[](const std::string& name) const { return values_(keys_.at(name)); }
  
  //! @brief Convenience read-write access operator for direct access to individual value elements.
  Scalar& operator[](const std::string& name) { return values_(keys_.at(name)); }
  
  //! @brief Convenience function to get printable string with the state of the metric.
  std::string info() const {
    std::stringstream out;
    for (size_t m = 0; m < names_.size(); ++m) { out << "\n  " << names_[m] << ": " << values_(m); }
    return out.str();
  }
  
  //! @brief Convenience function to add (register) all held metrics with a TensorBoard logger.
  void add_to(TensorBoardLogger* logger) {
    for (size_t m = 0; m < size(); ++m) { logger->addLoggingSignal(names_[m], 1); }
  }
  
  //! @brief Convenience function to append (assign) all held metrics to TensorBoard logger.
  void append_to(TensorBoardLogger* logger) {
    for (size_t m = 0; m < size(); ++m) { logger->appendScalar(names_[m], values_(m)); }
  }

private:
  //! @brief The vector of metric names.
  Names names_;
  //! @brief The vector of metric values.
  Values values_;
  //! @brief Internal helper which maps strings to integer array indices.
  std::unordered_map<std::string, int> keys_;
};

} // namespace log
} // namespace noesis

#endif // NOESIS_FRAMEWORK_LOG_METRIC_HPP_

/* EOF */
