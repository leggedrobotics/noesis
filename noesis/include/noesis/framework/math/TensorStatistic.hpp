/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_MATH_TENSOR_STATISTIC_HPP_
#define NOESIS_FRAMEWORK_MATH_TENSOR_STATISTIC_HPP_

// C++
#include <boost/algorithm/string/replace.hpp>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/Tensor.hpp"
#include "noesis/framework/math/statistics.hpp"

namespace noesis {

/*!
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 */
template<typename ScalarType_>
class TensorStatistic final: core::Object
{
public:
  // Aliases
  using ScalarType = ScalarType_;
  using TensorType = Tensor<ScalarType>;
  
  /*
   * Instantiation
   */
  
  explicit TensorStatistic(const std::string& name, const std::string& scope, bool verbose):
    core::Object(name, scope, verbose),
    mean_(utils::make_namescope({scope, name, "mean"})),
    variance_(utils::make_namescope({scope, name, "variance"}))
  {
  }
  
  ~TensorStatistic() final = default;
  
  /*
   * Configurations
   */
  
  void configure(const std::vector<size_t>& dimensions) {
    mean_.resize(dimensions, false);
    mean_.setConstant(0);
    variance_.resize(dimensions, false);
    variance_.setConstant(1);
  }
  
  /*
   * Properties
   */
  
  const TensorType& mean() const {
    return mean_;
  }
  
  const TensorType& variance() const {
    return variance_;
  }
  
  TensorType stddev() const {
    TensorType stddev;
    stddev.copy(variance_);
    // set name of the tensor
    std::string name =  variance_.name();
    boost::replace_all(name, "variance", "stddev");
    stddev.setName(name);
    // assign value of standard deviation
    stddev.asEigenMatrix() = stddev.asEigenMatrix().array().sqrt();
    return stddev;
  }
  
  const ScalarType& count() const {
    return count_;
  }
  
  /*
   * Operations
   */
  
  void initialize(const TensorType& mean, const TensorType& variance) {
    NFATAL_IF(mean.empty() || variance.empty(), "[" << namescope() << "]: Cannot initialize from empty distribution tensors!");
    NFATAL_IF(mean.dimensions() != mean_.dimensions(), "[" << namescope() << "]: Dimensions of 'mean' are not compatible!");
    NFATAL_IF(variance.dimensions() != variance_.dimensions(), "[" << namescope() << "]: Dimensions of 'variance' are not compatible!");
    mean_.copy(mean);
    variance_.copy(variance);
    count_ = 1;
  }
  
  void initialize(const TensorType& samples) {
    NFATAL_IF(samples.empty(), "[" << namescope() << "]: Cannot initialize from empty samples tensor!");
    NFATAL_IF(samples.datumDimensions() != mean_.datumDimensions(),
      "[" << namescope() << "]: Datum dimensions of 'samples' are not compatible!");
    auto moments = math::moments(samples);
    mean_ = std::move(moments.first);
    variance_ = std::move(moments.second);
    count_ = samples.totalTimeSteps();
  }
  
  void update(const TensorType& samples) {
    NFATAL_IF(samples.empty(), "[" << namescope() << "]: Cannot update from empty samples tensor!");
    NFATAL_IF(samples.datumDimensions() != mean_.datumDimensions(),
      "[" << namescope() << "]: Datum dimensions of 'samples' are not compatible!");
    auto samples_count = static_cast<ScalarType>(samples.totalTimeSteps());
    auto new_count = count_ + samples_count;
    auto moments = math::moments(samples);
    auto delta = moments.first - mean_;
    delta *= (samples_count/new_count);
    mean_ += delta;
    delta.asEigenMatrix().array() = delta.asEigenMatrix().array().square();
    delta *= (count_/samples_count);
    auto m_a = variance_ * ((count_-1)/(new_count-1));
    auto m_b = moments.second * ((samples_count-1)/(new_count-1));
    variance_ = m_a + m_b + delta;
    count_ = new_count;
  }
  
  void normalize(TensorType& samples) {
    NFATAL_IF(samples.empty(), "[" << namescope() << "]: Cannot normalize empty samples tensor!");
    NFATAL_IF(samples.datumDimensions() != mean_.datumDimensions(),
      "[" << namescope() << "]: Datum dimensions of 'samples' are not compatible!");
    TensorType stddev;
    stddev.copy(variance_);
    stddev.asEigenMatrix().array() = stddev.asEigenMatrix().array().sqrt();
    for (size_t b = 0; b < samples.batches(); ++b) {
      for (size_t t = 0; t < samples.timesteps()[b]; ++t) {
        auto map = samples(t,b);
        map -= mean_;
        map /= stddev;
      }
    }
  }
  
  /*
   * Helper functions
   */
  
  friend std::ostream& operator<< (std::ostream& os, const TensorStatistic& statistic) {
    os << "\n<TensorStatistic>:\n\n[SampleCount]: " << statistic.count_ << "\n" << statistic.mean_ << statistic.variance_;
    return os;
  }
  
private:
  TensorType mean_;
  TensorType variance_;
  ScalarType count_{1};
};

} // namespace noesis

#endif // NOESIS_FRAMEWORK_MATH_TENSOR_STATISTIC_HPP_

/* EOF */
