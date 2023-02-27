/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_MATH_TENSOR_TUPLE_STATISTIC_HPP_
#define NOESIS_FRAMEWORK_MATH_TENSOR_TUPLE_STATISTIC_HPP_

// C++
#include <boost/algorithm/string/replace.hpp>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/TensorTuple.hpp"
#include "noesis/framework/math/statistics.hpp"

namespace noesis {

/*!
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 */
template<typename ScalarType_>
class TensorTupleStatistic final: core::Object
{
public:
  // Aliases
  using ScalarType = ScalarType_;
  using TensorType = Tensor<ScalarType>;
  using TensorTupleType = TensorTuple<ScalarType>;
  
  /*
   * Instantiation
   */
  
  explicit TensorTupleStatistic(const std::string& name, const std::string& scope, bool verbose):
    core::Object(name, scope, verbose),
    mean_(utils::make_namescope({scope, name, "mean"})),
    variance_(utils::make_namescope({scope, name, "variance"}))
  {
  }
  
  ~TensorTupleStatistic() final = default;
  
  /*
   * Configurations
   */
  
  void configure(const TensorsSpec& spec) {
    mean_.setFromSpec(spec);
    mean_.setConstant(0);
    variance_.setFromSpec(spec);
    variance_.setConstant(1);
  }
  
  /*
   * Properties
   */
  
  const TensorTupleType& mean() const {
    return mean_;
  }
  
  const TensorTupleType& variance() const {
    return variance_;
  }
  
  TensorTupleType stddev() const {
    TensorTupleType stddev;
    stddev.copy(variance_);
    for (size_t k = 0; k < stddev.size(); ++k) {
      // set name of the tensor
      std::string name =  variance_[k].name();
      boost::replace_all(name, "variance", "stddev");
      stddev.setName(k, name);
      // assign value of standard deviation
      stddev[k].asEigenMatrix() = stddev[k].asEigenMatrix().array().sqrt();
    }
    return stddev;
  }
  
  const ScalarType& count() const {
    return count_;
  }
  
  /*
   * Operations
   */
  
  void initialize(const TensorTupleType& mean, const TensorTupleType& variance) {
    NFATAL_IF(mean.empty() || variance.empty(), "[" << namescope() << "]: Cannot initialize from empty distribution tuples!");
    NFATAL_IF(mean.dimensions() != mean_.dimensions(), "[" << namescope() << "]: Dimensions of 'mean' are not compatible!");
    NFATAL_IF(variance.dimensions() != variance_.dimensions(), "[" << namescope() << "]: Dimensions of 'variance' are not compatible!");
    mean_.copy(mean);
    variance_.copy(variance);
    count_ = 1;
  }
  
  void initialize(const TensorTupleType& samples) {
    NFATAL_IF(samples.empty(), "[" << namescope() << "]: Cannot initialize from empty samples tuple!");
    NFATAL_IF(samples.datumDimensions() != mean_.datumDimensions(),
              "[" << namescope() << "]: Datum dimensions of 'samples' are not compatible!");
    for (size_t k = 0; k < samples.size(); ++k) {
      auto moments = math::moments(samples[k]);
      mean_[k] = std::move(moments.first);
      variance_[k] = std::move(moments.second);
    }
    count_ = samples[0].totalTimeSteps();
  }
  
  void update(const TensorTupleType& samples) {
    NFATAL_IF(samples.empty(), "[" << namescope() << "]: Cannot update from empty samples tuple!");
    NFATAL_IF(samples.datumDimensions() != mean_.datumDimensions(),
              "[" << namescope() << "]: Datum dimensions of 'samples' are not compatible!");
    auto samples_count = static_cast<ScalarType>(samples[0].totalTimeSteps());
    auto new_count = count_ + samples_count;
    for (size_t k = 0; k < samples.size(); ++k) {
      auto moments = math::moments(samples[k]);
      auto delta = moments.first - mean_[k];
      delta *= (samples_count/new_count);
      mean_[k] += delta;
      delta.asEigenMatrix().array() = delta.asEigenMatrix().array().square();
      delta *= (count_/samples_count);
      auto m_a = variance_[k] * ((count_-1)/(new_count-1));
      auto m_b = moments.second * ((samples_count-1)/(new_count-1));
      variance_[k] = m_a + m_b + delta;
    }
    count_ = new_count;
  }
  
  void normalize(TensorTupleType& samples) {
    NFATAL_IF(samples.empty(), "[" << namescope() << "]: Cannot normalize empty samples tuple!");
    NFATAL_IF(samples.datumDimensions() != mean_.datumDimensions(),
              "[" << namescope() << "]: Datum dimensions of 'samples' are not compatible!");
    for (size_t k = 0; k < samples.size(); ++k) {
      TensorType stddev;
      stddev.copy(variance_[k]);
      stddev.asEigenMatrix().array() = stddev.asEigenMatrix().array().sqrt();
      for (size_t b = 0; b < samples[k].batches(); ++b) {
        for (size_t t = 0; t < samples[k].timesteps()[b]; ++t) {
          auto map = samples[k](t,b);
          map -= mean_[k];
          map /= stddev;
        }
      }
    }
  }

  /*
   * Helper functions
   */

  friend std::ostream& operator<< (std::ostream& os, const TensorTupleStatistic& statistic) {
    os << "\n<TensorTupleStatistic>:\n\n[SampleCount]: " << statistic.count_ << "\n" << statistic.mean_ << statistic.variance_;
    return os;
  }
  
private:
  TensorTupleType mean_;
  TensorTupleType variance_;
  ScalarType count_{1};
};

} // namespace noesis

#endif // NOESIS_FRAMEWORK_MATH_TENSOR_TUPLE_STATISTIC_HPP_

/* EOF */
