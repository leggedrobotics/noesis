/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_TRAIN_SAMPLER_INTERFACE_HPP_
#define NOESIS_GYM_TRAIN_SAMPLER_INTERFACE_HPP_

// Noesis
#include <noesis/framework/log/metric.hpp>

namespace noesis {
namespace gym {

template <typename ScalarType_>
class SamplerInterface
{
public:
  
  // Ensure that ScalarType_ is one of the supported types
  static_assert(
    std::is_arithmetic<ScalarType_>::value,
    "ScalarType_ must be an arithmetic type, e.g. {int, float, double, etc..}"
  );
  
  // Aliases
  using Scalar = ScalarType_;
  using Metrics = ::noesis::log::Metrics<Scalar>;
  
  /*
   * Instantiation
   */
  
  SamplerInterface() = default;
  
  virtual ~SamplerInterface() = default;
  
  /*
   * Properties
   */

  virtual const std::vector<size_t>& getSampleCounters() const = 0;
  
  virtual const std::vector<size_t>& getBatchCounters() const = 0;
  
  virtual size_t getTotalSamples() const = 0;
  
  virtual size_t getTotalBatches() const = 0;
  
  virtual const Metrics& metrics() const = 0;
  
  virtual std::string info() const = 0;
  
  /*
   * Operations
   */

  virtual void configure() = 0;
  
  virtual void reset() = 0;
  
  virtual bool sample() = 0;
  
  virtual void process() = 0;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_TRAIN_SAMPLER_INTERFACE_HPP_

/* EOF */
