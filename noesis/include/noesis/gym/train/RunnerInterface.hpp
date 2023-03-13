/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_TRAIN_RUNNER_INTERFACE_HPP_
#define NOESIS_GYM_TRAIN_RUNNER_INTERFACE_HPP_

// Noesis
#include "noesis/framework/log/metric.hpp"
#include "noesis/framework/log/tensorboard.hpp"
#include "noesis/framework/core/Graph.hpp"
#include "noesis/gym/train/SamplerInterface.hpp"
#include "noesis/gym/train/MonitorInterface.hpp"

namespace noesis {
namespace gym {

template <typename ScalarType_>
class RunnerInterface
{
public:

  // Ensure that ScalarType_ is one of the supported types
  static_assert(
    std::is_arithmetic<ScalarType_>::value,
    "ScalarType_ must be an arithmetic type, e.g. {int, float, double, etc..}"
  );
  
  //! @brief Defines the modes in which a runner operate for sample collection.
  enum class RunMode {
    Samples,
    Batches,
    Iterations
  };
  
  // Aliases
  using Scalar = ScalarType_;
  using Metrics = ::noesis::log::Metrics<Scalar>;
  using SamplerPtr = ::noesis::gym::SamplerInterface<Scalar>*;
  using MonitorPtr = ::noesis::gym::MonitorInterface<Scalar>*;
  using LoggerPtr = ::noesis::log::TensorBoardLogger*;
  using GraphPtr = ::noesis::core::Graph*;
  
  /*
   * Instantiation
   */
  
  RunnerInterface() = default;
  
  virtual ~RunnerInterface() = default;
  
  
  /*
   * Configurations
   */

  virtual void setSampler(SamplerPtr sampler) = 0;
  
  virtual void setMonitor(MonitorPtr monitor) = 0;
  
  virtual void setLogger(LoggerPtr logger) = 0;
  
  virtual void setGraph(GraphPtr graph) = 0;
  
  /*
   * Properties
   */

  virtual const std::vector<size_t>& getSampleCounters() const = 0;
  
  virtual const std::vector<size_t>& getBatchCounters() const = 0;
  
  virtual size_t getTotalSamples() const = 0;
  
  virtual size_t getTotalBatches() const = 0;
  
  virtual size_t getTotalIterations() const = 0;
  
  virtual std::string info() const = 0;
  
  /*
   * Operations
   */

  virtual void configure() = 0;

  virtual void reset() = 0;
  
  virtual void run(RunMode mode, size_t duration) = 0;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_TRAIN_RUNNER_INTERFACE_HPP_

/* EOF */
