/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_TRAIN_MONITOR_INTERFACE_HPP_
#define NOESIS_GYM_TRAIN_MONITOR_INTERFACE_HPP_

// Noesis
#include <noesis/framework/log/metric.hpp>

namespace noesis {
namespace gym {

template <class ScalarType_>
class MonitorInterface
{
public:
  
  // Aliases
  using Scalar = ScalarType_;
  using Metrics = ::noesis::log::Metrics<Scalar>;
  
  /*
   * Instantiation
   */

  MonitorInterface() = default;
  
  ~MonitorInterface() = default;
  
  /*
   * Properties
   */
  
  virtual const Metrics& metrics() const = 0;
  
  virtual std::string info() const = 0;
  
  /*
   * Operations
   */

  virtual void configure() = 0;
  
  virtual void reset() = 0;
  
  virtual bool update() = 0;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_TRAIN_MONITOR_INTERFACE_HPP_

/* EOF */
