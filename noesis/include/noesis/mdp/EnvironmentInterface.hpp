/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_MDP_ENVIRONMENT_INTERFACE_HPP_
#define NOESIS_MDP_ENVIRONMENT_INTERFACE_HPP_

// Noesis
#include "noesis/mdp/types.hpp"

namespace noesis {
namespace mdp {

template <typename ScalarType_>
class EnvironmentInterface
{
public:
  
  // Ensure that ScalarType_ is one of the supported types
  static_assert(
    std::is_arithmetic<ScalarType_>::value,
    "ScalarType_ must be an arithmetic type, e.g. {int, float, double, etc..}"
  );
  
  // Aliases
  using Scalar = ScalarType_;
  using Observations = ::noesis::mdp::Observations<Scalar>;
  using Actions = ::noesis::mdp::Actions<Scalar>;
  using Rewards = ::noesis::mdp::Rewards<Scalar>;
  using Metrics = ::noesis::mdp::Metrics<Scalar>;
  using Termination = ::noesis::mdp::Termination<Scalar>;
  using Terminations = ::noesis::mdp::Terminations<Scalar>;
  using Names = std::vector<std::string>;
  
  /*
   * Instantiation
   */

  EnvironmentInterface() = default;
  
  virtual ~EnvironmentInterface() = default;
  
  /*
   * Properties
   */
  
  virtual size_t batch_size() const = 0;
  
  virtual size_t history_size() const = 0;
  
  virtual size_t max_steps() const = 0;
  
  virtual Scalar time_limit() const = 0;
  
  virtual Scalar time_step() const = 0;
  
  virtual Scalar discount_factor() const = 0;
  
  virtual TensorsSpec actions_spec() const = 0;
  
  virtual TensorsSpec observations_spec() const = 0;
  
  virtual Names tasks() const = 0;
  
  virtual Actions& actions() = 0;
  
  virtual const Actions& actions() const = 0;
  
  virtual const Observations& observations() const = 0;
  
  virtual const Rewards& rewards() const = 0;
  
  virtual const Terminations& terminations() const = 0;
  
  virtual const Metrics& metrics() const = 0;
  
  virtual size_t steps() const = 0;
  
  virtual Scalar time() const = 0;
  
  virtual std::ostream& info(std::ostream& os) const = 0;
  
  /*
   * Operations
   */
  
  virtual void configure() = 0;
  
  virtual void seed(int seed) = 0;
  
  virtual void reset() = 0;
  
  virtual void step() = 0;
  
  friend std::ostream& operator<<(std::ostream& os, const EnvironmentInterface& interface) { return interface.info(os); }
};

} // namespace mdp
} // namespace noesis

#endif // NOESIS_MDP_ENVIRONMENT_INTERFACE_HPP_

/* EOF */
