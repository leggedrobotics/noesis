/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_FUNCTION_VALUE_STATE_VALUE_FUNCTION_HPP_
#define NOESIS_RL_FUNCTION_VALUE_STATE_VALUE_FUNCTION_HPP_

// Noesis
#include "noesis/framework/core/Graph.hpp"
#include "noesis/framework/core/Object.hpp"
#include "noesis/mdp/types.hpp"

namespace noesis {
namespace function {

template<typename ScalarType_>
class StateValueFunction final: public ::noesis::core::Object
{
public:
  
  // Aliases
  using Base = ::noesis::core::Object;
  using Scalar = ScalarType_;
  using Observations = ::noesis::mdp::Observations<Scalar>;
  using Tensor = ::noesis::Tensor<Scalar>;
  using Graph = ::noesis::core::Graph;

  /*
   * Instantiation
   */

  StateValueFunction() = delete;
  
  StateValueFunction(StateValueFunction&& other) = default;
  StateValueFunction& operator=(StateValueFunction&& other) = default;
  
  StateValueFunction(const StateValueFunction& other) = delete;
  StateValueFunction& operator=(const StateValueFunction& other) = delete;
  
  explicit StateValueFunction(
      const TensorsSpec& observations_spec,
      const std::string& observations_scope,
      const std::string& operations_scope,
      const std::string& name,
      Graph* graph):
    Base(name, operations_scope),
    observations_(observations_scope, observations_spec, 1, 1),
    values_(utils::make_namescope({operations_scope, name, "value_distribution/mean"}), {1,1,1}, true),
    graph_(graph)
  {
    const auto ns = namescope();
    NFATAL_IF(!graph_, "[" << ns << "]: 'graph' argument is invalid (nullptr)!");
    NINFO("[" << ns << "]: Graph: " << graph_->namescope());
    NINFO("[" << ns << "]: Observations Spec: " << observations_.spec());
    NINFO("[" << ns << "]: Observations Scope: " << observations_.scope());
    NINFO("[" << ns << "]: Operations Scope: " << utils::make_namescope({operations_scope, name}));
  }
  
  explicit StateValueFunction(
      const TensorsSpec& observations_spec,
      const std::string& scope,
      const std::string& name,
      Graph* graph):
    StateValueFunction(observations_spec, scope, scope, name, graph)
  {
  }

  ~StateValueFunction() final = default;

  /*
   * Configurations
   */
  
  void setParameters(const Tensor& params) {
    const auto ns = namescope();
    NFATAL_IF(!graph_->isActive(), "[" << ns << "]: Failed to initialize: Graph session is not active!");
    Tensor buffer(ns + "/parameters/input");
    buffer = params;
    graph_->run<Scalar>(buffer, ns + "/parameters/set_op");
  }
  
  /*
   * Properties
   */

  TensorsSpec observations_spec() const {
    return observations_.spec();
  }
  
  std::string observations_scope() const {
    return observations_.scope();
  }
  
  size_t parameters_size() const {
    const auto ns = namescope();
    NFATAL_IF(!graph_->isActive(), "[" << ns << "]: Failed to initialize: Graph session is not active!");
    tensorflow::Tensor output;
    graph_->run(ns + "/parameters/get_size_op", output);
    return static_cast<size_t>(output.scalar<int>()());
  }
  
  Graph* graph() const {
    return graph_;
  }
  
  /*
   * Operations
   */
  
  void initialize() {
    const auto ns = namescope();
    NFATAL_IF(!graph_->isActive(), "[" << ns << "]: Failed to initialize: Graph session is not active!");
    NINFO("[" << ns << "]: Number of parameters: " << parameters_size());
  }
  
  void evaluate(const Observations& observations, Tensor& values) {
    observations_ = observations;
    graph_->run(observations_, values_);
    DNFATAL_IF(!values_.allFinite(),
      "[" << namescope() << "]: Computed values are invalid (NaN/Inf)! Observations were: " << observations_);
    values = values_;
  }
  
  void parameters(Tensor& params) const {
    graph_->run<Scalar>(namescope() + "/parameters/get_op", params);
  }
  
  Tensor parameters() const {
    Tensor params;
    parameters(params);
    return params;
  }

private:
  //! @brief A buffer for forwarding observation inputs for function evaluation.
  Observations observations_;
  //! @brief Buffer for storing predicted (mean) values.
  Tensor values_;
  //! @brief The noesis::Graph instance with which to execute operations.
  Graph* graph_{nullptr};
};

} // namespace function
} // namespace noesis

#endif // NOESIS_RL_FUNCTION_VALUE_STATE_VALUE_FUNCTION_HPP_

/* EOF */
