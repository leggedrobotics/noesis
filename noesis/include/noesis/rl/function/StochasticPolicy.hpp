/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_FUNCTION_POLICIES_STOCHASTIC_POLICY_HPP_
#define NOESIS_RL_FUNCTION_POLICIES_STOCHASTIC_POLICY_HPP_

// Noesis
#include "noesis/framework/core/Graph.hpp"
#include "noesis/framework/core/Object.hpp"
#include "noesis/mdp/types.hpp"

namespace noesis {
namespace function {

template<typename ScalarType_>
class StochasticPolicy final: public ::noesis::core::Object
{
public:
  
  // Aliases
  using Base = ::noesis::core::Object;
  using Scalar = ScalarType_;
  using Observations = ::noesis::mdp::Observations<Scalar>;
  using Actions = ::noesis::mdp::Actions<Scalar>;
  using Tensor = ::noesis::Tensor<Scalar>;
  using Graph = ::noesis::core::Graph;
  
  /*
   * Instantiation
   */

  StochasticPolicy() = delete;
  
  StochasticPolicy(StochasticPolicy&& other) = default;
  StochasticPolicy& operator=(StochasticPolicy&& other) = default;
  
  StochasticPolicy(const StochasticPolicy& other) = delete;
  StochasticPolicy& operator=(const StochasticPolicy& other) = delete;
  
  explicit StochasticPolicy(
      const TensorsSpec& actions_spec,
      const TensorsSpec& observations_spec,
      const std::string& observations_scope,
      const std::string& operations_scope,
      const std::string& name,
      Graph* graph):
    Base(name, operations_scope),
    observations_(observations_scope, observations_spec, 1, 1),
    actionModes_(utils::make_namescope({operations_scope, name, "action_distribution/mode"}), actions_spec, 1, 1),
    actionSamples_(utils::make_namescope({operations_scope, name, "action_distribution/sample"}), actions_spec, 1, 1),
    graph_(graph)
  {
    const auto ns = namescope();
    NFATAL_IF(!graph_, "[" << ns << "]: 'graph' argument is invalid (nullptr)!");
    NINFO("[" << ns << "]: Graph: " << graph_->namescope());
    NINFO("[" << ns << "]: Actions Spec: " << actionModes_.spec());
    NINFO("[" << ns << "]: Observations Spec: " << observations_.spec());
    NINFO("[" << ns << "]: Observations Scope: " << observations_.scope());
    NINFO("[" << ns << "]: Operations Scope: " << utils::make_namescope({operations_scope, name}));
  }
  
  explicit StochasticPolicy(
      const TensorsSpec& actions_spec,
      const TensorsSpec& observations_spec,
      const std::string& scope,
      const std::string& name,
      Graph* graph):
    StochasticPolicy(actions_spec, observations_spec, scope, scope, name, graph)
  {
  }

  ~StochasticPolicy() final = default;

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

  TensorsSpec actions_spec() const {
    return actionModes_.spec();
  }
  
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
  
  void mode(const Observations& observations, Actions& actions) {
    observations_ = observations;
    graph_->run(observations_, actionModes_);
    DNFATAL_IF(!actionModes_.allFinite(),
      "[" << namescope() << "]: Computed actions are invalid (NaN/Inf)! Observations were: " << observations_);
    actions = actionModes_;
  }
  
  void sample(const Observations& observations, Actions& actions) {
    observations_ = observations;
    graph_->run(observations_, actionSamples_);
    DNFATAL_IF(!actionSamples_.allFinite(),
      "[" << namescope() << "]: Computed actions are invalid (NaN/Inf)! Observations were: " << observations_);
    actions = actionSamples_;
  }
  
  void parameters(Tensor& params) const {
    graph_->run<Scalar>(namescope() + "/parameters/get_op", params);
  }
  
  Tensor parameters() const {
    Tensor params;
    parameters(params);
    return params;
  }
  
protected:
  //! @brief A buffer for forwarding observation inputs for function evaluation.
  Observations observations_;
  //! @brief A buffer for storing actions computed from the mode of the distribution.
  Actions actionModes_;
  //! @brief A buffer for storing actions sampled from the distribution.
  Actions actionSamples_;
  //! @brief The noesis::Graph instance with which to execute operations.
  Graph* graph_{nullptr};
};

} // namespace function
} // namespace noesis

#endif // NOESIS_FUNCTION_POLICIES_STOCHASTIC_POLICY_HPP_

/* EOF */
