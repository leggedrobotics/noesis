/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_ALGORITHM_POLICY_EVALUATION_CLIPPED_POLICY_EVALUATION_HPP_
#define NOESIS_RL_ALGORITHM_POLICY_EVALUATION_CLIPPED_POLICY_EVALUATION_HPP_

// Noesis
#include "noesis/framework/log/metric.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"
#include "noesis/rl/function/StateValueFunction.hpp"

namespace noesis {
namespace algorithm {

/*!
 * @brief A policy evaluation class implementing the TD(1) algorithm
 * with clipping of the optimization objective according to [1]. It is
 * used for training state-value function approximators.
 *
 * For details regarding the implementation of TD(1) (a.k.a Monte-Carlo
 * TD(lambda) with lambda=1) see `value_estimation.hpp`.
 *
 * The clipping of the objective serves as an approximate first-order method
 * of enforcing a trust-region constraint on the optimization objective.
 *
 *  [1] John Schulman, Filipp Wolski, Prafulla Dhariwal, ALex Radford, Oleg Klimov,
 *      "Proximal Policy Optimization",
 *      arXiv:1707.06347, 2017
 *
 *  [2] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. Vol. 1. No. 1.
 *      Cambridge: MIT press, 1998.
 *
 * @note This is implementation is designed to operate as a batched method.
 * @tparam ScalarType_ The scalar type used for all containers and operations.
 */
template<typename ScalarType_>
class ClippedPolicyEvaluation final: public ::noesis::core::Object
{
public:
  
  // Aliases
  using Base = ::noesis::core::Object;
  using Scalar = ScalarType_;
  using Observations = ::noesis::mdp::Observations<Scalar>;
  using Actions = ::noesis::mdp::Actions<Scalar>;
  using Tensor = ::noesis::Tensor<Scalar>;
  using Metrics = ::noesis::log::Metrics<Scalar>;
  using ValueFunction = ::noesis::function::StateValueFunction<Scalar>;
  using Graph = ::noesis::core::Graph;
  
  // Constants
  static constexpr Scalar MaxScalar = std::numeric_limits<Scalar>::max();
  
  /*
   * Instantiation
   */

  ClippedPolicyEvaluation() = delete;
  
  ClippedPolicyEvaluation(ClippedPolicyEvaluation&& other) = default;
  ClippedPolicyEvaluation& operator=(ClippedPolicyEvaluation&& other) = default;
  
  ClippedPolicyEvaluation(const ClippedPolicyEvaluation& other) = delete;
  ClippedPolicyEvaluation& operator=(const ClippedPolicyEvaluation& other) = delete;
  
  explicit ClippedPolicyEvaluation(
      ValueFunction* critic,
      const std::string& scope,
      const std::string& name):
    Base(name, scope),
    clipping_(0.1, utils::make_namescope({scope, name, "clipping"}), {0.0, 1.0}),
    valueErrorCoefficient_(1.0, utils::make_namescope({scope, name, "ve_coefficient"}), {0.0, MaxScalar}),
    l2Regularization_(0.0, utils::make_namescope({scope, name, "l2_regularization"}), {0.0, MaxScalar}),
    learningRate_(1e-3, utils::make_namescope({scope, name, "learning_rate"}), {0.0, MaxScalar}),
    maxGradientNorm_(MaxScalar, utils::make_namescope({scope, name, "max_grad_norm"}), {0.0, MaxScalar}),
    valueFunction_(critic)
  {
    const auto ns = namescope();
    NFATAL_IF(!valueFunction_, "[" << ns << "]: Value-function pointer is invalid (nullptr)!");
    graph_ = valueFunction_->graph();
    NFATAL_IF(!graph_, "[" << ns << "]: Graph pointer is invalid (nullptr)!");
    // Add the hyper-parameters to the global manager
    hyperparam::manager->addParameter(clipping_);
    hyperparam::manager->addParameter(valueErrorCoefficient_);
    hyperparam::manager->addParameter(l2Regularization_);
    hyperparam::manager->addParameter(learningRate_);
    hyperparam::manager->addParameter(maxGradientNorm_);
  }

  ~ClippedPolicyEvaluation() override {
    // Remove hyper-parameters from the global manager
    hyperparam::manager->removeParameter(clipping_);
    hyperparam::manager->removeParameter(valueErrorCoefficient_);
    hyperparam::manager->removeParameter(l2Regularization_);
    hyperparam::manager->removeParameter(learningRate_);
    hyperparam::manager->removeParameter(maxGradientNorm_);
  }
  
  /*
   * Properties
   */
  
  const Metrics& metrics() const {
    return metrics_;
  }
  
  Graph* graph() const {
    return graph_;
  }
  
  /*
   * Operations
   */

  void configure() {
    const auto ns = this->namescope();
    NINFO("[" << ns << "]: Clipping: " << static_cast<Scalar>(clipping_));
    NINFO("[" << ns << "]: Value-error coefficient: " << static_cast<Scalar>(valueErrorCoefficient_));
    NINFO("[" << ns << "]: L2-norm regularization: " << static_cast<Scalar>(l2Regularization_));
    NINFO("[" << ns << "]: Max. gradient norm: " << static_cast<Scalar>(maxGradientNorm_));
    // NOTE: the inputs buffer consists of the N observations tensors
    // plus the value targets and previous values --> dims={1, N+2}
    auto obsSpec = valueFunction_->observations_spec();
    auto obsScope = valueFunction_->observations_scope();
    size_t n = obsSpec.size();
    trainInputsBuffer_.clear();
    for (size_t k=0; k < n; k++) { trainInputsBuffer_.emplace_back(utils::make_namescope({obsScope, obsSpec[k].first})); }
    trainInputsBuffer_.emplace_back(utils::make_namescope({ns, "values/targets"}));
    trainInputsBuffer_.emplace_back(utils::make_namescope({ns, "values/previous"}));
    trainOutputsBuffer_.clear();
    trainOutputsBuffer_.emplace_back("loss/value_prediction/squared_clipped/loss_op");
    trainOutputsBuffer_.emplace_back("loss/l2_regularization/loss_op");
    trainOutputsBuffer_.emplace_back("loss/total/loss_op");
    trainOutputsBuffer_.emplace_back("gradient/norm_op");
    trainOutputsBuffer_.emplace_back("gradient/clipped_norm_op");
    trainOutputsBuffer_.emplace_back("learning_rate/value");
    trainOutputsBuffer_.emplace_back("clipping/value");
    for (auto& output: trainOutputsBuffer_) { output.setName(utils::make_namescope({ns, output.name()})); }
    trainOperationName_ = utils::make_namescope({ns, "train/train_op"});
    metrics_.clear();
    metrics_.push_back("CPE/value_error_loss");
    metrics_.push_back("CPE/l2_regularization");
    metrics_.push_back("CPE/total_loss");
    metrics_.push_back("CPE/gradient_norm");
    metrics_.push_back("CPE/clipped_gradient_norm");
    metrics_.push_back("CPE/learning_rate");
    metrics_.push_back("CPE/clipping");
  }
  
  void initialize()  {
    const auto ns = this->namescope();
    NFATAL_IF(!graph_->isActive(), "[" << ns << "]: Failed to initialize: Graph session is not active!");
    graph_->writeModifiable<Scalar>(ns + "/configurations/clipping", clipping_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/ve_weight", valueErrorCoefficient_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/l2_regularization", l2Regularization_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/learning_rate", learningRate_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/max_grad_norm", maxGradientNorm_);
    metrics_.reset();
  }
  
  void train(
      const Observations& observations,
      const Tensor& target_values,
      const Tensor& values) {
    // Assign new data to the forwarding buffers.
    size_t n = observations.size();
    for (size_t k = 0; k < n; k++) { trainInputsBuffer_[k] = observations[k]; }
    trainInputsBuffer_[n] = target_values;
    trainInputsBuffer_[n+1] = values;
    // Perform gradient descent
    graph_->run(trainInputsBuffer_, trainOperationName_, trainOutputsBuffer_);
    metrics_["CPE/value_error_loss"] = trainOutputsBuffer_[0][0];
    metrics_["CPE/l2_regularization"] = trainOutputsBuffer_[1][0];
    metrics_["CPE/total_loss"] = trainOutputsBuffer_[2][0];
    metrics_["CPE/gradient_norm"] = trainOutputsBuffer_[3][0];
    metrics_["CPE/clipped_gradient_norm"] = trainOutputsBuffer_[4][0];
    metrics_["CPE/learning_rate"] = trainOutputsBuffer_[5][0];
    metrics_["CPE/clipping"] = trainOutputsBuffer_[6][0];
  }
  
  void update() {
    // Progress the local_step counter responsible for
    // computing the damping and learning-rate decays.
    // NOTE: We use a separate counter for the optimizer
    // since we may execute multiple SGD steps at each
    // algorithm iteration.
    graph_->run(namescope() + "/decays/step_counter/increment_op");
  }

private:
  //! @brief Container of metrics recorded by this class.
  Metrics metrics_;
  //! @brief The interpolation factor for the computed values
  hyperparam::HyperParameter<Scalar> clipping_;
  //! @brief Weight of the value error loss term.
  hyperparam::HyperParameter<Scalar> valueErrorCoefficient_;
  //! @brief Weight of the entropy regularization loss term.
  hyperparam::HyperParameter<Scalar> l2Regularization_;
  //! @brief The optimizer's learning rate used for SGD.
  hyperparam::HyperParameter<Scalar> learningRate_;
  //! @brief The maximal value of the gradient for gradient descent
  hyperparam::HyperParameter<Scalar> maxGradientNorm_;
  //! @brief The value inference operation name in the graph
  std::string trainOperationName_;
  //! @brief Buffer for training input samples and targets
  std::vector<Tensor> trainInputsBuffer_;
  //! @brief Buffer for training output losses
  std::vector<Tensor> trainOutputsBuffer_;
  //! @brief The value function we would like to estimate the values of
  ValueFunction* valueFunction_{nullptr}; // TODO: Remove and make pure
  //! @brief The noesis::Graph instance with which to execute operations.
  Graph* graph_{nullptr};
};

} // namespace algorithm
} // namespace noesis

#endif // NOESIS_RL_ALGORITHM_POLICY_EVALUATION_CLIPPED_POLICY_EVALUATION_HPP_

/* EOF */
