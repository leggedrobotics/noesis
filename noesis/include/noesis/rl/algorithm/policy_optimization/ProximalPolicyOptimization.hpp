/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_ALGORITHM_POLICY_OPTIMIZATION_PPO_HPP_
#define NOESIS_RL_ALGORITHM_POLICY_OPTIMIZATION_PPO_HPP_

// Noesis
#include "noesis/framework/log/metric.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"
#include "noesis/rl/math/optimization.hpp"
#include "noesis/rl/function/StochasticPolicy.hpp"

namespace noesis {
namespace algorithm {

template<typename ScalarType_>
class ProximalPolicyOptimization: public ::noesis::core::Object
{
public:
  
  // Aliases
  using Base = ::noesis::core::Object;
  using Scalar = ScalarType_;
  using Observations = ::noesis::mdp::Observations<Scalar>;
  using Actions = ::noesis::mdp::Actions<Scalar>;
  using Tensor = ::noesis::Tensor<Scalar>;
  using Metrics = ::noesis::log::Metrics<Scalar>;
  using PolicyFunction = ::noesis::function::StochasticPolicy<Scalar>;
  using Graph = ::noesis::core::Graph;
  
  // Constants
  static constexpr auto MaxScalar = std::numeric_limits<Scalar>::max();
  
  /*
   * Instantiation
   */

  ProximalPolicyOptimization() = delete;
  
  ProximalPolicyOptimization(ProximalPolicyOptimization&& other) = default;
  ProximalPolicyOptimization& operator=(ProximalPolicyOptimization&& other) = default;
  
  ProximalPolicyOptimization(const ProximalPolicyOptimization& other) = delete;
  ProximalPolicyOptimization& operator=(const ProximalPolicyOptimization& other) = delete;
  
  explicit ProximalPolicyOptimization(
    PolicyFunction* policy,
    const std::string& scope,
    const std::string& name):
    Base(name, scope),
    clipping_(0.2, utils::make_namescope({scope, name, "clipping"}), {0.0, 1.0}),
    klDivergencePenalty_(0.0, utils::make_namescope({scope, name, "kld_penalty"}), {0.0, MaxScalar}),
    klDivergenceTarget_(0.0, utils::make_namescope({scope, name, "kld_target"}), {0.0, MaxScalar}),
    entropyWeight_(0.0, utils::make_namescope({scope, name, "entropy_weight"}), {0.0, MaxScalar}),
    learningRate_(1e-3, utils::make_namescope({scope, name, "learning_rate"}), {0.0, MaxScalar}),
    maxGradientNorm_(0.1, utils::make_namescope({scope, name, "max_grad_norm"}), {0.0, MaxScalar}),
    useClipping_(true, utils::make_namescope({scope, name, "use_clipping"})),
    useAdaptiveKld_(false, utils::make_namescope({scope, name, "use_adaptive_kld"})),
    policyFunction_(policy)
  {
    const auto ns = this->namescope();
    NFATAL_IF(!policyFunction_, "[" << ns << "]: Policy-function pointer is invalid (nullptr)!");
    graph_ = policyFunction_->graph();
    NFATAL_IF(!graph_, "[" << ns << "]: Graph pointer is invalid (nullptr)!");
    // Add hyper-parameters to the global hyper-parameter manager
    hyperparam::manager->addParameter(clipping_);
    hyperparam::manager->addParameter(klDivergencePenalty_);
    hyperparam::manager->addParameter(klDivergenceTarget_);
    hyperparam::manager->addParameter(entropyWeight_);
    hyperparam::manager->addParameter(learningRate_);
    hyperparam::manager->addParameter(maxGradientNorm_);
    hyperparam::manager->addParameter(useClipping_);
    hyperparam::manager->addParameter(useAdaptiveKld_);
  }

  ~ProximalPolicyOptimization() override {
    // Remove hyper-parameters from the global hyper-parameter manager
    hyperparam::manager->removeParameter(clipping_);
    hyperparam::manager->removeParameter(klDivergencePenalty_);
    hyperparam::manager->removeParameter(klDivergenceTarget_);
    hyperparam::manager->removeParameter(entropyWeight_);
    hyperparam::manager->removeParameter(learningRate_);
    hyperparam::manager->removeParameter(maxGradientNorm_);
    hyperparam::manager->removeParameter(useClipping_);
    hyperparam::manager->removeParameter(useAdaptiveKld_);
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
    NINFO("[" << ns << "]: Using clipping: " << std::boolalpha << static_cast<bool>(useClipping_));
    NINFO("[" << ns << "]: Using adaptive KL-divergence penalty: " << std::boolalpha << static_cast<bool>(useAdaptiveKld_));
    NINFO("[" << ns << "]: Clipping: " << static_cast<Scalar>(clipping_));
    NINFO("[" << ns << "]: KL-divergence penalty: " << static_cast<Scalar>(klDivergencePenalty_));
    NINFO("[" << ns << "]: Entropy weight.: " << static_cast<Scalar>(entropyWeight_));
    NINFO("[" << ns << "]: Adam Learning Rate: " << static_cast<Scalar>(learningRate_));
    NINFO("[" << ns << "]: Max. gradient norm: " << static_cast<Scalar>(maxGradientNorm_));
    configureBuffers();
    metrics_.clear();
    metrics_.push_back("PPO/probability_ratio");
    metrics_.push_back("PPO/entropy");
    metrics_.push_back("PPO/kl_divergence");
    metrics_.push_back("PPO/clip");
    metrics_.push_back("PPO/clip_fraction");
    metrics_.push_back("PPO/loss");
    metrics_.push_back("PPO/policy_gradient_norm");
    metrics_.push_back("PPO/clipped_policy_gradient_norm");
    metrics_.push_back("PPO/learning_rate");
  }
  
  void initialize() {
    const auto ns = this->namescope();
    NFATAL_IF(!graph_->isActive(), "[" << ns << "]: Failed to initialize: Graph session is not active!");
    graph_->writeModifiable<Scalar>(ns + "/configurations/learning_rate", learningRate_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/kld_penalty", klDivergencePenalty_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/clipping", clipping_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/entropy_weight", entropyWeight_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/max_grad_norm", maxGradientNorm_);
    metrics_.reset();
  }
  
  void logprobs(const Observations& observations, const Actions& actions, Tensor& logprobs) {
    size_t k=0;
    for (size_t i = 0; i < actions.size(); i++) { logprobInputsBuffer_[k++] = actions[i]; }
    for (size_t i = 0; i < observations.size(); i++) { logprobInputsBuffer_[k++] = observations[i]; }
    graph_->run(logprobInputsBuffer_, logprobOutputsBuffer_);
    logprobs = logprobOutputsBuffer_;
    std::vector<size_t> shape = {1, logprobs.size()};
    if (observations.isBatched()) { shape.push_back(observations.batches()); }
    logprobs.reshape(shape, observations.isBatched());
  }
  
  void train(
    const Observations& observations,
    const Actions& actions,
    const Tensor& advantages,
    const Tensor& logprobs
  ) {
    // Update the training buffers
    size_t k=0;
    trainInputsBuffer_[k++] = advantages;
    trainInputsBuffer_[k++] = logprobs;
    for (size_t i = 0; i < actions.size(); i++) { trainInputsBuffer_[k++] = actions[i]; }
    for (size_t i = 0; i < observations.size(); i++) { trainInputsBuffer_[k++] = observations[i]; }
    // Execute a single instance of gradient descent and record loss terms
    graph_->run(trainInputsBuffer_, trainOperationName_, trainOutputsBuffer_);
    metrics_[MeanProbabilityRatio] = trainOutputsBuffer_[0][0];
    metrics_[MeanEntropy] = trainOutputsBuffer_[1][0];
    metrics_[MeanKlDivergence] = trainOutputsBuffer_[2][0];
    metrics_[ClipLoss] = trainOutputsBuffer_[3][0];
    metrics_[ClipFraction] = trainOutputsBuffer_[4][0];
    metrics_[TotalLoss] = trainOutputsBuffer_[5][0];
    metrics_[PolicyGradientNorm] = trainOutputsBuffer_[6][0];
    metrics_[ClippedGradientNorm] = trainOutputsBuffer_[7][0];
    metrics_[LearningRate] = trainOutputsBuffer_[8][0];
  }
  
  void update() {
    const auto ns = namescope();
    // Optional KL-Divergence penalty adaptation
    if (useAdaptiveKld_) {
      Scalar kld = trainOutputsBuffer_[2][0];
      Scalar delta = klDivergenceTarget_;
      Scalar beta = readKlDivergencePenalty();
      if (kld < delta/1.5) {
        beta *= 0.5;
        writeKlDivergencePenalty(beta);
      } else if (kld > delta*1.5) {
        beta *= 2.0;
        writeKlDivergencePenalty(beta);
      }
      NINFO("[" << ns << "]: KL-divergence penalty: " << beta);
    }
    // Progress the local_step counter responsible for
    // computing the damping and learning-rate decays.
    // NOTE: We use a separate counter for the optimizer
    // since we may execute multiple SGD steps at each
    // algorithm iteration.
    graph_->run(ns + "/decays/step_counter/increment_op");
  }
  
private:

  void configureBuffers() {
    using namespace utils;
    const auto ns = namescope();
    auto obsScope = policyFunction_->observations_scope();
    auto obsSpec = policyFunction_->observations_spec();
    auto actSpec = policyFunction_->actions_spec();
    // Action log-probability computation buffers
    logprobOutputsBuffer_.setName(make_namescope({ns, "loss/probability_ratio/log_prob_op"}));
    logprobOutputsBuffer_.resize({1,1,1}, true);
    logprobInputsBuffer_.clear();
    for (auto& spec: actSpec) { logprobInputsBuffer_.emplace_back(make_namescope({ns, "inputs/actions", spec.first})); }
    for (auto& spec: obsSpec) { logprobInputsBuffer_.emplace_back(make_namescope({obsScope, spec.first})); }
    // Training sample inputs
    trainInputsBuffer_.clear();
    trainInputsBuffer_.emplace_back(make_namescope({ns, "inputs/advantages"}));
    trainInputsBuffer_.emplace_back(make_namescope({ns, "inputs/logprobs"}));
    for (auto& spec: actSpec) { trainInputsBuffer_.emplace_back(make_namescope({ns, "inputs/actions", spec.first})); }
    for (auto& spec: obsSpec) { trainInputsBuffer_.emplace_back(make_namescope({obsScope, spec.first})); }
    // Policy gradient estimation outputs
    trainOutputsBuffer_.clear();
    trainOutputsBuffer_.emplace_back("loss/probability_ratio/prob_ratio_op");
    trainOutputsBuffer_.emplace_back("loss/entropy/mean_entropy_op");
    trainOutputsBuffer_.emplace_back("loss/kl_divergence/mean_kld_op");
    trainOutputsBuffer_.emplace_back("loss/clip/clip_op");
    trainOutputsBuffer_.emplace_back("loss/clip/clip_fraction_op");
    trainOutputsBuffer_.emplace_back("loss/total/loss_op");
    trainOutputsBuffer_.emplace_back("gradient/norm_op");
    trainOutputsBuffer_.emplace_back("gradient/clipped_norm_op");
    trainOutputsBuffer_.emplace_back("learning_rate/value");
    for (auto& output: trainOutputsBuffer_) { output.setName(make_namescope({ns, output.name()})); }
    // Policy optimization operation
    trainOperationName_ = make_namescope({ns, "train/train_op"});
  }
  
  Scalar readKlDivergencePenalty() const {
    const auto ns = namescope();
    return graph_->readModifiable<Scalar>(ns + "/configurations/kld_penalty");
  }
  
  void writeKlDivergencePenalty(Scalar delta) {
    const auto ns = namescope();
    graph_->writeModifiable<Scalar>(ns + "/configurations/kld_penalty", delta);
  }
  
private:
  //! Defines indices for metrics collected by this class.
  enum Metric {
    MeanProbabilityRatio = 0,
    MeanEntropy,
    MeanKlDivergence,
    ClipLoss,
    ClipFraction,
    TotalLoss,
    PolicyGradientNorm,
    ClippedGradientNorm,
    LearningRate
  };
  //! @brief Container of metrics recorded by this class.
  Metrics metrics_;
  //! @brief Clipping operation applied to the L^CPI surrogate cost term.
  hyperparam::HyperParameter<Scalar> clipping_;
  //! @brief Weight of the KL-Divergence loss term.
  hyperparam::HyperParameter<Scalar> klDivergencePenalty_;
  //! @brief Target KL-Divergence used for the KL-penalty cost.
  hyperparam::HyperParameter<Scalar> klDivergenceTarget_;
  //! @brief Weight of the entropy regularization loss term.
  hyperparam::HyperParameter<Scalar> entropyWeight_;
  //! @brief The optimizer's learning rate used for SGD.
  hyperparam::HyperParameter<Scalar> learningRate_;
  //! @brief The maximal value of the gradient for gradient descent.
  hyperparam::HyperParameter<Scalar> maxGradientNorm_;
  //! @brief Enabled/disables the use of the clipped CPI loss term.
  hyperparam::HyperParameter<bool> useClipping_;
  //! @brief Enabled/disables the adaptive the KL-divergence loss term.
  hyperparam::HyperParameter<bool> useAdaptiveKld_;
  //! @brief Buffer storing computed action sample log-probabilities.
  Tensor logprobOutputsBuffer_;
  //! @brief Buffer forwarding samples used for computing action-sample log-probabilities
  std::vector<Tensor> logprobInputsBuffer_;
  //! @brief Buffers storing computed outputs for the primary training operation.
  std::vector<Tensor> trainOutputsBuffer_;
  //! @brief Buffers forwarding samples used for training operations.
  std::vector<Tensor> trainInputsBuffer_;
  //! @brief The gradient descent operation name in the graph.
  std::string trainOperationName_;
  //! @brief The policy we perform the optimization on.
  PolicyFunction* policyFunction_{nullptr};
  //! @brief The noesis::Graph instance with which to execute operations.
  Graph* graph_{nullptr};
};

} // namespace algorithm
} // namespace noesis

#endif // NOESIS_RL_ALGORITHM_POLICY_OPTIMIZATION_PPO_HPP_

/* EOF*/
