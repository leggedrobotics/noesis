/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_ALGORITHM_POLICY_OPTIMIZATION_TRPO_HPP_
#define NOESIS_RL_ALGORITHM_POLICY_OPTIMIZATION_TRPO_HPP_

// Noesis
#include "noesis/framework/log/metric.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"
#include "noesis/rl/math/optimization.hpp"
#include "noesis/rl/function/StochasticPolicy.hpp"

namespace noesis {
namespace algorithm {

/*!
 * @brief Implements Trust Region Policy Optimization (TRPO) by Schulman et al [2,3].
 *
 * This class provides the Policy Optimization (PO) part of TRPO, namely:
 *  1. Computes a policy gradient estimate using the Conservative Policy Iteration
 *     (CPO) loss described in Kakade et al [1].
 *  2. Computes the natural gradient estimate using KL-divergence constraint to
 *     compute an optimizing step-direction.
 *  3. Computes Backtracking Line-Search (BLS) to select an optimizing step-length.
 *
 * This implementation realizes the variant of TRPO described in [2,3] and also inherits
 * elements from the implementation found in OpenAI Baselines [4].
 *
 * [1] Sham Kakade and John Langford,
 *     "Approximately Optimal Approximate Reinforcement Learning",
 *     In ICML, vol. 2, pp. 267-274. 2002
 *
 * [2] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz,
 *     "Trust Region Policy Optimization",
 *      In International conference on machine learning, pp. 1889-1897, 2015.
 *
 * [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel,
 *     "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
 *     arXiv:1506.02438, 2018
 *
*  [4] TRPI, OpenAI Baselines, (https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
 *
 */
template<typename ScalarType_>
class TrustRegionPolicyOptimization: public ::noesis::core::Object
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
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using LossFuncional = std::function<Scalar(const VectorX&)>;
  
  // Constants
  static constexpr auto MaxInt = std::numeric_limits<int>::max();
  static constexpr auto MaxScalar = std::numeric_limits<Scalar>::max();
  
  /*
   * Instantiation
   */

  TrustRegionPolicyOptimization() = delete;
  
  TrustRegionPolicyOptimization(TrustRegionPolicyOptimization&& other) = default;
  TrustRegionPolicyOptimization& operator=(TrustRegionPolicyOptimization&& other) = default;
  
  TrustRegionPolicyOptimization(const TrustRegionPolicyOptimization& other) = delete;
  TrustRegionPolicyOptimization& operator=(const TrustRegionPolicyOptimization& other) = delete;
  
  explicit TrustRegionPolicyOptimization(
      PolicyFunction* policy,
      const std::string& scope,
      const std::string& name):
    Base(name, scope),
    klDivergenceThreshold_(0.01, utils::make_namescope({scope, name, "kld_threshold"}), {0.0, MaxScalar}),
    entropyWeight_(0.0, utils::make_namescope({scope, name, "entropy_weight"}), {0.0, MaxScalar}),
    cgDamping_(0.1, utils::make_namescope({scope, name, "cg_damping"}), {0.0, MaxScalar}),
    cgTolerance_(1.0e-15, utils::make_namescope({scope, name, "cg_tolerance"}), {0.0, MaxScalar}),
    cgMaxIterations_(100, utils::make_namescope({scope, name, "cg_max_iterations"}), {1, MaxInt}),
    lineSearchContractionFactor_(0.5, utils::make_namescope({scope, name, "line_search_contraction_factor"}), {0.0, 1.0}),
    lineSearchMaxIterations_(20, utils::make_namescope({scope, name, "line_search_max_iterations"}), {1, MaxInt}),
    policyFunction_(policy)
  {
    const auto ns = this->namescope();
    NFATAL_IF(!policyFunction_, "[" << ns << "]: Policy-function pointer is invalid (nullptr)!");
    graph_ = policyFunction_->graph();
    NFATAL_IF(!graph_, "[" << ns << "]: Graph pointer is invalid (nullptr)!");
    // Add hyper-parameters to the global hyper-parameter manager
    hyperparam::manager->addParameter(klDivergenceThreshold_);
    hyperparam::manager->addParameter(entropyWeight_);
    hyperparam::manager->addParameter(cgDamping_);
    hyperparam::manager->addParameter(cgTolerance_);
    hyperparam::manager->addParameter(cgMaxIterations_);
    hyperparam::manager->addParameter(lineSearchContractionFactor_);
    hyperparam::manager->addParameter(lineSearchMaxIterations_);
  }

  ~TrustRegionPolicyOptimization() override {
    // Remove hyper-parameters from the global hyper-parameter manager
    hyperparam::manager->removeParameter(klDivergenceThreshold_);
    hyperparam::manager->removeParameter(entropyWeight_);
    hyperparam::manager->removeParameter(cgDamping_);
    hyperparam::manager->removeParameter(cgTolerance_);
    hyperparam::manager->removeParameter(cgMaxIterations_);
    hyperparam::manager->removeParameter(lineSearchContractionFactor_);
    hyperparam::manager->removeParameter(lineSearchMaxIterations_);
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
    NINFO("[" << ns << "]: KL-divergence threshold: " << static_cast<Scalar>(klDivergenceThreshold_));
    NINFO("[" << ns << "]: Entropy weight: " << static_cast<Scalar>(entropyWeight_));
    NINFO("[" << ns << "]: CG: tolerance: " << static_cast<Scalar>(cgTolerance_));
    NINFO("[" << ns << "]: CG: damping: " << static_cast<Scalar>(cgDamping_));
    NINFO("[" << ns << "]: CG: Max. iterations: " << static_cast<int>(cgMaxIterations_));
    NINFO("[" << ns << "]: Line-Search: Contraction factor: " << static_cast<Scalar>(lineSearchContractionFactor_));
    NINFO("[" << ns << "]: Line-Search: Maximum iterations: " << static_cast<int>(lineSearchMaxIterations_));
    configureBuffers();
    metrics_.clear();
    metrics_.push_back("TRPO/probability_ratio");
    metrics_.push_back("TRPO/entropy");
    metrics_.push_back("TRPO/kl_divergence");
    metrics_.push_back("TRPO/cpi");
    metrics_.push_back("TRPO/loss");
    metrics_.push_back("TRPO/cg_error");
    metrics_.push_back("TRPO/policy_gradient_norm");
    metrics_.push_back("TRPO/natural_gradient_norm");
    metrics_.push_back("TRPO/max_gradient_norm");
    metrics_.push_back("TRPO/line_search_best_step");
    metrics_.push_back("TRPO/line_search_best_index");
  }
  
  void initialize() {
    const auto ns = this->namescope();
    NFATAL_IF(!graph_->isActive(), "[" << ns << "]: Failed to initialize: Graph session is not active!");
    graph_->writeModifiable<Scalar>(ns + "/configurations/entropy_weight", entropyWeight_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/cg_damping", cgDamping_);
    graph_->writeModifiable<Scalar>(ns + "/configurations/cg_tolerance", cgTolerance_);
    graph_->writeModifiable(
      ns + "/configurations/cg_max_iterations",
      tensorflow::DT_INT64,
      static_cast<tensorflow::int64>(cgMaxIterations_));
    policyFunction_->parameters(policyParameters_);
    metrics_.reset();
  }
  
  void train(
    const Observations& observations,
    const Actions& actions,
    const Tensor& advantages
  ) {
    const auto ns = this->namescope();
    const auto is_verbose = this->isVerbose();
    // Update the sample buffers
    size_t k=0;
    trainInputsBuffer_[k++] = advantages;
    for (size_t i = 0; i < actions.size(); i++) { trainInputsBuffer_[k++] = actions[i]; }
    for (size_t i = 0; i < observations.size(); i++) { trainInputsBuffer_[k++] = observations[i]; }
    // Retrieve current parameter vector
    auto parameters = policyFunction_->parameters();
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Parameters: " << parameters.info());
    DNFATAL_IF(!parameters.allFinite(), "[" << ns << "]: TRPO: Policy Parameters: Contains invalid values (NaN/Inf)!");
    // Compute the the CPI-based policy gradient.
    graph_->run(trainInputsBuffer_, trainOutputsBuffer_);
    const auto policyGradient = trainOutputsBuffer_.back().asFlat();
    DNFATAL_IF(!policyGradient.allFinite(), "[" << ns << "]: TRPO: Policy Gradient: Contains invalid values (NaN/Inf)!");
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Policy Gradient: Norm: " << policyGradient.norm());
    // Estimate the natural policy gradient.
    ngInputsBuffer_.front() = trainOutputsBuffer_.back();
    for (size_t i = 1; i < trainInputsBuffer_.size(); i++) { ngInputsBuffer_[i] = trainInputsBuffer_[i]; }
    graph_->run(ngInputsBuffer_, ngOutputsBuffer_);
    const auto naturalGradient = ngOutputsBuffer_.back().asFlat();
    DNFATAL_IF(!naturalGradient.allFinite(), "[" << ns << "]: TRPO: Natural Gradient: Contains invalid values (NaN/Inf)!");
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Natural Gradient: Norm: " << naturalGradient.norm());
    // Compute the maximum step-length
    const Scalar delta = klDivergenceThreshold_;
    const Scalar beta = std::sqrt(2 * delta / (naturalGradient.array()*policyGradient.array()).sum());
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Natural Gradient: Beta: " << beta);
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Natural Gradient: pg.dot(ng): " << (naturalGradient.array()*policyGradient.array()).sum());
    // **NOTE**: We invert the sign of the gradient because math::backtracking_line_search() performs minimization.
    VectorX gradient = -beta * naturalGradient;
    const auto maxGradNorm = gradient.norm();
    DNFATAL_IF(!gradient.allFinite(), "[" << ns << "]: TRPO: Natural Gradient: Max. gradient contains invalid values (NaN/Inf)!");
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Max. Gradient: Norm: " << maxGradNorm);
    // Perform line-search to optimize the step-size of the parameter update.
    const int k_max = lineSearchMaxIterations_;
    const Scalar rho = lineSearchContractionFactor_;
    const LossFuncional lsObjective = [this](const VectorX& theta){ return this->loss(theta); };
    const auto lsInfo = math::backtracking_line_search(lsObjective, VectorX(parameters.asFlat()), gradient, rho, k_max, is_verbose);
    NINFO_IF(is_verbose, "[" << ns << "]: TRPO: Opt. Gradient: Norm: " << lsInfo.first);
    // Update the in-graph policy parameters.
    // **NOTE**: We again invert the sign of parameter update because math::backtracking_line_search() performs minimization.
    parameters.asFlat() -= gradient;
    policyFunction_->setParameters(parameters);
    DNFATAL_IF(!parameters.allFinite(), "[" << ns << "]: TRPO: Policy Parameters: Contains invalid values (NaN/Inf)!");
    metrics_[MeanProbabilityRatio] = trainOutputsBuffer_[0][0];
    metrics_[MeanEntropy] = trainOutputsBuffer_[1][0];
    metrics_[CpiLoss] = trainOutputsBuffer_[2][0];
    metrics_[TotalLoss] = trainOutputsBuffer_[3][0];
    metrics_[PolicyGradientNorm] = policyGradient.norm();
    metrics_[MeanKlDivergence] = ngOutputsBuffer_[0][0];
    metrics_[CgError] = ngOutputsBuffer_[1][0];
    metrics_[NaturalGradientNorm] = naturalGradient.norm();
    metrics_[MaxGradientNorm] = maxGradNorm;
    metrics_[LineSearchBestStep] = lsInfo.first;
    metrics_[LineSearchBestIndex] = lsInfo.second;
  }
  
private:

  void configureBuffers() {
    using namespace utils;
    const auto ns = namescope();
    auto obsScope = policyFunction_->observations_scope();
    auto obsSpec = policyFunction_->observations_spec();
    auto actSpec = policyFunction_->actions_spec();
    // Training sample inputs
    trainInputsBuffer_.clear();
    trainInputsBuffer_.emplace_back(make_namescope({ns, "inputs/advantages"}));
    for (auto& spec: actSpec) { trainInputsBuffer_.emplace_back(make_namescope({ns, "inputs/actions", spec.first})); }
    for (auto& spec: obsSpec) { trainInputsBuffer_.emplace_back(make_namescope({obsScope, spec.first})); }
    // Policy gradient estimation outputs
    trainOutputsBuffer_.clear();
    trainOutputsBuffer_.emplace_back("loss/probability_ratio/prob_ratio_op");
    trainOutputsBuffer_.emplace_back("loss/entropy/mean_entropy_op");
    trainOutputsBuffer_.emplace_back("loss/surrogate/cpi_op");
    trainOutputsBuffer_.emplace_back("loss/total/loss_op");
    trainOutputsBuffer_.emplace_back("gradient/policy_gradient_op");
    for (auto& output: trainOutputsBuffer_) { output.setName(make_namescope({ns, output.name()})); }
    // Natural gradient estimation outputs
    ngInputsBuffer_.clear();
    ngInputsBuffer_.emplace_back(make_namescope({ns, "inputs/policy_gradient"}));
    for (auto& spec: actSpec) { ngInputsBuffer_.emplace_back(make_namescope({ns, "inputs/actions", spec.first})); }
    for (auto& spec: obsSpec) { ngInputsBuffer_.emplace_back(make_namescope({obsScope, spec.first})); }
    ngOutputsBuffer_.clear();
    ngOutputsBuffer_.emplace_back("loss/kl_divergence/mean_kld_op");
    ngOutputsBuffer_.emplace_back("gradient/conjugate_gradient_error_op");
    ngOutputsBuffer_.emplace_back("gradient/natural_gradient_op");
    for (auto& output: ngOutputsBuffer_) { output.setName(make_namescope({ns, output.name()})); }
    // The TRPO loss outputs used in line-search
    lossOutputsBuffer_.setName(make_namescope({ns, "loss/total/loss_op"}));
    // Configure the name of the parameters container
    policyParameters_.setName(make_namescope({ns, "policy_parameters"}));
  }
  
  inline Scalar loss(const Tensor& parameters) {
    policyFunction_->setParameters(parameters);
    graph_->run(trainInputsBuffer_, lossOutputsBuffer_);
    return lossOutputsBuffer_[0];
  }
  
  inline Scalar loss(const VectorX& parameters) {
    policyParameters_.asFlat() = parameters;
    return loss(policyParameters_);
  }

private:
  //! Defines indices for metrics collected by this class.
  enum Metric {
    MeanProbabilityRatio = 0,
    MeanEntropy,
    MeanKlDivergence,
    CpiLoss,
    TotalLoss,
    CgError,
    PolicyGradientNorm,
    NaturalGradientNorm,
    MaxGradientNorm,
    LineSearchBestStep,
    LineSearchBestIndex
  };
  //! @brief Container of metrics recorded by this class.
  Metrics metrics_;
  //! @brief Threshold of the KL-Divergence.
  hyperparam::HyperParameter<Scalar> klDivergenceThreshold_;
  //! @brief Weight of the entropy regularization loss term.
  hyperparam::HyperParameter<Scalar> entropyWeight_;
  //! @brief Damping rate for the conjugate gradient descent
  hyperparam::HyperParameter<Scalar> cgDamping_;
  //! @brief Tolerance for the conjugate gradient descent
  hyperparam::HyperParameter<Scalar> cgTolerance_;
  //! @brief Maximum number of conjugate gradient iteration.
  hyperparam::HyperParameter<int> cgMaxIterations_;
  //! @brief The line-search contraction factor (rho).
  hyperparam::HyperParameter<Scalar> lineSearchContractionFactor_;
  //! @brief Maximum number of line search iteration (k_max).
  hyperparam::HyperParameter<int> lineSearchMaxIterations_;
  //! @brief Buffer for a flattened vector of trainable parameters.
  Tensor policyParameters_;
  //! @brief The output buffer for the TRPO loss.
  Tensor lossOutputsBuffer_;
  //! @brief Buffer containing samples used for training operations.
  std::vector<Tensor> trainInputsBuffer_;
  //! @brief The output buffer for the gradients.
  std::vector<Tensor> trainOutputsBuffer_;
  //! @brief Buffer containing samples used for training operations.
  std::vector<Tensor> ngInputsBuffer_;
  //! @brief The output buffer for the gradients.
  std::vector<Tensor> ngOutputsBuffer_;
  //! @brief The policy we perform the optimization on.
  PolicyFunction* policyFunction_{nullptr};
  //! @brief The noesis::Graph instance with which to execute operations.
  Graph* graph_{nullptr};
};

} // namespace algorithm
} // namespace noesis

#endif // NOESIS_RL_ALGORITHM_POLICY_OPTIMIZATION_TRPO_HPP_

/* EOF*/
