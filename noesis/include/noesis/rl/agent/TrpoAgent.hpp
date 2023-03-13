/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_AGENT_TRPO_AGENT_HPP_
#define NOESIS_RL_AGENT_TRPO_AGENT_HPP_

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/Graph.hpp"
#include "noesis/framework/log/timer.hpp"
#include "noesis/framework/log/tensorboard.hpp"
#include "noesis/rl/function/StateValueFunction.hpp"
#include "noesis/rl/function/StochasticPolicy.hpp"
#include "noesis/rl/algorithm/policy_evaluation/value_estimation.hpp"
#include "noesis/rl/algorithm/policy_evaluation/DampedPolicyEvaluation.hpp"
#include "noesis/rl/algorithm/policy_evaluation/GeneralizedAdvantageEstimation.hpp"
#include "noesis/rl/algorithm/policy_optimization/TrustRegionPolicyOptimization.hpp"
#include "noesis/mdp/AgentInterface.hpp"

namespace noesis {
namespace agent {

template <typename ScalarType_>
class TrpoAgent final:
  public ::noesis::core::Object,
  public ::noesis::mdp::AgentInterface<ScalarType_>
{
public:
  
  // Aliases
  using Object = ::noesis::core::Object;
  using Interface = ::noesis::mdp::AgentInterface<ScalarType_>;
  using Scalar = ScalarType_;
  using Observations = typename Interface::Observations;
  using Actions = typename Interface::Actions;
  using Metrics = typename Interface::Metrics;
  using Names = typename Interface::Names;
  using Tensor = ::noesis::Tensor<Scalar>;
  using Graph = ::noesis::core::Graph;
  using Logger = ::noesis::log::TensorBoardLogger;
  using TrajectoryMemory = ::noesis::memory::TrajectoryMemory<Scalar>;
  using ValueFunction = ::noesis::function::StateValueFunction<Scalar>;
  using PolicyFunction = ::noesis::function::StochasticPolicy<Scalar>;
  using PolicyEvaluator = ::noesis::algorithm::DampedPolicyEvaluation<Scalar>;
  using PolicyOptimizer = ::noesis::algorithm::TrustRegionPolicyOptimization<Scalar>;
  using AdvantageEstimator = ::noesis::algorithm::GeneralizedAdvantageEstimation<Scalar>;
  using RandomNumberGeneratorType = ::noesis::math::RandomNumberGenerator<Scalar>;
  using Indeces = Eigen::VectorXi;
  
  /*
   * Instantiation
   */

  TrpoAgent() = delete;
  
  TrpoAgent(TrpoAgent&& other) = default;
  TrpoAgent& operator=(TrpoAgent&& other) = default;
  
  TrpoAgent(const TrpoAgent& other) = delete;
  TrpoAgent& operator=(const TrpoAgent& other) = delete;
  
  explicit TrpoAgent(
      Graph* graph,
      const std::string& name,
      const std::string& scope,
      const TensorsSpec& actions_spec,
      const TensorsSpec& observations_spec,
      const std::vector<std::string>& tasks,
      size_t batch_size,
      size_t max_trajectory_length):
    TrpoAgent(graph, nullptr, name, scope, actions_spec,
     observations_spec, tasks, batch_size, max_trajectory_length)
  {
  }
  
  explicit TrpoAgent(
      Graph* graph,
      Logger* logger,
      const std::string& name,
      const std::string& scope,
      const TensorsSpec& actions_spec,
      const TensorsSpec& observations_spec,
      const std::vector<std::string> tasks,
      size_t batch_size,
      size_t max_trajectory_length):
    Object(name, scope),
    Interface(),
    tasks_(tasks),
    timers_(name, scope),
    samplesPerIteration_(1, utils::make_namescope({scope, name, "samples_per_iteration"}), {1, std::numeric_limits<int>::max()}),
    numberOfEpochs_(1, utils::make_namescope({scope, name, "number_of_epochs"}), {1, std::numeric_limits<int>::max()}),
    logger_(logger)
  {
    NFATAL_IF(!graph, "[" << namescope() << "]: 'graph' pointer argument is invalid (nullptr).")
    const auto ns = this->namescope();
    // Construct a trajectory memory to collect sample transitions
    memory::TrajectoryMemoryConfig memConf;
    memConf.observations_spec = observations_spec;
    memConf.actions_spec = actions_spec;
    memConf.name = "TrajectoryMemory";
    memConf.scope = ns;
    memConf.number_of_tasks = tasks_.size();
    memConf.number_of_instances = batch_size;
    memConf.max_trajectory_length = max_trajectory_length;
    memConf.verbose = this->isVerbose();
    trajectoryMemory_ = std::make_unique<TrajectoryMemory>(memConf);
    // Construct the policy and state-value function interfaces
    const auto obs_scope = utils::make_namescope({ns, "Observations"});
    stochasticPolicy_ = std::make_unique<PolicyFunction>(actions_spec, observations_spec, obs_scope, ns, "Policy", graph);
    stateValueFunction_ = std::make_unique<ValueFunction>(observations_spec, obs_scope, ns, "Value", graph);
    // Construct the algorithms used for training the agent
    policyOptimizer_ = std::make_unique<PolicyOptimizer>(stochasticPolicy_.get(), ns, "PolicyOptimizer");
    policyEvaluator_ = std::make_unique<PolicyEvaluator>(stateValueFunction_.get(), ns, "PolicyEvaluator");
    advantageEstimator_ = std::make_unique<AdvantageEstimator>(ns, "AdvantageEstimator");
    // Add hyper-parameters to the global manager
    hyperparam::manager->addParameter(samplesPerIteration_);
    hyperparam::manager->addParameter(numberOfEpochs_);
  }

  ~TrpoAgent() override {
    // Remove hyper-parameters from the global manager
    hyperparam::manager->removeParameter(samplesPerIteration_);
    hyperparam::manager->removeParameter(numberOfEpochs_);
  }
  
  /*
   * Configurations
   */
  
  void setLogger(Logger* logger) {
    NFATAL_IF(!logger, "[" << namescope() << "]: 'logger' pointer argument is invalid (nullptr).")
    logger_ = logger;
  }
  
  void setDiscountFactor(Scalar gamma) {
    advantageEstimator_->setDiscountFactor(gamma);
  }
  
  /*
   * Properties
   */

  size_t batch_size() const override { return trajectoryMemory_->getNumberOfInstances(); }
  
  TensorsSpec actions_spec() const override { return stochasticPolicy_->actions_spec(); }
  
  TensorsSpec observations_spec() const override { return stochasticPolicy_->observations_spec(); }
  
  Names tasks() const override { return tasks_; }
  
  const Metrics& metrics() const override { return metrics_; }
  
  bool ready() const override {
    return (trajectoryMemory_->getTotalTransitions() >= static_cast<size_t>(samplesPerIteration_));
  }
  
  std::ostream& info(std::ostream& os) const override {
    os << "[" << namescope() << "]: " << metrics_.info();
    return os;
  }
  
  const TrajectoryMemory& memory() const { return *trajectoryMemory_; }
  
  TrajectoryMemory& memory() { return *trajectoryMemory_; }
  
  const PolicyFunction& policy() const { return *stochasticPolicy_; }
  
  PolicyFunction& policy() { return *stochasticPolicy_; }
  
  const ValueFunction& value() const { return *stateValueFunction_; }
  
  ValueFunction& value() { return *stateValueFunction_; }
  
  /*
   * Operations
   */

  void configure() override {
    const auto ns = this->namescope();
    NINFO("[" << ns << "]: Configuring agent ...")
    NINFO("[" << ns << "]: Samples-per-iter.: " << static_cast<int>(samplesPerIteration_))
    NINFO("[" << ns << "]: Number of critic epochs: " << static_cast<int>(numberOfEpochs_))
    // Configure all elements requiring post-hyper-parameter configurations
    trajectoryMemory_->configure();
    policyEvaluator_->configure();
    advantageEstimator_->configure();
    policyOptimizer_->configure();
    // Configure performance metrics
    metrics_.clear();
    metrics_.push_back("Critic/mean_value");
    metrics_.push_back("Critic/mean_terminal_value");
    metrics_.push_back("Critic/value_explained_variance");
    metrics_.push_back("Time/Agent/Memory");
    metrics_.push_back("Time/Agent/GAE");
    metrics_.push_back("Time/Agent/TRPO");
    metrics_.push_back("Time/Agent/DPE");
    metrics_.push_back("Time/Agent/Total");
    for (const auto& metric: policyEvaluator_->metrics().names()) { metrics_.push_back(metric); }
    for (const auto& metric: advantageEstimator_->metrics().names()) { metrics_.push_back(metric); }
    for (const auto& metric: policyOptimizer_->metrics().names()) { metrics_.push_back(metric); }
    // Optionally add all metrics to a logger.
    if (logger_) { metrics_.add_to(logger_); }
    // Configure duration timers
    timers_.addTimer("durations");
  }

  void initialize() override {
    NINFO("[" << this->namescope() << "]: Initializing agent ...")
    // Initialize all components
    trajectoryMemory_->reset();
    stochasticPolicy_->initialize();
    stateValueFunction_->initialize();
    advantageEstimator_->initialize();
    policyOptimizer_->initialize();
    policyEvaluator_->initialize();
  }
  
  void seed(int seed) override {
    // Nothing to be seeded in this class.
    // NOTE: Only the graph seed affects this type of agent.
  }
  
  void reset() override {
    trajectoryMemory_->reset();
  }

  void act(const Observations& observations, Actions& actions) override {
    stochasticPolicy_->mode(observations, actions);
  }

  void explore(const Observations& observations, Actions& actions) override {
    stochasticPolicy_->sample(observations, actions);
  }

  void learn() override {
    // Start performance measurement
    timers_.reset("durations");
    // Truncate terminal observations and flatten the memory to collapse all valid transitions into contiguous memory
    trajectoryMemory_->flatten();
    timers_.measure("durations", true);
    auto memTime = timers_.getElapsedTime("durations");
    // Print the current state of the memory member
    NINFO_IF(this->isVerbose(), *trajectoryMemory_);
    // Local references to parameters and samples
    const auto gamma = advantageEstimator_->discount_factor();
    const auto& terminations = trajectoryMemory_->getTerminations();
    const auto& rewards = trajectoryMemory_->getRewards();
    const auto& terminalObservations = trajectoryMemory_->getFlattenedTerminalObservations();
    auto& observations = trajectoryMemory_->getFlattenedObservations();
    auto& actions = trajectoryMemory_->getFlattenedActions();
    // Local buffers
    Tensor values("values", {1,1,1}, true);
    Tensor terminalValues("terminal_values", {1,1,1}, true);
    Tensor targetValues("target_values");
    Tensor advantages("advantages");
    // Value prediction step using current value-function
    stateValueFunction_->evaluate(observations, values);
    stateValueFunction_->evaluate(terminalObservations, terminalValues);
    // Advantage estimation step using GAE(γ,λ)
    advantageEstimator_->computeAdvantages(*trajectoryMemory_, values, terminalValues, advantages);
    timers_.measure("durations", true);
    const auto gaeTime = timers_.getElapsedTime("durations");
    // Value estimation step using TD(1)
    algorithm::td1_value_estimates(gamma, rewards, terminalValues, terminations, targetValues);
    timers_.measure("durations", true);
    const auto tdTime = timers_.getElapsedTime("durations");
    // Policy improvement step
    policyOptimizer_->train(observations, actions, advantages);
    timers_.measure("durations", true);
    auto poTime = timers_.getElapsedTime("durations");
    // Buffers for training metrics of policy evaluation.
    auto peMetrics = policyEvaluator_->metrics().zeros();
    double peTime = 0.0;
    // Perform policy evaluation using stochastic gradient descent
    auto Ne = static_cast<size_t>(numberOfEpochs_);
    for (size_t epoch = 0; epoch < Ne; ++epoch) {
      policyEvaluator_->train(observations, targetValues, values);
      peMetrics += policyEvaluator_->metrics().values();
      timers_.measure("durations", true);
      peTime += timers_.getElapsedTime("durations");
    }
    // Compute metric averages over total gradient descent iterations of policy evaluation.
    peMetrics *= 1.0/static_cast<Scalar>(Ne);
    // Post-gradient processing
    policyEvaluator_->update();
    // Record total duration of learning operations
    timers_.stop("durations");
    const auto totalTime = timers_.getTotalTime("durations");
    // Record metrics
    metrics_[Metric::CriticMeanValue] = math::cwise_mean(values);
    metrics_[Metric::CriticMeanTerminalValue] = math::cwise_mean(terminalValues);
    metrics_[Metric::CriticValueExplainedVariance] = math::cwise_explained_variance(values, targetValues);
    metrics_[Metric::TrajectoryMemoryTime] = memTime;
    metrics_[Metric::AdvantageEstimatorTime] = gaeTime;
    metrics_[Metric::PolicyOptimizerTime] = poTime;
    metrics_[Metric::PolicyEvaluatorTime] = peTime;
    metrics_[Metric::TotalTime] = totalTime;
    const auto Ndpe = policyEvaluator_->metrics().size();
    const auto Ngae = advantageEstimator_->metrics().size();
    const auto Ntrpo = policyOptimizer_->metrics().size();
    for (size_t i = 0; i < Ndpe; ++i) { metrics_[i+8] = peMetrics(i); }
    for (size_t i = 0; i < Ngae; ++i) { metrics_[i+8+Ndpe] = advantageEstimator_->metrics()[i]; }
    for (size_t i = 0; i < Ntrpo; ++i) { metrics_[i+8+Ndpe+Ngae] = policyOptimizer_->metrics()[i]; }
    // Optionally append all metrics to a logger.
    if (logger_) { metrics_.append_to(logger_); }
  }
  
private:
  //! Defines indices for metrics collected by this class.
  enum Metric {
    CriticMeanValue = 0,
    CriticMeanTerminalValue,
    CriticValueExplainedVariance,
    TrajectoryMemoryTime,
    AdvantageEstimatorTime,
    PolicyOptimizerTime,
    PolicyEvaluatorTime,
    TotalTime
  };
  Metrics metrics_;
  Names tasks_;
  log::MultiTimer timers_;
  hyperparam::HyperParameter<int> samplesPerIteration_;
  hyperparam::HyperParameter<int> numberOfEpochs_;
  std::unique_ptr<TrajectoryMemory> trajectoryMemory_{nullptr};
  std::unique_ptr<PolicyFunction> stochasticPolicy_{nullptr};
  std::unique_ptr<ValueFunction> stateValueFunction_{nullptr};
  std::unique_ptr<PolicyOptimizer> policyOptimizer_{nullptr};
  std::unique_ptr<AdvantageEstimator> advantageEstimator_{nullptr};
  std::unique_ptr<PolicyEvaluator> policyEvaluator_{nullptr};
  Logger* logger_{nullptr};
};

} // namespace agent
} // namespace noesis

#endif // NOESIS_RL_AGENT_TRPO_AGENT_HPP_

/* EOF */
