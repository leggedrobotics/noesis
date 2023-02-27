/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_AGENT_PPO_AGENT_HPP_
#define NOESIS_RL_AGENT_PPO_AGENT_HPP_

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/Graph.hpp"
#include "noesis/framework/log/timer.hpp"
#include "noesis/framework/log/tensorboard.hpp"
#include "noesis/rl/function/StateValueFunction.hpp"
#include "noesis/rl/function/StochasticPolicy.hpp"
#include "noesis/rl/algorithm/policy_evaluation/value_estimation.hpp"
#include "noesis/rl/algorithm/policy_evaluation/ClippedPolicyEvaluation.hpp"
#include "noesis/rl/algorithm/policy_evaluation/GeneralizedAdvantageEstimation.hpp"
#include "noesis/rl/algorithm/policy_optimization/ProximalPolicyOptimization.hpp"
#include "noesis/mdp/AgentInterface.hpp"

namespace noesis {
namespace agent {

template <typename ScalarType_>
class PpoAgent final:
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
  using PolicyEvaluator = ::noesis::algorithm::ClippedPolicyEvaluation<Scalar>;
  using PolicyOptimizer = ::noesis::algorithm::ProximalPolicyOptimization<Scalar>;
  using AdvantageEstimator = ::noesis::algorithm::GeneralizedAdvantageEstimation<Scalar>;
  using RandomNumberGeneratorType = ::noesis::math::RandomNumberGenerator<Scalar>;
  using Indeces = Eigen::VectorXi;
  
  /*
   * Instantiation
   */

  PpoAgent() = delete;
  
  PpoAgent(PpoAgent&& other) = default;
  PpoAgent& operator=(PpoAgent&& other) = default;
  
  PpoAgent(const PpoAgent& other) = delete;
  PpoAgent& operator=(const PpoAgent& other) = delete;
  
  explicit PpoAgent(
      Graph* graph,
      const std::string& name,
      const std::string& scope,
      const TensorsSpec& actions_spec,
      const TensorsSpec& observations_spec,
      const std::vector<std::string>& tasks,
      size_t batch_size,
      size_t max_trajectory_length):
    PpoAgent(graph, nullptr, name, scope, actions_spec,
     observations_spec, tasks, batch_size, max_trajectory_length)
  {
  }
  
  explicit PpoAgent(
      Graph* graph,
      Logger* logger,
      const std::string& name,
      const std::string& scope,
      const TensorsSpec& actions_spec,
      const TensorsSpec& observations_spec,
      const std::vector<std::string>& tasks,
      size_t batch_size,
      size_t max_trajectory_length):
    Object(name, scope),
    Interface(),
    mbObservations_("minibatch/observations", 1, 1),
    mbActions_("minibatch/actions", 1, 1),
    mbValues_("minibatch/values", {1,1,1}, true),
    mbTargetValues_("minibatch/target_values", {1,1,1}, true),
    mbAdvantages_("minibatch/advantages", {1,1,1}, true),
    mbLogProbabilities_("minibatch/actions_log_probabilities", {1, 1, 1}, true),
    tasks_(tasks),
    timers_(name, scope),
    samplesPerIteration_(1, utils::make_namescope({scope, name, "samples_per_iteration"}), {1, std::numeric_limits<int>::max()}),
    numberOfEpochs_(1, utils::make_namescope({scope, name, "number_of_epochs"}), {1, std::numeric_limits<int>::max()}),
    numberOfMinibatches_(1, utils::make_namescope({scope, name, "number_of_minibatches"}), {1, std::numeric_limits<int>::max()}),
    shuffleMinibatches_(false, utils::make_namescope({scope, name, "shuffle_minibatches"})),
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
    hyperparam::manager->addParameter(numberOfMinibatches_);
    hyperparam::manager->addParameter(shuffleMinibatches_);
  }

  ~PpoAgent() override {
    // Remove hyper-parameters from the global manager
    hyperparam::manager->removeParameter(samplesPerIteration_);
    hyperparam::manager->removeParameter(numberOfEpochs_);
    hyperparam::manager->removeParameter(numberOfMinibatches_);
    hyperparam::manager->removeParameter(shuffleMinibatches_);
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
    // We first retrieve the sizes determining the training data.
    const auto batch_size = this->batch_size();
    const auto minibatches = static_cast<size_t>(numberOfMinibatches_);
    auto dataset_size = static_cast<size_t>(samplesPerIteration_);
    // NOTE: We want to ensure that we have samples which are integer multiples
    // of the mini-batches and agent's batch-size: S = D*M*B.
    // If necessary we adjust D to the nearest integer value.
    const auto B = static_cast<double>(batch_size);
    const auto M = static_cast<double>(minibatches);
    auto S = static_cast<double>(dataset_size);
    auto D = S/(M*B);
    if (std::ceil(D) - D > 0) {
      NERROR("[" << ns << "]: Samples-per-iter is not an integer multiple of mini-batches and batch-size!")
      NERROR("[" << ns << "]: Adjusting samples-per-iter to nearest integer ...")
      NERROR("[" << ns << "]: Old: " << dataset_size);
      dataset_size = static_cast<size_t>(std::ceil(D)*M*B);
      samplesPerIteration_ = dataset_size;
      NERROR("[" << ns << "]: New: " << dataset_size);
    }
    // We finally configure the size of each mini-bach and and report the sizes of the training data.
    minibatchSize_ = dataset_size/minibatches;
    NINFO("[" << ns << "]: Samples-per-iter.: " << static_cast<int>(samplesPerIteration_))
    NINFO("[" << ns << "]: Mini-batch: Segments: " << static_cast<int>(numberOfMinibatches_))
    NINFO("[" << ns << "]: Mini-batch: Size: " << minibatchSize_);
    NINFO("[" << ns << "]: Mini-batch: Shuffling: " << std::boolalpha << static_cast<bool>(shuffleMinibatches_))
    // Retrieve the MDP specifications
    auto observationsSpec = this->observations_spec();
    auto actionsSpec = this->actions_spec();
    auto tasks = this->tasks();
    // Configure mini-batch sample buffers
    mbObservations_.setFromSpec(observationsSpec);
    mbActions_.setFromSpec(actionsSpec);
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
    metrics_.push_back("Time/Agent/PPO");
    metrics_.push_back("Time/Agent/CPE");
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
    // Allocate mini-batch buffers
    mbObservations_.resize(minibatchSize_, 1);
    mbActions_.resize(minibatchSize_, 1);
    mbValues_.resize({1, minibatchSize_, 1}, true);
    mbTargetValues_.resize({1, minibatchSize_, 1}, true);
    mbAdvantages_.resize({1, minibatchSize_, 1}, true);
    mbLogProbabilities_.resize({1, minibatchSize_, 1}, true);
    // Initialize the data indices used for mini-batch shuffling
    indices_ = Eigen::VectorXi::LinSpaced(static_cast<size_t>(samplesPerIteration_), 0, static_cast<int>(samplesPerIteration_));
  }
  
  void seed(int seed) override {
    NINFO_IF(this->isVerbose(), "[" << this->namescope() << "]: Setting new PRNG seed: " << seed)
    generator_.seed(seed);
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
    Tensor logprobs("logprobs");
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
    // Buffers for training metrics
    auto poMetrics = policyOptimizer_->metrics().zeros();
    auto peMetrics = policyEvaluator_->metrics().zeros();
    double poTime = 0.0;
    double peTime = 0.0;
    // Compute log-prob values for the collected action samples using the current (old) policy
    policyOptimizer_->logprobs(observations, actions, logprobs);
    // Perform policy evaluation and improvement using mini-batch stochastic gradient descent
    auto Ne = static_cast<size_t>(numberOfEpochs_);
    auto Nmb = static_cast<size_t>(numberOfMinibatches_);
    for (size_t epoch = 0; epoch < Ne; ++epoch) {
      if (shuffleMinibatches_) {
        // Create new random shuffling of indices
        std::shuffle(indices_.data(), indices_.data() + indices_.size(), generator_.generator());
        observations.shuffle(indices_);
        actions.shuffle(indices_);
        logprobs.shuffle(indices_);
        values.shuffle(indices_);
        targetValues.shuffle(indices_);
        advantages.shuffle(indices_);
      }
      // Iterate over minibatches and apply gradient updates
      for (size_t mb = 0; mb < Nmb; ++mb) {
        // Compute current mini-batch block index range
        size_t mbStart = mb * minibatchSize_;
        // Copy mini-batch block data
        for (size_t k = 0; k < observations.size(); k++) {
          mbObservations_[k].asEigenMatrix() = observations[k].block(mbStart, minibatchSize_).asEigenMatrix();
        }
        for (size_t k = 0; k < actions.size(); k++) {
          mbActions_[k].asEigenMatrix() = actions[k].block(mbStart, minibatchSize_).asEigenMatrix();
        }
        mbValues_.asEigenMatrix() = values.block(mbStart, minibatchSize_).asEigenMatrix();
        mbTargetValues_.asEigenMatrix() = targetValues.block(mbStart, minibatchSize_).asEigenMatrix();
        mbAdvantages_.asEigenMatrix() = advantages.block(mbStart, minibatchSize_).asEigenMatrix();
        mbLogProbabilities_.asEigenMatrix() = logprobs.block(mbStart, minibatchSize_).asEigenMatrix();
        // Policy improvement step using PPO
        policyOptimizer_->train(mbObservations_, mbActions_, mbAdvantages_, mbLogProbabilities_);
        poMetrics += policyOptimizer_->metrics().values();
        timers_.measure("durations", true);
        poTime += timers_.getElapsedTime("durations");
        // Policy evaluation step using SL w/ clipped loss
        policyEvaluator_->train(mbObservations_, mbTargetValues_, mbValues_);
        peMetrics += policyEvaluator_->metrics().values();
        timers_.measure("durations", true);
        peTime += timers_.getElapsedTime("durations");
      }
    }
    // Compute metric averages over total gradient descent iterations of policy evaluation and optimization.
    poMetrics *= 1.0/static_cast<Scalar>(Ne*Nmb);
    peMetrics *= 1.0/static_cast<Scalar>(Ne*Nmb);
    // Post-gradient processing
    policyOptimizer_->update();
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
    const auto Ncpe = policyEvaluator_->metrics().size();
    const auto Ngae = advantageEstimator_->metrics().size();
    const auto Nppo = policyOptimizer_->metrics().size();
    for (size_t i = 0; i < Ncpe; ++i) { metrics_[i+8] = peMetrics(i); }
    for (size_t i = 0; i < Ngae; ++i) { metrics_[i+8+Ncpe] = advantageEstimator_->metrics()[i]; }
    for (size_t i = 0; i < Nppo; ++i) { metrics_[i+8+Ncpe+Ngae] = poMetrics(static_cast<int>(i)); }
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
  Observations mbObservations_;
  Actions mbActions_;
  Tensor mbValues_;
  Tensor mbTargetValues_;
  Tensor mbAdvantages_;
  Tensor mbLogProbabilities_;
  Indeces indices_;
  size_t minibatchSize_{1};
  RandomNumberGeneratorType generator_;
  Metrics metrics_;
  Names tasks_;
  log::MultiTimer timers_;
  hyperparam::HyperParameter<int> samplesPerIteration_;
  hyperparam::HyperParameter<int> numberOfEpochs_;
  hyperparam::HyperParameter<int> numberOfMinibatches_;
  hyperparam::HyperParameter<bool> shuffleMinibatches_;
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

#endif // NOESIS_RL_AGENT_PPO_AGENT_HPP_

/* EOF */
