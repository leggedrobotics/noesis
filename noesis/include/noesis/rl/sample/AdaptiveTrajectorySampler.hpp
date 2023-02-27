/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_SAMPLE_ADAPTIVE_TRAJECTORY_SAMPLER_HPP_
#define NOESIS_RL_SAMPLE_ADAPTIVE_TRAJECTORY_SAMPLER_HPP_

// Noesis
#include "noesis/framework/system/time.hpp"
#include "noesis/mdp/AgentInterface.hpp"
#include "noesis/gym/core/Vector.hpp"
#include "noesis/gym/train/SamplerInterface.hpp"
#include "noesis/rl/memory/TrajectoryMemory.hpp"

namespace noesis {
namespace algorithm {

template <typename ScalarType_>
class AdaptiveTrajectorySampler final:
  public ::noesis::core::Object,
  public ::noesis::gym::SamplerInterface<ScalarType_>
{
public:
  
  // Aliases
  using Object = ::noesis::core::Object;
  using Interface = ::noesis::gym::SamplerInterface<ScalarType_>;
  using Scalar = typename Interface::Scalar;
  using Metrics = typename Interface::Metrics;
  using Logger = ::noesis::log::TensorBoardLogger;
  using Termination = ::noesis::mdp::Termination<Scalar>;
  using Environment = ::noesis::gym::Vector<Scalar>;
  using Agent = ::noesis::mdp::AgentInterface<Scalar>;
  using Memory = ::noesis::memory::TrajectoryMemory<Scalar>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  
  /*
   * Instantiation
   */
  
  //! @note Default construction is disabled for this class to ensure validity of instances.
  AdaptiveTrajectorySampler() = delete;
  
  //! @note Move and move-assignment construction is permissible for this class.
  AdaptiveTrajectorySampler(AdaptiveTrajectorySampler&& other) noexcept = default;
  AdaptiveTrajectorySampler& operator=(AdaptiveTrajectorySampler&& other) noexcept = default;
  
  //! @note Copy and copy-assignment construction is permissible for this class.
  AdaptiveTrajectorySampler(const AdaptiveTrajectorySampler& other) = default;
  AdaptiveTrajectorySampler& operator=(const AdaptiveTrajectorySampler& other) = default;
  
  explicit AdaptiveTrajectorySampler(
      Agent* agent,
      Memory* memory,
      Environment* environment,
      Logger* logger=nullptr,
      const std::string& name="Sampler",
      const std::string& scope="/",
      bool verbose=false):
    Object(name, scope, verbose),
    Interface(),
    agent_(agent),
    memory_(memory),
    environment_(environment),
    stopWhenReady_(false, utils::make_namescope({scope, name, "preempt_sampling"})),
    logger_(logger)
  {
    const auto ns = namescope();
    NFATAL_IF(!agent_, "[" << ns << "]: 'agent' argument is invalid (nullptr).");
    NFATAL_IF(!memory_, "[" << ns << "]: 'agent' argument is invalid (nullptr).");
    NFATAL_IF(!environment_, "[" << ns << "]: 'environment' argument is invalid (nullptr).");
    NWARNING_IF(isVerbose() && !logger_, "[" << ns << "]: 'logger' argument is not set.");
    // We add the local HPs to the global manager and will only remove them in the destructor.
    hyperparam::manager->addParameter(stopWhenReady_);
  }
  
  ~AdaptiveTrajectorySampler() final {
    // Remove hyper-parameters from the global manager
    // NOTE: This is necessary in order to ensure that
    // the hyper-parameter is not accidentally dangling.
    hyperparam::manager->removeParameter(stopWhenReady_);
  }
  
  /*
   * Configurations
   */
  
  void setAgent(Agent* agent) {
    NFATAL_IF(!agent, "[" << namescope() << "]: 'agent' pointer argument is invalid (nullptr).")
    agent_ = agent;
  }
  
  void setMemory(Memory* memory) {
    NFATAL_IF(!memory, "[" << namescope() << "]: 'memory' pointer argument is invalid (nullptr).")
    memory_ = memory;
  }
  
  void setEnvironment(Environment* environment) {
    NFATAL_IF(!environment, "[" << namescope() << "]: 'environment' pointer argument is invalid (nullptr).")
    environment_ = environment;
  }
  
  void setLogger(Logger* logger) {
    NFATAL_IF(!logger, "[" << namescope() << "]: 'logger' pointer argument is invalid (nullptr).")
    logger_ = logger;
  }
  
  /*
   * Properties
   */

  const std::vector<size_t>& getSampleCounters() const override { return transitionCounters_; }
  
  const std::vector<size_t>& getBatchCounters() const override { return episodeCounters_; }
  
  const std::vector<size_t>& getErrorCounters() const { return errorCounters_; }
  
  size_t getTotalSamples() const override { return totalTransitions_; }
  
  size_t getTotalBatches() const override { return totalEpisodes_; }
  
  const Metrics& metrics() const override { return metrics_; }
  
  std::string info() const override {
    std::stringstream out;
    const auto name = this->name();
    out << "\n  Mean Step Reward    : " << metrics_[name + "/agent/mean_step_reward"];
    out << "\n  Mean Episode Reward : " << metrics_[name + "/agent/mean_episode_reward"];
    out << "\n  Mean Episode Length : " << metrics_[name + "/agent/mean_episode_length"];
    out << "\n  Mean Episode Return : " << metrics_[name + "/agent/mean_episode_return"];
    out << "\n  Mean Terminal Value : " << metrics_[name + "/agent/mean_terminal_value"];
    out << "\n  Samples-per-sec     : " << metrics_[name + "/agent/samples_per_second"];
    out << "\n  Samples-per-iter    : " << metrics_[name + "/agent/samples_per_iteration"];
    out << "\n  Episodes-per-iter   : " << metrics_[name + "/agent/episodes_per_iteration"];
    out << "\n  Samples             : " << metrics_[name + "/agent/total_samples"];
    out << "\n  Episodes            : " << metrics_[name + "/agent/total_episodes"];
    out << "\n  Errors              : " << metrics_[name + "/agent/errors"];
    return out.str();
  }
  
  /*
   * Operations
   */
  
  /*!
   * @brief Configures the sampler according to the specifications of
   *        the agent and environment attached to the sampler.
   */
  void configure() override {
    const auto name = this->name();
    const auto ns = this->namescope();
    NFATAL_IF(agent_->batch_size() != environment_->batch_size(),
      "[" << ns << "]: Environment and agent batch sizes do not match!")
    NINFO("[" << ns << "]: Configuring sampler ...")
    NINFO("[" << ns << "]: Preempting sampling: " << std::boolalpha << static_cast<bool>(stopWhenReady_))
    // Define metrics collected by the sampler.
    metrics_.clear();
    metrics_.push_back(name + "/agent/mean_step_reward");
    metrics_.push_back(name + "/agent/mean_episode_reward");
    metrics_.push_back(name + "/agent/mean_episode_length");
    metrics_.push_back(name + "/agent/mean_episode_return");
    metrics_.push_back(name + "/agent/mean_terminal_value");
    metrics_.push_back(name + "/agent/samples_per_second");
    metrics_.push_back(name + "/agent/samples_per_iteration");
    metrics_.push_back(name + "/agent/episodes_per_iteration");
    metrics_.push_back(name + "/agent/total_samples");
    metrics_.push_back(name + "/agent/total_episodes");
    metrics_.push_back(name + "/agent/errors");
    // Append environment-specific metrics.
    for (const auto& metric: environment_->metrics().names()) {
      metrics_.push_back(name + "/metric/terminal/" + metric);
      metrics_.push_back(name + "/metric/step/" + metric);
    }
    // NOTE: We skip the first task because it's the total and that
    // is already recorded in the default sampler metrics (see above).
    for (size_t t = 1; t < environment_->tasks().size(); t++) {
      metrics_.push_back(name + "/task/" + environment_->tasks()[t]);
    }
    // Add all metrics to the logger (if set).
    if (logger_) {
      metrics_.add_to(logger_);
      // Add the termination id to the logger
      logger_->addLoggingSignal(name + "/terminations", 1);
    }
    // NOTE: We pre-allocate all recording containers to ensure efficient
    // operation at run-time
    const auto Ne = environment_->batch_size();
    episodeTerminalMetrics_.resize(Ne);
    episodeTerminalValues_.resize(Ne);
    episodeStepMetrics_.resize(Ne);
    episodeRewards_.resize(Ne);
    episodeLengths_.resize(Ne);
    episodeTerminations_.resize(Ne);
    episodeCounters_.resize(Ne, 0);
    transitionCounters_.resize(Ne, 0);
    errorCounters_.resize(Ne, 0);
    // TODO: Compute smaller capacity to minimize footprint
    const size_t capacity = memory_->getTrajectoryCapacity();
    for (size_t e = 0; e < Ne; ++e) {
      episodeTerminalMetrics_[e].reserve(capacity);
      episodeTerminalValues_[e].reserve(capacity);
      episodeStepMetrics_[e].reserve(capacity);
      episodeRewards_[e].reserve(capacity);
      episodeLengths_[e].reserve(capacity);
      episodeTerminations_[e].reserve(capacity);
    }
    // NOTE: We initialize all recording containers to ensure
    // that they are in a proper state after each (re)configuration.
    totalEpisodes_ = 0;
    totalTransitions_ = 0;
    for (size_t e = 0; e < environment_->batch_size(); e++) {
      episodeTerminalMetrics_[e].clear();
      episodeTerminalValues_[e].clear();
      episodeStepMetrics_[e].clear();
      episodeRewards_[e].clear();
      episodeLengths_[e].clear();
      episodeTerminations_[e].clear();
      episodeCounters_[e] = 0;
      transitionCounters_[e] = 0;
      errorCounters_[e] = 0;
    }
  }
  
  /*!
   * @brief Resets the agent and environment attached to the sampler, as
   *        well as all internal counters, episode recordings, and metrics.
   */
  void reset() override {
    timer_.reset();
    agent_->reset();
    environment_->reset();
    const auto Ne = environment_->batch_size();
    for (size_t e = 0; e < Ne; e++) {
      auto& env = (*environment_)[e];
      const auto& obs = env.observations();
      memory_->initializeTrajectory(e, obs);
      episodeRewards_[e].clear();
      episodeRewards_[e].push_back(Vector::Zero(environment_->tasks().size()));
      episodeStepMetrics_[e].clear();
      episodeStepMetrics_[e].push_back(Vector::Zero(environment_->metrics().size()));
      episodeTerminalMetrics_[e].clear();
      episodeTerminalMetrics_[e].push_back(Vector::Zero(environment_->metrics().size()));
      episodeTerminalValues_[e].clear();
      episodeTerminalValues_[e].push_back(0);
      episodeLengths_[e].clear();
      episodeLengths_[e].push_back(0);
      episodeTerminations_[e].clear();
      episodeTerminations_[e].push_back(0);
      episodeCounters_[e] = 0;
      transitionCounters_[e] = 0;
      errorCounters_[e] = 0;
    }
  }
  
  /*!
   * @brief Samples actions and observations from the agent and environment respectively.
   * @note This function should be called step-wise, equivalently to the environments `step()`.
   * @return Returns true when sampling should terminate and sample processing should commence.
   */
  bool sample() override {
    // Step 1: We resize the observation-action buffers
    // according to the number of active instances.
    // NOTE: This prevents in-graph operations from
    // performing wasteful computations for inactive instances.
    environment_->resize_buffers();
    // Step 2: We collect observations from each individual
    // environment into the batched observations buffer.
    environment_->collect_observations();
    // Step 3: Generate  action samples given observations
    // from the previous transition
    agent_->explore(environment_->observations(), environment_->actions());
    // Step 4: We distribute actions from the batched buffer
    // to the individual instances.
    environment_->distribute_actions();
    // Step 5: Generate transition samples given the
    // actions generated by the agent.
    // NOTE: In contrast to the standard `step()` operation,
    // this function only steps the environment and does not
    // perform any sample aggregation. This is why we have used
    // he `collect_observations` and `distribute_actions()` ops.
    environment_->step_environments();
    // We buffer the HP value here since we use it in multiple places below.
    // NOTE: This performs a casting operation from the HP to the underlying value.
    bool stopWhenReady = stopWhenReady_;
    // Step 6: After samples have been generated in the environment's `step()` call,
    // we need to first collect the data of the transition before any other
    // operation so that the total sample count is up-to-date.
    // TODO: (@vt): can this also be parallelized?
    const auto Ne = environment_->batch_size();
    for (size_t e = 0; e < Ne; e++) {
      // NOTE: We skip inactive environment instances.
      if (!environment_->isActive(e)) { continue; }
      // Retrieve instance-specific references
      auto& env = (*environment_)[e];
      const auto& act = env.actions();
      const auto& obs = env.observations();
      const auto& rew = env.rewards();
      const auto& trm = env.terminations().back();
      const auto& met = env.metrics();
      // NOTE: We identify invalid samples (if data has NaN or Inf) when
      // the termination type corresponds to negative enum value.
      if (static_cast<int>(trm.type) >= 0) {
        transitionCounters_[e]++;
        memory_->addTransition(e, act, obs, rew);
        episodeRewards_[e].back() += rew.asFlat(); // NOTE: This assumes there is no history
        episodeStepMetrics_[e].back() += met.values();
      }
    }
    // Step 7: Since transitions were added above, we can now check if the target
    // agent's criteria for terminating sampling are met.
    bool isReady = agent_->ready();
    // Step 8: We proceed to perform episode termination checks for this round transitions.
    // TODO: (@vt): can this also be parallelized?
    for (size_t e = 0; e < Ne; e++) {
      // NOTE: We skip inactive environment instances.
      if (!environment_->isActive(e)) { continue; }
      // Retrieve instance-specific references
      auto& env = (*environment_)[e];
      const auto& obs = env.observations();
      const auto& trm = env.terminations().back();
      const auto& met = env.metrics();
      // NOTE: Process the transition only if valid otherwise we discard all
      // relevant samples and restart the trajectory.
      if (static_cast<int>(trm.type) >= 0) {
        // Step 9: We terminate trajectories when they time-out or a terminal state
        // or also when we want to follow strict adherence to the agent's termination criteria.
        // NOTE: If preemption is disabled (stopWhenReady:=false), then sampling stops only when
        // the agent is ready and all environments have reached termination on their own.
        if ( trm.type != Termination::Type::Unterminated || (isReady && stopWhenReady) ) {
          // NOTE: We store terminal experiences separately so that
          // all terminal observations are contiguous in memory. This
          // allows policy evaluation to be performed directly on the
          // terminal observations without re-sifting through the memory.
          memory_->terminateTrajectory(e, obs, trm);
          episodeCounters_[e]++;
          episodeTerminalMetrics_[e].back() = met.values();
          episodeTerminalValues_[e].back() = trm.value;
          episodeLengths_[e].back() = transitionCounters_[e];
          episodeTerminations_[e].back() = trm.id;
          NWARNING_IF(isReady && isVerbose(), "[" << namescope() << "]: Environment terminated:"
            << "\n  with Length: " << transitionCounters_[e]
            << "\n  at Instance: " << e)
          // Step 10: Start a new episode only if the agent has not signalled that it has
          // enough data to perform training. Otherwise deactivate the environment instance.
          if (!isReady) {
            env.reset();
            memory_->initializeTrajectory(e, obs);
            episodeTerminalMetrics_[e].push_back(Vector::Zero(environment_->metrics().size()));
            episodeStepMetrics_[e].push_back(Vector::Zero(environment_->metrics().size()));
            episodeRewards_[e].push_back(Vector::Zero(environment_->tasks().size()));
            episodeTerminalValues_[e].push_back(0);
            episodeLengths_[e].push_back(0);
            episodeTerminations_[e].push_back(0);
            transitionCounters_[e] = 0;
          } else {
            environment_->deactivate(e);
            NWARNING_IF(isVerbose(), "[" << namescope() << "]: Environment '" << e << "' is deactivating:"
              << "\n  while " << environment_->active() << " others are still active, and"
              << "\n  with Samples: " << memory_->getTotalTransitions()
              << "\n  and Episodes: " << memory_->getNumberOfTrajectories())
          }
        }
      } else {
        NERROR("[" << namescope() << "]: Environment '" << e << "': Episode has invalid data!\n" << env)
        transitionCounters_[e] -= env.steps();
        episodeStepMetrics_[e].back().setZero();
        episodeRewards_[e].back().setZero();
        env.reset();
        memory_->restartTrajectory(e, obs);
        errorCounters_[e]++;
      }
    }
    NWARNING_IF(isReady && isVerbose(), "[" << namescope() << "]: Agent is ready, while:"
      << "\n Active: " << environment_->active()
      << "\n Samples: " << memory_->getTotalTransitions()
      << "\n Episodes: " << memory_->getNumberOfTrajectories())
    // Step 11: We terminate sampling ONLY when all environments have been deactivated.
    return (environment_->active() == 0);
  }
  
  /*!
   * @brief Processes the samples collected from the latest round of sampling.
   * @note This method should **only** be called after the `sample()` returns `true`.
   */
  void process() override {
    // Step 1: First we process all meta-data recorded for the current sampling session.
    record();
    // Step 2: Process the sample trajectories collected and stored in the agent's memory.
    // NOTE: The actual type of processing is up to the agent and whatever it's interpretation
    // (and implementation) of whatever "learning" actually is.
    agent_->learn();
  }
  
private:
  
  void record() {
    // Capture elapsed time at the point where sampling ended.
    const auto elapsed = timer_.elapsed().toSeconds();
    // Useful constants
    const auto name = this->name();
    const auto ns = this->namescope();
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const size_t Nm = environment_->metrics().size();
    const size_t Nt = environment_->tasks().size();
    // Define accumulators
    Vector term_metrics_sum = Vector::Zero(Nm);
    Vector step_metrics_sum = Vector::Zero(Nm);
    Vector rewards_sum = Vector::Zero(Nt);
    Scalar terminal_value_sum = 0;
    Scalar number_of_episodes = 0;
    Scalar lengths_sum = 0;
    Scalar errors_sum = 0;
    // Compute sample statistics for episode rewards, returns and lengths
    const auto Ne = episodeRewards_.size();
    for (size_t e = 0; e < Ne; ++e) {
      errors_sum += static_cast<Scalar>(errorCounters_[e]);
      number_of_episodes += static_cast<Scalar>(episodeLengths_[e].size());
      const auto Nep = episodeRewards_[e].size();
      for (size_t episode = 0; episode < Nep; ++episode) {
        term_metrics_sum += episodeTerminalMetrics_[e][episode];
        step_metrics_sum += episodeStepMetrics_[e][episode];
        lengths_sum += static_cast<Scalar>(episodeLengths_[e][episode]);
        rewards_sum += episodeRewards_[e][episode];
        terminal_value_sum +=episodeTerminalValues_[e][episode];
      }
    }
    totalEpisodes_ += number_of_episodes;
    totalTransitions_ += lengths_sum;
    // Update the sampler's metrics using current episode statistics
    // NOTE: The first element of the rewards vector holds the TOTAL reward.
    metrics_[name + "/agent/mean_step_reward"] = rewards_sum(0)/(lengths_sum + eps);
    metrics_[name + "/agent/mean_episode_reward"] = rewards_sum(0)/(number_of_episodes + eps);
    metrics_[name + "/agent/mean_episode_length"] = lengths_sum/(number_of_episodes + eps);
    metrics_[name + "/agent/mean_episode_return"] = (rewards_sum(0) + terminal_value_sum)/(number_of_episodes + eps);
    metrics_[name + "/agent/mean_terminal_value"] = (terminal_value_sum)/(number_of_episodes + eps);
    metrics_[name + "/agent/samples_per_second"] = std::max(0.0, lengths_sum/(elapsed + eps));
    metrics_[name + "/agent/samples_per_iteration"] = lengths_sum;
    metrics_[name + "/agent/episodes_per_iteration"] = number_of_episodes;
    metrics_[name + "/agent/total_samples"] = totalTransitions_;
    metrics_[name + "/agent/total_episodes"] = totalEpisodes_;
    metrics_[name + "/agent/errors"] = errors_sum;
    NWARNING_IF(isVerbose(), "[" << ns << "]: Episodes: " << number_of_episodes)
    NWARNING_IF(isVerbose(), "[" << ns << "]: Samples: " << lengths_sum)
    // Environment-specific metrics
    // NOTE: We define length of default metrics Nd (see right above).
    const size_t Nd = metrics_.size() - 2*Nm - Nt + 1;
    for (size_t m = 0; m < Nm; ++m) {
      metrics_[Nd + 2*m] = term_metrics_sum(m) / (number_of_episodes + eps);
      metrics_[Nd + 2*m + 1] = step_metrics_sum(m) / (lengths_sum + eps);
    }
    for (size_t t = 0; t < Nt - 1; ++t) {
      metrics_[Nd + 2*Nm + t] = rewards_sum(t + 1) / (lengths_sum + eps);
    }
    // Optionally append all metrics to a logger.
    if (logger_) {
      metrics_.append_to(logger_);
      std::vector<int> terminations;
      for (const auto& terms: episodeTerminations_) {
        terminations.insert(terminations.end(), terms.begin(), terms.end());
      }
      logger_->appendHistogram(name + "/terminations", terminations);
    }
  }
  
private:
  //! @brief Container of metrics recorded by this class.
  //! @note Metrics are updated only after the end of each round of sampling,
  //!       which is why this happens in the `train()` operation.
  Metrics metrics_;
  //! @brief Collection of recorded episode sample rewards.
  //! @note This collection is of episodes already terminated.
  std::vector<std::vector<Vector>> episodeRewards_;
  //! @brief Collection of recorded episode metrics.
  //! @note This collection is of episodes already terminated.
  std::vector<std::vector<Vector>> episodeStepMetrics_;
  //! @brief Collection of recorded episode metrics.
  //! @note This collection is of episodes already terminated.
  std::vector<std::vector<Vector>> episodeTerminalMetrics_;
  //! @brief Collection of recorded episodes terminal values.
  //! @note This collection is of episodes already terminated.
  std::vector<std::vector<Scalar>> episodeTerminalValues_;
  //! @brief Collection of recorded episodes lengths.
  //! @note This collection is of episodes already terminated.
  std::vector<std::vector<size_t>> episodeLengths_;
  //! @brief Collection of recorded episode termination identifiers (info).
  //! @note This collection is of episodes already terminated.
  std::vector<std::vector<int>> episodeTerminations_;
  //! @brief Counters of transitions currently executed by each environment instance.
  std::vector<size_t> transitionCounters_;
  //! @brief Counter of episodes currently executed each environment instance.
  std::vector<size_t> episodeCounters_;
  //! @brief Counters of sample errors currently recorded from each environment instance.
  std::vector<size_t> errorCounters_;
  //! @brief Counter of the total number of generated environment episodes/trajectories.
  size_t totalEpisodes_{0};
  //! @brief Counter of the total number of generated environment steps/transitions.
  size_t totalTransitions_{0};
  //! @brief Timer used to measure the duration of sampling operations.
  Time timer_;
  //! @brief Hyper-parameter optionally enabling sampling preemption.
  //! @note Preemption means that we terminate the sampling operation immediately
  //!       once the agent's criteria is fulfilled. Otherwise we permit currently
  //!       active episodes to proceed until timing-out or reaching a terminal state.
  hyperparam::HyperParameter<bool> stopWhenReady_;
  //! @brief Pointer to the agent from which to sample actions.
  Agent* agent_{nullptr};
  //! @brief Pointer to the agent from which to sample actions.
  Memory* memory_{nullptr};
  //! @brief Pointer to the environment (vector) from which to sample observations.
  Environment* environment_{nullptr};
  //! @brief The TensorBoard logger used by all modules to record logging signals.
  Logger* logger_{nullptr};
};

} // namespace algorithm
} // namespace noesis

#endif // NOESIS_RL_SAMPLE_ADAPTIVE_TRAJECTORY_SAMPLER_HPP_

/* EOF */
