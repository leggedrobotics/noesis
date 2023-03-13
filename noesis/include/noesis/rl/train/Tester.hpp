/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_TRAIN_TEST_MONITOR_HPP_
#define NOESIS_RL_TRAIN_TEST_MONITOR_HPP_

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/log/metric.hpp"
#include "noesis/mdp/AgentInterface.hpp"
#include "noesis/gym/core/Vector.hpp"
#include "noesis/gym/train/MonitorInterface.hpp"

namespace noesis {
namespace train {

template <class ScalarType_>
class Tester final:
  public ::noesis::core::Object,
  public ::noesis::gym::MonitorInterface<ScalarType_>
{
public:

  // Aliases
  using Object = ::noesis::core::Object;
  using Interface = ::noesis::gym::MonitorInterface<ScalarType_>;
  using Scalar = typename Interface::Scalar;
  using Metrics = typename Interface::Metrics;
  using Logger = ::noesis::log::TensorBoardLogger;
  using Termination = ::noesis::mdp::Termination<Scalar>;
  using Environment = ::noesis::gym::Vector<Scalar>;
  using Agent = ::noesis::mdp::AgentInterface<Scalar>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  
  // Constants
  static constexpr int MaxInt = std::numeric_limits<int>::max();
  
  // Configuration helper
  struct Config {
    Agent* agent{nullptr};
    Environment* environment{nullptr};
    Logger* logger{nullptr};
    size_t episodes{1};
    std::string name{"Tester"};
    std::string scope{"/"};
    bool verbose{false};
  };
  
  /*
   * Instantiation
   */
  
  //! @note Move and move-assignment construction is permissible for this class.
  Tester(Tester&& other) noexcept = default;
  Tester& operator=(Tester&& other) noexcept = default;
  
  //! @note Copy and copy-assignment construction is permissible for this class.
  Tester(const Tester& other) = delete;
  Tester& operator=(const Tester& other) = delete;
  
  explicit Tester(const Config& config=Config()):
    Tester(config.agent, config.environment, config.logger,
     config.episodes, config.name, config.scope, config.verbose)
  {
  }
  
  explicit Tester(
      Agent* agent,
      Environment* environment,
      Logger* logger,
      size_t episodes,
      const std::string& name="Tester",
      const std::string& scope="/",
      bool verbose=false):
    Object(name, scope, verbose),
    Interface(),
    targetEpisodes_(episodes),
    agent_(agent),
    environment_(environment),
    logger_(logger)
  {
  }

  ~Tester() override = default;

  /*
   * Configurations
   */

  void setAgent(Agent* agent) {
    NFATAL_IF(!agent, "[" << namescope() << "]: 'agent' pointer argument is invalid (nullptr).")
    agent_ = agent;
  }

  void setEnvironment(Environment* environment) {
    NFATAL_IF(!environment, "[" << namescope() << "]: 'environment' pointer argument is invalid (nullptr).")
    environment_ = environment;
  }
  
  void setLogger(Logger* logger) {
    NFATAL_IF(!logger, "[" << namescope() << "]: 'logger' pointer argument is invalid (nullptr).")
    logger_ = logger;
  }

  void setTargetEpisodes(int episodes) {
    targetEpisodes_ = episodes;
  }
  
  /*
   * Properties
   */

  const Metrics& metrics() const override { return metrics_; }

  std::string info() const override { return metrics_.info(); }

  /*
   * Operations
   */

  void configure() override {
    const auto name = this->name();
    const auto ns = this->namescope();
    NINFO("[" << ns << "]: Configuring tester ...")
    NFATAL_IF(!agent_, "[" << ns << "]: An agent has not been set!")
    NFATAL_IF(!environment_, "[" << ns << "]: An environment has not been set!")
    NINFO("[" << ns << "]: Episodes per run: " << targetEpisodes_)
    // Define metrics collected by the sampler.
    metrics_.clear();
    metrics_.push_back(name + "/agent/mean_step_reward");
    metrics_.push_back(name + "/agent/mean_episode_reward");
    metrics_.push_back(name + "/agent/mean_episode_length");
    metrics_.push_back(name + "/agent/mean_episode_return");
    metrics_.push_back(name + "/agent/mean_terminal_value");
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
    transitionCounters_.resize(Ne, 0);
    errorCounters_.resize(Ne, 0);
    // NOTE: We allocate the maximum amount of space for each instance a
    // s a precaution. Maybe we can do smarter in the future.
    const auto capacity = targetEpisodes_;
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
    for (size_t e = 0; e < Ne; e++) {
      episodeTerminalMetrics_[e].clear();
      episodeTerminalValues_[e].clear();
      episodeStepMetrics_[e].clear();
      episodeRewards_[e].clear();
      episodeLengths_[e].clear();
      episodeTerminations_[e].clear();
      transitionCounters_[e] = 0;
      errorCounters_[e] = 0;
    }
  }

  void reset() override {
    // TODO
  }

  // TODO: why return bool here?
  bool update() override {
    // NOTE: We reset the environment to ensure that it all
    // instances are initialized properly.
    environment_->reset();
    // Prepare the internal recording accumulators and counters.
    prepare();
    // Execute a fixed number of trajectories to compute performance statistics.
    collect();
    // Record performance metrics for this round of testing.
    record();
    // Success
    return true;
  }

private:

  void prepare() {
    // If the number of target episodes is less than the number of environment
    // instance,s we deactivate the extra instances that aren't needed.
    for (size_t e = 0; e < environment_->batch_size(); e++) {
      if (e < static_cast<size_t>(targetEpisodes_)) {
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
        transitionCounters_[e] = 0;
        errorCounters_[e] = 0;
      } else {
        environment_->deactivate(e);
      }
    }
    // Initialize the episodes counter to the number
    // of active instances used to collect test data.
    totalEpisodes_ = environment_->active();
  }
  
  void collect() {
    while (environment_->active() > 0) {
      environment_->resize_buffers();
      environment_->collect_observations();
      agent_->act(environment_->observations(), environment_->actions());
      environment_->distribute_actions();
      environment_->step_environments();
      for (size_t e = 0; e < environment_->batch_size(); e++) {
        if (!environment_->isActive(e)) { continue; }
        auto& env = (*environment_)[e];
        const auto& rew = env.rewards();
        const auto& trm = env.terminations().back();
        const auto& met = env.metrics();
        if (static_cast<int>(trm.type) >= 0) {
          transitionCounters_[e]++;
          episodeRewards_[e].back() += rew.asFlat();
          episodeStepMetrics_[e].back() += met.values();
          if (trm.type != Termination::Type::Unterminated) {
            DNWARNING_IF(isVerbose(), "[" << namescope() << "]: Environment '" << e << "' was terminal!"
              << "\nTermination: " << static_cast<int>(trm.type))
            episodeTerminalMetrics_[e].back() = met.values();
            episodeTerminalValues_[e].back() = trm.value;
            episodeLengths_[e].back() = transitionCounters_[e];
            episodeTerminations_[e].back() = trm.id;
            if (totalEpisodes_ < targetEpisodes_) {
              DNWARNING_IF(isVerbose(), "[" << namescope() << "]: Resetting environment '" << e << "'!"
                << "\nEpisode length: " << episodeLengths_[e].back())
              env.reset();
              totalEpisodes_++;
              episodeTerminalMetrics_[e].push_back(Vector::Zero(environment_->metrics().size()));
              episodeStepMetrics_[e].push_back(Vector::Zero(environment_->metrics().size()));
              episodeRewards_[e].push_back(Vector::Zero(environment_->tasks().size()));
              episodeTerminalValues_[e].push_back(0);
              episodeLengths_[e].push_back(0);
              episodeTerminations_[e].push_back(0);
              transitionCounters_[e] = 0;
            } else {
              DNWARNING_IF(isVerbose(), "[" << namescope() << "]: Deactivating environment '" << e << "'!"
                << "\nEpisode length: " << episodeLengths_[e].back())
              environment_->deactivate(e);
            }
          }
        } else {
          NERROR("[" << namescope() << "]: Environment '" << e << "': Episode has invalid data!\n" << env)
          transitionCounters_[e] -= env.steps();
          episodeStepMetrics_[e].back().setZero();
          episodeRewards_[e].back().setZero();
          env.reset();
          errorCounters_[e]++;
        }
      }
    }
  }
  
  void record() {
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
    // Update the sampler's metrics using current episode statistics
    // NOTE: The first element of the rewards vector holds the TOTAL reward.
    metrics_[name + "/agent/mean_step_reward"] = rewards_sum(0)/(lengths_sum + eps);
    metrics_[name + "/agent/mean_episode_reward"] = rewards_sum(0)/(number_of_episodes + eps);
    metrics_[name + "/agent/mean_episode_length"] = lengths_sum/(number_of_episodes + eps);
    metrics_[name + "/agent/mean_episode_return"] = (rewards_sum(0) + terminal_value_sum)/(number_of_episodes + eps);
    metrics_[name + "/agent/mean_terminal_value"] = (terminal_value_sum)/(number_of_episodes + eps);
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
  //! @brief Counters of sample errors currently recorded from each environment instance.
  std::vector<size_t> errorCounters_;
  //! @brief Defines the total number of episodes used to test agent performance.
  int targetEpisodes_{1};
  //! @brief Counter of the total number of generated environment episodes/trajectories.
  int totalEpisodes_{0};
  //! @brief Pointer to the agent from which to sample actions.
  Agent* agent_{nullptr};
  //! @brief Pointer to the environment (vector) from which to sample observations.
  Environment* environment_{nullptr};
  //! @brief The TensorBoard logger used by all modules to record logging signals.
  Logger* logger_{nullptr};
};

} // namespace train
} // namespace noesis

#endif // NOESIS_RL_TRAIN_TEST_MONITOR_HPP_

/* EOF */
