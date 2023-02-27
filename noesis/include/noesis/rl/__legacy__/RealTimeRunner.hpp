/*!
 * @author    Markus Staeuble
 * @email     markus.staeuble@mavt.ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_RUNNER_REALTIME_REALTIME_RUNNER_HPP_
#define NOESIS_RL_RUNNER_REALTIME_REALTIME_RUNNER_HPP_

#include <thread>

#include <noesis_agents/runner/RunnerInterface.hpp>
#include <noesis/framework/core/Object.hpp>
#include <noesis/framework/hyperparam/hyper_parameters.hpp>
#include <noesis_agents/agent/rl/TrajectoryAgent.hpp>

namespace noesis {
namespace runner {

struct RealTimeRunnerConfig : RunnerConfig {
  bool use_learning_thread{false};
};

template<typename ScalarType_>
class RealTimeRunner : public core::Object, public RunnerInterface<ScalarType_>
{
public:
  using ScalarType = ScalarType_;
  using EnvironmentPtr = typename RunnerInterface<ScalarType>::EnvironmentPtr;
  using VisualizerPtr = typename RunnerInterface<ScalarType>::VisualizerPtr;  
  using GraphPtr = typename RunnerInterface<ScalarType>::GraphPtr;
  using AgentPtr = typename RunnerInterface<ScalarType>::AgentPtr;
  using ObservationsType = core::Observations<ScalarType>;
  using ActionsType = core::Actions<ScalarType>;
  using TensorType = Tensor<ScalarType>;
  using RewardsType = TensorType;
  static constexpr int MaxInt = std::numeric_limits<int>::max();

  explicit RealTimeRunner(const RealTimeRunnerConfig& config);

  ~RealTimeRunner() override;

  /*!
   * @brief Abstract interface for environment instance management.
   * @note The following functions are presented in order of use.
   */

  void setGraph(const GraphPtr& graph) final;
  
  void setEnvironment(const EnvironmentPtr& environment) final {
    // do nothing
  }
  
  void setAgent(const AgentPtr& agent) final;
  
  void setVisualizer(const VisualizerPtr& visualizer) final {
    // do nothing
  }

  void configure() final;

  void startup() final;

  void shutdown() final;

  void reset() final;
  
  void run(RunMode mode, size_t duration) final;

  bool isRunning() final {
    return true;
  }

  bool trajectoryHasEnded() {
    return endOfTrajectory_;
  }

  void act();

  /*!
   * @brief Configures the tensor specifications of the agent observations inputs.
   */
  void setObservationsSpecifications(const TensorsSpec& spec) {
    observationsSpec_ = spec;
  }
  
  /*!
   * @brief Configures the tensor specifications of the agent actions outputs.
   */
  void setActionsSpecifications(const TensorsSpec& spec) {
    actionsSpec_ = spec;
  }

  /*!
   * @brief Retrieves the tensor specifications of the estimator inputs.
   * @return TensorsSpec containing name-dimensions pairs defining estimator inputs.
   */
  const TensorsSpec& getObservationsSpecifications() const {
    return observationsSpec_;
  }
  
  /*!
   * @brief Retrieves the tensor specifications of the estimator outputs.
   * @return TensorsSpec containing name-dimensions pairs defining estimator outputs.
   */
  const TensorsSpec& getActionsSpecifications() const {
    return actionsSpec_;
  }

  /*!
   * @brief Abstract interface for environment runtime operations.
   */
  virtual void updateObservations() = 0;
  virtual void updateRewards() = 0;
  virtual ActionsType getAction() = 0;
  virtual void shutdownImpl() = 0;
  virtual void resetImpl() = 0;
  virtual void configureImpl() = 0;

  void updateAgent();

  void saveAgent();

protected:
  size_t getNumInstances() const {
    return numberOfInstances_;
  }

  ActionsType& actions() {
    return actions_;
  }

  const ActionsType& actions() const {
    return actions_;
  }

  ObservationsType& observations() {
    return observations_;
  }

  const ObservationsType& observations() const {
    return observations_;
  }

  RewardsType& rewards() {
    return rewards_;
  }

  const RewardsType& rewards() const {
    return rewards_;
  }

  RewardsType& terminalRewards() {
    return terminalRewards_;
  }

  const RewardsType& terminalRewards() const {
    return terminalRewards_;
  }

  Terminations& terminations() {
    return terminations_;
  } 

  const Terminations& terminations() const {
    return terminations_;
  }

  inline void updateLogger() {
    logger_->flush();
  }

  void learningCallback();

  void configureObservationsAndActions();

private:
  const size_t numberOfInstances_{1};
  //! @brief A vector of counters to record the current number of episodes executed by each environment instance.
  std::vector<size_t> episodeCounters_;
  //! @brief A vector of counters to record the current number of steps per environment instance.
  std::vector<size_t> stepCounters_;
  //! @brief Mutex protecting the agent.
  std::mutex agentMutex_;
  //! @brief The pointer to the agent from which to generate and collect action samples.
  agent::TrajectoryAgent<ScalarType>* agent_{nullptr};
  //! @brief The maximum number of environment steps (i.e. transitions) per episode.
  noesis::hyperparam::HyperParameter<int> maxStepsPerEpisode_;
  //! @brief The total number of transition samples (i.e. SARS tuples) collected from the environment.
  size_t totalSampleTransitionsCounter_{0};
  //! @brief The total number of episodes executed by the runner.
  size_t totalEpisodesCounter_{0};
  //! @brief Logger used to record training performance for visualization in TensorBoard
  std::unique_ptr<log::TensorBoardLogger> logger_{nullptr};
  //! @brief A computation graph which may contain operations for both the agent and environment.
  GraphPtr graph_{nullptr};
  //! @brief The interval (in terms of agent updates) at which the agent's state is saved.
  hyperparam::HyperParameter<int> savingInterval_;
  //! @brief The number of agent update operations (i.e. calls to learn()) performed.
  size_t totalGraphSavesCounter_{0};
  //! @brief The number of agent update operations (i.e. calls to learn()) performed.
  size_t totalAgentUpdatesCounter_{0};

  //! Stores the tensor specifications for observations
  TensorsSpec observationsSpec_;
  //! Stores the tensor specifications for actions
  TensorsSpec actionsSpec_;

  ActionsType actions_;
  ObservationsType observations_;
  RewardsType rewards_;
  RewardsType terminalRewards_;
  Terminations terminations_;

  bool endOfTrajectory_{false};
  std::atomic_bool isLearning_{false};
  bool useLearningThread_{false};
  std::condition_variable cvLearning_;
  std::mutex mutexLearningUpdate_;  
  std::atomic_uint updateCounter_{0u};
  std::atomic_bool stopUpdating_{false};
  std::unique_ptr<std::thread> learningWorker_{nullptr};

};

} // namespace runner
} // namespace noesis

#include <noesis_agents/runner/train/realtime/RealTimeRunner.tpp>

#endif // NOESIS_RL_RUNNER_REALTIME_REALTIME_RUNNER_HPP_

/* EOF */
