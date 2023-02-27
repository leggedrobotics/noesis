/*!
 * @author    Markus Staeuble
 * @email     markus.staeuble@mavt.ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

#include <noesis_agents/runner/train/realtime/RealTimeRunner.hpp>

namespace noesis {
namespace runner {

template<typename ScalarType_>
RealTimeRunner<ScalarType_>::RealTimeRunner(const RealTimeRunnerConfig& config):
core::Object(config.name, config.scope, config.verbose),
observations_(noesis::utils::make_namescope({config.scope, "observations"}), 1, numberOfInstances_),
actions_(noesis::utils::make_namescope({config.scope, "actions"}), 1, numberOfInstances_),
rewards_(noesis::utils::make_namescope({config.scope, "rewards"}), {1, 1, numberOfInstances_}, false),
terminalRewards_(noesis::utils::make_namescope({config.scope, "terminal_rewards"}), {1, 1, numberOfInstances_}, false),
terminations_(numberOfInstances_, noesis::Termination::Unterminated),  
savingInterval_(0, utils::make_namescope({config.scope, config.name,"/saving_interval"}), {0, MaxInt}),
maxStepsPerEpisode_(0, noesis::utils::make_namescope({config.scope, config.name, "max_steps_per_episode"}), {0, std::numeric_limits<int>::max()}),
useLearningThread_(config.use_learning_thread)
{
  hyperparam::manager->addParameter(savingInterval_);
  hyperparam::manager->addParameter(maxStepsPerEpisode_);
  logger_ = std::make_unique<log::TensorBoardLogger>("", config.name, config.name, config.scope, config.verbose);
  logger_->setAutosaveEnabled(false);
}

template<typename ScalarType_>
RealTimeRunner<ScalarType_>::~RealTimeRunner() {
  hyperparam::manager->removeParameter(savingInterval_);
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::setGraph(const GraphPtr& graph) {
  NFATAL_IF(graph == nullptr, "[" << namescope() << "]: 'graph' pointer argument is invalid (nullptr).");
  graph_ = graph;
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::setAgent(const AgentPtr& agent) {
  NFATAL_IF(agent == nullptr, "[" << namescope() << "]: 'agent' pointer argument is invalid (nullptr).");
  auto* agentPtr = dynamic_cast<agent::TrajectoryAgent<ScalarType>*>(agent);
  NFATAL_IF(agentPtr == nullptr, "[" << namescope() << "]: Could not cast agent. Make sure to derive from TrajectoryAgent.");
  agent_ = agentPtr;
  agent_->setLogger(logger_.get());
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::configure() {
  auto name = namescope();
  NINFO("[" << name << "]: Configuring runner ...");
  NINFO("[" << name << "]: Agent saving interval: " << static_cast<int>(savingInterval_));
  NINFO("[" << name << "]: Is verbose: " << std::boolalpha << isVerbose());
  NFATAL_IF(!agent_, "[" << name << "]: An agent has not been set.");
  // Configure the data sampler
  NFATAL_IF(maxStepsPerEpisode_ == 0, "[" << namescope() << "]: The maximum number of steps per episode has not been set!");
  episodeCounters_.resize(numberOfInstances_, 0);
  stepCounters_.resize(numberOfInstances_, 0);
  configureObservationsAndActions();
  configureImpl();
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::startup() {
  NINFO("[" << namescope() << "]: Starting-up runner ...");
  logger_->startup();

  if(useLearningThread_) {
    learningWorker_ = std::make_unique<std::thread>(&RealTimeRunner::learningCallback, this);
  }
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::shutdown() {
  saveAgent();
  shutdownImpl();
  noesis::hyperparam::manager->removeParameter(maxStepsPerEpisode_);

  if(useLearningThread_) {
    stopUpdating_ = true;
    cvLearning_.notify_all();
    if(learningWorker_->joinable()) {
      learningWorker_->join();
    }
  }

  agent_ = nullptr;
  episodeCounters_.clear();
  stepCounters_.clear();
  maxStepsPerEpisode_ = 0;
  totalSampleTransitionsCounter_ = 0;
  totalEpisodesCounter_ = 0;
  logger_->shutdown();
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::reset() {
  totalSampleTransitionsCounter_ = 0;
  totalEpisodesCounter_ = 0;
  std::lock_guard<std::mutex> lock(agentMutex_);
  agent_->reset();
  for (size_t instance = 0; instance < numberOfInstances_; instance++) {
    agent_->experienceInitialization(instance, observations_);
    episodeCounters_.front() = 0;
    stepCounters_.front() = 0;
  }
  totalAgentUpdatesCounter_ = 0;
  totalGraphSavesCounter_ = 0;

  resetImpl();
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::run(RunMode mode, size_t duration) {
  static bool firstTime = true;
  updateObservations();

  endOfTrajectory_ = false;
  std::lock_guard<std::mutex> lock(agentMutex_);

  if(!isLearning_) {
    if (firstTime) {
      agent_->reset();
      agent_->experienceInitializations(observations_);
      firstTime = false;
    }

    // run
    updateRewards();
    agent_->experienceTransitions(actions_, observations_, rewards_);
    stepCounters_.front()++;
    bool agentIsReadyToUpdate = agent_->hasEnoughExperience();
    if (stepCounters_.front() >= maxStepsPerEpisode_) {
      terminations_[0] = noesis::Termination::TimeOut;
    }
    if (terminations_[0] != noesis::Termination::Unterminated || agentIsReadyToUpdate) {
      // Store the terminal experience separately
      agent_->experienceTerminations(observations_, terminalRewards_, terminations_);
      endOfTrajectory_ = true;
      // If agent is ready for updating, deactivate the instance otherwise start a new episode
      if (!agentIsReadyToUpdate) {
        updateObservations();
        agent_->experienceInitializations(observations_);
        stepCounters_.front() = 0;
      }
    }

    if(agentIsReadyToUpdate && useLearningThread_ ) {
      updateCounter_++;
      cvLearning_.notify_all();
    }
  }

  agent_->explore(observations_, actions_);
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::act() {
  updateObservations();
  std::lock_guard<std::mutex> lock(agentMutex_);
  agent_->act(observations_, actions_);
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::updateAgent() {
  isLearning_ = true;
  std::lock_guard<std::mutex> lock(agentMutex_);
  agent_->learn();
  totalAgentUpdatesCounter_++;
  int savingInterval = savingInterval_;
  if (savingInterval == 0 ? false : (totalAgentUpdatesCounter_) % savingInterval == 0) {
    saveAgent();
  }
  
  updateLogger();
  agent_->reset();
  agent_->experienceInitializations(observations_);

  stepCounters_.front() = 0;
  isLearning_ = false;
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::saveAgent() {
  if (graph_) {
    graph_->save();
    totalGraphSavesCounter_++;
    NINFO("[" << getScope() << "]: Saving graph: Version '" << totalGraphSavesCounter_ << "' at agent update '"
      << totalAgentUpdatesCounter_ << "'");
  }
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::learningCallback() {
  unsigned long localCounter = 0lu;

  while(!stopUpdating_) {
    std::unique_lock<std::mutex> lock{mutexLearningUpdate_};
    cvLearning_.wait(lock,[this,localCounter](){
           if (stopUpdating_) return true;
           return (updateCounter_ > localCounter);
    });

    localCounter = updateCounter_;

    // Stop immediately
    if (stopUpdating_) {
      return;
    }

    updateAgent();
  }
}

template<typename ScalarType_>
void RealTimeRunner<ScalarType_>::configureObservationsAndActions() {
  observations_.setFromSpec(observationsSpec_);
  actions_.setFromSpec(actionsSpec_);
}


} // namespace runner
} // namespace noesis
