/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_CORE_SYNCHRONIZED_HPP_
#define NOESIS_GYM_CORE_SYNCHRONIZED_HPP_

// C/C++
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

// Boost
#include <boost/thread/shared_mutex.hpp>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/mdp/EnvironmentInterface.hpp"

namespace noesis {
namespace gym {

template <typename ScalarType_>
class Synchronizer final:
  public ::noesis::core::Object,
  public ::noesis::mdp::EnvironmentInterface<ScalarType_>
{
public:

  // Aliases
  using Object = ::noesis::core::Object;
  using Interface = ::noesis::mdp::EnvironmentInterface<ScalarType_>;
  using Scalar = typename Interface::Scalar;
  using Observations = typename Interface::Observations;
  using Actions = typename Interface::Actions;
  using Rewards = typename Interface::Rewards;
  using Metrics = typename Interface::Metrics;
  using Termination = typename Interface::Termination;
  using Terminations = typename Interface::Terminations;
  using Names = typename Interface::Names;
  
  /*
   * Instantiation
   */

  Synchronizer() = delete;

  Synchronizer(Synchronizer&& other) = default;
  Synchronizer& operator=(Synchronizer&& other) = default;

  Synchronizer(const Synchronizer& other) = delete;
  Synchronizer& operator=(const Synchronizer& other) = delete;
  
  explicit Synchronizer(
    Interface* environment,
    const std::string& name="Synchronizer",
    const std::string& scope="/",
    bool verbose=false):
    Object(name, scope, verbose),
    Interface(),
    environment_(environment)
  {
    NFATAL_IF(!environment_, "[" << namescope() << "]: 'environment' pointer argument is invalid (nullptr).");
    NNOTIFY("[Synchronizer]: New instance at: " << std::hex << this);
  }

  ~Synchronizer() final = default;
  
  /*
   * Properties
   */

  size_t batch_size() const override { return environment_->batch_size(); }
  
  size_t history_size() const override { return environment_->history_size(); }
  
  size_t max_steps() const override { return environment_->max_steps(); }
  
  Scalar time_limit() const override { return environment_->time_limit(); }
  
  Scalar time_step() const override { return environment_->time_step(); }
  
  Scalar discount_factor() const override { return environment_->discount_factor(); }
  
  TensorsSpec actions_spec() const override { return environment_->actions_spec(); }
  
  TensorsSpec observations_spec() const override { return environment_->observations_spec(); }
  
  Names tasks() const override { return environment_->tasks(); }
  
  Actions& actions() override { return environment_->actions(); }
  
  const Actions& actions() const override { return environment_->actions(); }
  
  const Observations& observations() const override { return environment_->observations(); }
  
  const Rewards& rewards() const override { return environment_->rewards(); }
  
  const Terminations& terminations() const override { return environment_->terminations(); }
  
  const Metrics& metrics() const override { return environment_->metrics(); }
  
  size_t steps() const override { return environment_->steps(); }
  
  Scalar time() const override { return environment_->time(); }
  
  std::ostream& info(std::ostream& os) const override { return environment_->info(os); }
  
  Interface* env() const { return environment_; }
  
  /*
   * Operations
   */

  void configure() override {
    environment_->configure();
  }
  
  void seed(int seed) override {
    environment_->seed(seed);
  }
  
  void reset() override {
    environment_->reset();
  }
  
  void step() override {
    // Wait for external synchronization if enabled.
    if (useLocking_) { sync(); }
    environment_->step();
  }
  
  /*!
   * @brief Enables asynchronous operation.
   */
  void lock() {
    useLocking_.store(true);
    shouldRun_.store(false);
  }
  
  /*!
   * @brief Disables asynchronous operation.
   */
  void unlock() {
    useLocking_.store(false);
    shouldRun_.store(false);
    cv_.notify_one();
  }
  
  /*!
   * @brief Triggers execution of step() if blocked on waiting.
   * @note Use this to synchronize with the thread calling the step methods.
   */
  void start() {
    cv_.notify_one();
  }
  
  /*!
   * @brief Waits until the synchronization call from step().
   * @note Use this to synchronize with the thread calling the step method.
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{return !this->shouldRun_.load();});
    shouldRun_.store(true);
  }
  
private:

  inline void sync() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.notify_one();
    cv_.wait(lock, [this]{return this->shouldRun_.load();});
    shouldRun_.store(false);
  }

private:
  //! @brief Condition variable synchronizing the renderer with the environment.
  std::condition_variable cv_;
  //! @brief Mutex to protect access to the condition variable member.
  std::mutex mutex_;
  //! @brief Internal flag indicating if external synchronization is enabled.
  std::atomic_bool useLocking_{false};
  //! @brief Internal flag signaling the state of the renderer-environment synchronization.
  std::atomic_bool shouldRun_{false};
  //! @brief A pointer to the environment to be wrapped.
  Interface* environment_{nullptr};
};

template <class ScalarType_>
auto make_synchronized_wrapper(::noesis::mdp::EnvironmentInterface<ScalarType_>* env) {
  return std::make_unique<Synchronizer<ScalarType_>>(env);
}

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_CORE_SYNCHRONIZED_HPP_

/* EOF */
