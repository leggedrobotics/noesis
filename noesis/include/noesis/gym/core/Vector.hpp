/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_CORE_VECTOR_HPP_
#define NOESIS_GYM_CORE_VECTOR_HPP_

// C/C++
#include <omp.h>
#include <vector>
#include <type_traits>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/mdp/EnvironmentInterface.hpp"

namespace noesis {
namespace gym {

template <typename ScalarType_>
class Vector final:
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
  using Environments = std::vector<std::unique_ptr<Interface>>;
  
  /*
   * Instantiation
   */

  Vector(Vector&& other) = default;
  Vector& operator=(Vector&& other) = default;
  
  //! @note We do not permit copy operations because Tensor and environment
  //! copies would point to the same memory allocations as the source objects.
  Vector(const Vector& other) = delete;
  Vector& operator=(const Vector& other) = delete;
  
  explicit Vector(
      const std::string& name="Vector",
      const std::string& scope="/",
      bool verbose=false):
    Object(name, scope, verbose),
    Interface()
  {
  }
  
  ~Vector() final = default;
  
  template<class EnvironmentType_, typename... Arguments_>
  void build(size_t size, Arguments_&&... args) {
    // NOTE: We must ensure that the environment implementation is derived from the interface class.
    static_assert(
      std::is_base_of<noesis::mdp::EnvironmentInterface<typename EnvironmentType_::Scalar>, EnvironmentType_>::value,
      "EnvironmentType_ is not a subclass of noesis::mdp::EnvironmentInterface<>."
    );
    environments_.clear();
    NFATAL_IF(size == 0, "[Vector]: Argument 'size' (size_t) must be non-zero!");
    for (size_t e = 0; e < size; ++e) { environments_.emplace_back(std::make_unique<EnvironmentType_>(std::forward<Arguments_>(args)...)); }
    NNOTIFY_IF(isVerbose(), "[Vector]: New instance at: " << std::hex << this);
    setup();
  }
  
  /*
   * Properties
   */

  size_t batch_size() const override { return environments_.size()*environments_.front()->batch_size(); }
  
  size_t history_size() const override { return environments_.front()->history_size(); }
  
  size_t max_steps() const override { return environments_.front()->max_steps(); }
  
  Scalar time_limit() const override { return environments_.front()->time_limit(); }
  
  Scalar time_step() const override { return environments_.front()->time_step(); }
  
  Scalar discount_factor() const override { return environments_.front()->discount_factor(); }
  
  TensorsSpec actions_spec() const override { return actions_.spec(); }
  
  TensorsSpec observations_spec() const override { return observations_.spec(); }
  
  Names tasks() const override { return environments_.front()->tasks(); }
  
  Actions& actions() override { return actions_; }
  
  const Actions& actions() const override { return actions_; };
  
  const Observations& observations() const override { return observations_; };
  
  const Rewards& rewards() const override { return rewards_; };
  
  const Terminations& terminations() const override { return terminations_; };
  
  const Metrics& metrics() const override { return metrics_; }
  
  size_t steps() const override {
    size_t total = 0;
    for (size_t e = 0; e < environments_.size(); e++) { total += environments_[e]->steps(); }
    return total;
  }
  
  Scalar time() const override {
    Scalar total = 0;
    for (size_t e = 0; e < environments_.size(); e++) { total += environments_[e]->time(); }
    return total;
  }
  
  size_t active() const {
    return active_;
  }
  
  bool isActive(size_t index) const {
    return isActive_[index];
  }
  
  void activate(size_t index) {
    isActive_[index] = 1;
    active_++;
  }
  
  void deactivate(size_t index) {
    isActive_[index] = 0;
    active_--;
  }
  
  /*
   * Operations
   */
  
  void configure() override {
    for (auto& env: environments_) { env->configure(); }
    setup();
  }
  
  void seed(int seed) override {
    for (auto& env: environments_) { env->seed(seed++); }
  }
  
  void reset() override {
    const auto Ne = environments_.size();
    active_ = Ne;
    resize(active_);
    #pragma omp parallel for schedule(static)
    for (size_t e = 0; e < Ne; e++) {
      isActive_[e] = 1;
      reset(e);
    }
  }
  
  void step() override {
    const auto Ne = environments_.size();
    // NOTE: We perform parallel stepping of the environment instances
    // using OpenMP, and only those currently active actually perform steps.
    // This ensures that inactive instances do not perform any operations.
    #pragma omp parallel for schedule(static)
    for (size_t e = 0; e < Ne; e++) { if (isActive_[e]) { step(e); } }
    record();
  }
  
  /*
   * Operations for constant-sized batching
   */
  
  /*!
   * @brief Resets a single environment instance at a specified index.
   * @note This operations is thread-compatible.
   * @param index The instance to be reset.
   */
  void reset(size_t index) {
    const size_t B = environments_[index]->batch_size();
    environments_[index]->reset();
    for (size_t k = 0; k < observations_.size(); k++) {
      observations_[k].batch_block(index*B, B) = environments_[index]->observations()[k].batch_block(0, B);
    }
  }
  
  /*!
   * @brief Steps a single environment instance at a specified index.
   * @note This operations is thread-compatible.
   * @param index The instance to be stepped.
   */
  void step(size_t index) {
    const size_t B = environments_[index]->batch_size();
    const size_t Na = actions_.size();
    const size_t No = observations_.size();
    for (size_t k = 0; k < Na; k++) {
      environments_[index]->actions()[k].batch_block(0, B) = actions_[k].batch_block(index*B, B);
    }
    environments_[index]->step();
    for (size_t k = 0; k < No; k++) {
      observations_[k].batch_block(index*B, B) = environments_[index]->observations()[k].batch_block(0, B);
    }
    rewards_.batch_block(index*B, B) = environments_[index]->rewards().batch_block(0, B);
    terminations_[index] = environments_[index]->terminations().back();
  }
  
  /*
   * Operations for variable-sized batching
   */
  
  /*!
   * @brief Distributes actions from the batched buffer to active instances.
   * @note The batching index `b` is computed using within the loop based on the
   *       activity flags so that we can parallelize the for loop.
   */
  void distribute_actions() {
    const auto Ne = environments_.size();
    const size_t B = environments_.front()->batch_size();
    #pragma omp parallel for schedule(static)
    for (size_t e = 0; e < Ne; e++) {
      if (isActive_[e]) {
        const size_t b = std::accumulate(isActive_.begin(), isActive_.begin()+e, static_cast<size_t>(0u));
        for (size_t k = 0; k < actions_.size(); k++) {
          environments_[e]->actions()[k].batch_block(0, B) = actions_[k].batch_block(b*B, B);
        }
      }
    }
  }
  
  /*!
   * @brief Steps all the currently active instances.
   * @note This operation also performs the recording of metrics.
   */
  void step_environments() {
    const auto Ne = environments_.size();
    #pragma omp parallel for schedule(static)
    for (size_t e = 0; e < Ne; e++) { if (isActive_[e]) { environments_[e]->step(); } }
    record();
  }
  
  /*!
   * @brief Collects observations from individual instances into the batched buffer.
   * @note The batching index `b` is computed using within the loop based on the
   *       activity flags so that we can parallelize the for loop.
   */
  void collect_observations() {
    const auto Ne = environments_.size();
    const size_t B = environments_.front()->batch_size();
    #pragma omp parallel for schedule(static)
    for (size_t e = 0; e < Ne; e++) {
      if (isActive_[e]) {
        const size_t b = std::accumulate(isActive_.begin(), isActive_.begin()+e, static_cast<size_t>(0u));
        for (size_t k = 0; k < observations_.size(); k++) {
          observations_[k].batch_block(b*B, B) = environments_[e]->observations()[k].batch_block(0, B);
        }
      }
    }
  }
  
  /*!
   * @brief Resizes the batched observations-actions buffers according
   *        to the current number of active environment instances.
   * @note This can be used to prevent redundant and/or wasteful graph operations.
   */
  void resize_buffers() {
    if (observations_.batches() != active_) { resize(active_); }
  }
  
  /*
   * Accessor operations
   */
  
  Interface* begin() { return environments_.front().get(); }
  
  Interface* end() { return environments_.back().get(); }
  
  Interface& front() { return *environments_.front(); }
  
  Interface& back() { return *environments_.back(); }
  
  Interface& operator[](size_t instance) { return *environments_[instance]; }
  
  /*
   * Helper operations
   */

  std::ostream& info(std::ostream& os) const override {
    os << "\n[noesis::gym::Vector]:";
    os << "\nAddress: " << std::hex << this;
    os << "\nType: " << utils::typename_to_string(environments_.back());
    os << "\nName: " << namescope();
    os << "\nSize {Batch, History}: {" << batch_size() << ", " << history_size() << "}";
    for (const auto& env: environments_) { os << *env; }
    return os;
  }
  
private:
  
  void setup() {
    using namespace utils;
    // The reference environment is set to the first instance.
    auto& env = environments_.front();
    // Retrieve environment properties
    const size_t batch_size = this->batch_size();
    const size_t history_size = this->history_size();
    const auto obs = env->observations_spec();
    const auto act = env->actions_spec();
    const auto tasks = env->tasks().size();
    const auto ns = namescope();
    // Configure the OpenMP thread-pool
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(batch_size));
    // Configure data containers
    observations_.clone(Observations(make_namescope({ns, "observations"}), obs, history_size, batch_size));
    actions_.clone(Actions(make_namescope({ns, "actions"}), act, history_size, batch_size));
    rewards_.clone(Rewards(make_namescope({ns, "rewards"}), {tasks, history_size, batch_size}, true));
    metrics_ = environments_.front()->metrics();
    metrics_.reset();
    terminations_.clear();
    terminations_.resize(batch_size);
    isActive_.resize(batch_size, 0);
  }
  
  void resize(size_t size) {
    observations_.resize(size);
    actions_.resize(size);
  }
  
  void record() {
    metrics_.reset();
    if (metrics_.size() > 0 && active_ > 0) {
      for (size_t e = 0; e < environments_.size(); e++) {
        if (isActive_[e]) {
          metrics_.values() += environments_[e]->metrics().values();
        }
      }
      metrics_.values() /= static_cast<Scalar>(active_);
    }
  }
  
private:
  //! @brief Buffer of aggregated observations samples.
  //! @note (1) This buffer determines the batch-wise parallelization
  //!       when given as input to graph operations.
  //! @note (2) This is filled with sample collected from individual
  //!       environment instances.
  Observations observations_;
  //! @brief Buffer of aggregated actions samples.
  Actions actions_;
  //! @brief Buffer of aggregated rewards samples.
  Rewards rewards_;
  //! @brief Metrics container for the vectorized environment.
  //! @note Values contained herein are averages over the instances.
  Metrics metrics_;
  //! @brief Buffer of aggregated terminations.
  Terminations terminations_;
  //! @brief The vector of environment instances.
  //! @note Instances are held via pointer to EnvironmentInterface.
  Environments environments_;
  //! @brief A vector of flags indicating whether the corresponding
  //! environment instance is currently active (1) or not (0).
  //! @note We use `size_t` integers instead of booleans because:
  //!       (1) std::vector<bool> is not thread safe due to the bit-casting.
  //!       (2) Using binary integer values allows us to compute batching indices
  //!           by computing partial sums over the vector values.
  std::vector<size_t> isActive_;
  size_t active_{0};
};

/*
 * Helper functions for vector construction
 */

template <class EnvironmentType_, typename... Arguments_>
auto make_vectorized(size_t size, Arguments_&&... args) {
  auto env = std::make_unique<Vector<typename EnvironmentType_::Scalar>>();
  env->template build<EnvironmentType_>(size, std::forward<Arguments_>(args)...);
  return env;
}

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_CORE_VECTOR_HPP_

/* EOF */
