/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_CORE_ENVIRONMENT_HPP_
#define NOESIS_GYM_CORE_ENVIRONMENT_HPP_

// Noesis
#include "noesis/framework/utils/macros.hpp"
#include "noesis/framework/utils/string.hpp"
#include "noesis/framework/log/message.hpp"
#include "noesis/framework/core/Object.hpp"
#include "noesis/mdp/EnvironmentInterface.hpp"

namespace noesis {
namespace gym {

template <typename ScalarType_>
class Environment:
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
  
  //! @note Move and move-assignment construction is permissible for this class.
  Environment(Environment&& other) noexcept = default;
  Environment& operator=(Environment&& other) noexcept = default;
  
  //! @note Copy and copy-assignment construction should not be permitted.
  //! @note This behavior is already enforced by using std::unique_ptr to wrap the session pointer.
  Environment(const Environment& other) = delete;
  Environment& operator=(const Environment& other) = delete;
  
  //! @brief Constructor for discrete-time problems.
  explicit Environment(
      const TensorsSpec& observations,
      const TensorsSpec& actions,
      const Names& rewards={"total"},
      const Names& metrics={},
      Scalar discount_factor=1.0,
      size_t max_steps=std::numeric_limits<size_t>::max(),
      size_t batch_size=1u,
      size_t history_size=1u,
      const std::string& name="Environment",
      const std::string& scope="/",
      bool verbose=false):
    Object(name, scope, verbose),
    Interface(),
    observations_(utils::make_namescope({scope, name, "observations"}), observations, history_size, batch_size),
    actions_(utils::make_namescope({scope, name, "actions"}), actions, history_size, batch_size),
    rewards_(utils::make_namescope({scope, name, "rewards"}), {rewards.size(), history_size, batch_size}, true),
    metrics_(metrics),
    tasks_(rewards),
    gamma_(discount_factor),
    maxSteps_(max_steps),
    terminations_((batch_size>1) ? batch_size+1 : batch_size)
  {
    NNOTIFY_IF(verbose, "[Environment]: New instance at: " << std::hex << this);
  }
  
  //! @brief Constructor for discretized continuous-time problems.
  explicit Environment(
      const TensorsSpec& observations,
      const TensorsSpec& actions,
      const Names& rewards={"total"},
      const Names& metrics={},
      Scalar discount_factor=1.0,
      Scalar time_limit=std::numeric_limits<Scalar>::max(),
      Scalar time_step=1.0,
      size_t batch_size=1u,
      size_t history_size=1u,
      const std::string& name="Environment",
      const std::string& scope="/",
      bool verbose=false):
    Object(name, scope, verbose),
    Interface(),
    observations_(utils::make_namescope({scope, name, "observations"}), observations, history_size, batch_size),
    actions_(utils::make_namescope({scope, name, "actions"}), actions, history_size, batch_size),
    rewards_(utils::make_namescope({scope, name, "rewards"}), {rewards.size(), history_size, batch_size}, true),
    terminations_((batch_size>1) ? batch_size+1 : batch_size),
    metrics_(metrics),
    tasks_(rewards),
    gamma_(discount_factor),
    dt_(time_step)
  {
    NNOTIFY_IF(verbose, "[Environment]: New instance at: " << std::hex << this);
    setTimeLimit(time_limit);
  }
  
  virtual ~Environment() = default;
  
  /*
   * Configurations
   */
  
  void setDiscountFactor(Scalar gamma) { gamma_ = gamma; }
  
  void setTimeStep(Scalar dt) { dt_ = dt; }
  
  void setTimeLimit(Scalar tmax) { maxSteps_ = static_cast<size_t>(tmax/dt_); }
  
  void setMaxSteps(size_t max) { maxSteps_ = max; }
  
  /*
   * Properties
   */
  
  size_t batch_size() const override { return observations_.batches(); }
  
  size_t history_size() const override { return observations_.timesteps(); }
  
  size_t max_steps() const override { return maxSteps_; }
  
  Scalar time_limit() const override { return static_cast<Scalar>(maxSteps_)*dt_; }
  
  Scalar time_step() const override { return dt_; }
  
  Scalar discount_factor() const override { return gamma_; }
  
  TensorsSpec actions_spec() const override { return actions_.spec(); }
  
  TensorsSpec observations_spec() const override { return observations_.spec(); }
  
  Names tasks() const override { return tasks_; }
  
  Actions& actions() override { return actions_; }
  
  const Actions& actions() const override { return actions_; }
  
  const Observations& observations() const override { return observations_; }
  
  const Rewards& rewards() const override { return rewards_; }
  
  const Terminations& terminations() const override { return terminations_; }
  
  const Metrics& metrics() const override { return metrics_; }
  
  size_t steps() const override { return steps_; }
  
  Scalar time() const override { return time_; }
  
  /*
   * Operations
   */
  
  void configure() override {
    setup();
  }
  
  void seed(int seed) override {
    randomize(seed);
  }
  
  void reset() override {
    steps_ = 0;
    time_ = 0.0;
    rewards_.setZero();
    auto& trm = terminations_.back().type;
    trm = Termination::Type::Unterminated;
    do { initialize(observations_, terminations_); } while(trm != Termination::Type::Unterminated);
  }
  
  void step() override {
    steps_++;
    const bool isValid = transition(actions_, observations_, rewards_, terminations_, metrics_);
    time_ += dt_;
    auto& trm = terminations_.back().type;
    auto& id = terminations_.back().id;
    if (steps_ >= maxSteps_ && trm == Termination::Type::Unterminated) { trm = Termination::Type::TimeOut; }
    // TODO: @vt: Decide on flexible way to optionally enable these
    if (!isValid) {
      trm = Termination::Type::InvalidState;
      id = static_cast<int>(trm);
    } else if (!rewards_.allFinite()) {
      trm = Termination::Type::InvalidReward;
      id = static_cast<int>(trm);
    } else if (!actions_.allFinite()) {
      trm = Termination::Type::InvalidAction;
      id = static_cast<int>(trm);
    } else if (!observations_.allFinite()) {
      trm = Termination::Type::InvalidObservation;
      id = static_cast<int>(trm);
    }
  }
  
  std::ostream& info(std::ostream& os) const override {
    os << "\n[noesis::gym::Environment]:";
    os << "\nAddress: " << std::hex << this;
    os << "\nType: " << utils::typename_to_string(*this);
    os << "\nName: " << namescope();
    os << "\nActions: " << actions_spec();
    os << "\nObservations: " << observations_spec();
    os << "\nTasks: " << utils::vector_to_string(tasks());
    os << "\nMetrics: " << utils::vector_to_string(metrics().names());
    os << "\nSize {Batch, History}: {" << batch_size() << ", " << history_size() << "}";
    os << "\nTime-Step: " << time_step();
    os << "\nTime-Limit: " << time_limit();
    os << "\nTime: " << time();
    os << "\nSteps: " << steps();
    os << "\nTermination: " << terminations().back();
    os << "\nActions: " << actions_;
    os << "\nObservations: " << observations_;
    os << "\nRewards: " << rewards_;
    os << "\nMetrics: " << metrics_.info();
    os << "\nInfo: " << info();
    return os;
  }

protected:
  
  /*
   * Implementations
   */
  
  virtual void setup() {}
  
  virtual void randomize(int seed) { UNUSED_VARIABLE(seed); }
  
  virtual bool initialize(Observations& observations, Terminations& terminations) = 0;
  
  virtual bool transition(
    const Actions& actions,
    Observations& observations,
    Rewards& rewards,
    Terminations& terminations,
    Metrics& metrics
  ) = 0;
  
  virtual std::string info() const { return ""; }
  
private:
  Observations observations_;
  Actions actions_;
  Rewards rewards_;
  Metrics metrics_;
  Names tasks_;
  Scalar gamma_{1.0};
  Scalar dt_{1.0};
  Scalar time_{0.0};
  size_t steps_{0};
  size_t maxSteps_{std::numeric_limits<size_t>::max()};
  Terminations terminations_{Termination()};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_CORE_ENVIRONMENT_HPP_

/* EOF */
