/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_ALGORITHM_POLICY_EVALUATION_GENERALIZED_ADVANTAGE_ESTIMATION_HPP_
#define NOESIS_RL_ALGORITHM_POLICY_EVALUATION_GENERALIZED_ADVANTAGE_ESTIMATION_HPP_

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/log/metric.hpp"
#include "noesis/framework/math/statistics.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"
#include "noesis/rl/memory/TrajectoryMemory.hpp"

namespace noesis {
namespace algorithm {

/*!
 * @brief An implementation of generalized advantage estimation by Schulman et al [1].
 *
 * [1] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel,
 *     "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
 *     arXiv:1506.02438, 2018
 */
template<typename ScalarType_>
class GeneralizedAdvantageEstimation: public ::noesis::core::Object
{
public:

  //! @brief Determines the type of advantage normalization to perform.
  enum class Normalization : int {
    None = 0,
    Trajectory,
    Batch
  };

  // Aliases
  using Base = ::noesis::core::Object;
  using Scalar = ScalarType_;
  using Tensor = ::noesis::Tensor<Scalar>;
  using Metrics = ::noesis::log::Metrics<Scalar>;
  using Termination = ::noesis::mdp::Termination<Scalar>;
  using TrajectoryMemory = ::noesis::memory::TrajectoryMemory<Scalar>;
  
  /*
   * Instantiation
   */

  GeneralizedAdvantageEstimation() = delete;
  
  GeneralizedAdvantageEstimation(GeneralizedAdvantageEstimation&& other) = default;
  GeneralizedAdvantageEstimation& operator=(GeneralizedAdvantageEstimation&& other) = default;
  
  GeneralizedAdvantageEstimation(const GeneralizedAdvantageEstimation& other) = delete;
  GeneralizedAdvantageEstimation& operator=(const GeneralizedAdvantageEstimation& other) = delete;
  
  explicit GeneralizedAdvantageEstimation(const std::string& scope, const std::string& name):
    Base(name, scope),
    discountFactor_(0.99, utils::make_namescope({scope, name, "discount_factor"}), {0.0, 1.0}),
    traceDecay_(0.97, utils::make_namescope({scope, name, "trace_decay"}), {0.0, 1.0}),
    normalization_("none", utils::make_namescope({scope, name, "normalization"}), {"none", "trajectory", "batch"})
  {
    // Add the hyper-parameters to the global manager
    hyperparam::manager->addParameter(discountFactor_);
    hyperparam::manager->addParameter(traceDecay_);
    hyperparam::manager->addParameter(normalization_);
  }

  ~GeneralizedAdvantageEstimation() {
    // Remove hyper-parameters from the global manager
    hyperparam::manager->removeParameter(discountFactor_);
    hyperparam::manager->removeParameter(traceDecay_);
    hyperparam::manager->removeParameter(normalization_);
  }

  /*
   * Configurations
   */

  

  void setDiscountFactor(Scalar gamma) {
    discountFactor_ = gamma;
  }
  
  void setTraceDecay(Scalar lambda) {
    traceDecay_ = lambda;
  }
  
  void setNormalizationMode(Normalization mode) {
    return normalization_ = normalizeModeToString(mode);
  }

  /*
   * Properties
   */

  Scalar discount_factor() const {
    return discountFactor_;
  }
  
  Scalar trace_decay() const {
    return traceDecay_;
  }
  
  Normalization normalization() {
    return normalizeModeFromString(normalization_);
  }
  
  const Metrics& metrics() const {
    return metrics_;
  }
  
  /*
   * Operations
   */

  void configure() {
    const auto ns = this->namescope();
    NINFO("[" << ns << "]: Discount factor: " << static_cast<Scalar>(discountFactor_));
    NINFO("[" << ns << "]: Trace decay: " << static_cast<Scalar>(traceDecay_));
    NINFO("[" << ns << "]: Normalization: " << static_cast<std::string>(normalization_));
    metrics_.clear();
    metrics_.push_back("GAE/advantage_mean");
    metrics_.push_back("GAE/advantage_stddev");
  }
  
  void initialize() {
    metrics_.reset();
  }
  
  void computeAdvantages(
      const TrajectoryMemory& trajectories,
      const Tensor& values,
      const Tensor& terminal_values,
      Tensor& advantages) {
    // Local copies of hyper-parameters
    Scalar gamma = discountFactor_;
    Scalar lambda = traceDecay_;
    Normalization normalizeMode = normalizeModeFromString(normalization_);
    // The advantages output buffer is allocated to hold the super-sequence of concatenated rewards trajectories over all batches
    advantages.resize({1, trajectories.getTotalTransitions(), 1}, true);
    // timeIndex keeps track of time-steps along all the trajectories in target_values
    size_t timeIndex = 0;
    for (size_t trajectoryIndex = 0; trajectoryIndex < trajectories.getNumberOfTrajectories(); trajectoryIndex++) {
      // Set the offset of the current trajectory
      size_t trajectoryLength = trajectories.getTrajectoryTransitions(trajectoryIndex);
      timeIndex += trajectoryLength - 1;
      // Internal buffers
      Tensor bellmanResidual({1}, false);
      Eigen::Matrix<Scalar, 1, -1> advantage(trajectoryLength);
      // Set the terminal values - there are two cases:
      // 1. Trajectories which terminated received a terminal reward which is the exact value into the infinite horizon.
      // 2. Trajectories which timed-out did not experience a terminal state. The respective value is approximated using the value function.
      const auto& termination = trajectories.getTrajectoryTermination(trajectoryIndex);
      if (termination.type == Termination::Type::TerminalState) {
        bellmanResidual[0] = trajectories.getRewards()(trajectoryLength - 1, trajectoryIndex)[0] +
                             gamma * termination.value - values[timeIndex];
      } else {
        bellmanResidual[0] = trajectories.getRewards()(trajectoryLength - 1, trajectoryIndex)[0] +
                             gamma * terminal_values[trajectoryIndex] - values[timeIndex];
      }
      // Iterate back from the terminal to the initial reward and compute TD(1) targets
      // t = T-1
      advantage[trajectoryLength - 1] = bellmanResidual[0];
      // t = 0:T-2
      if (trajectoryLength > 1) {
        for (int k = static_cast<int>(trajectoryLength) - 2; k > -1; k--) {
          timeIndex--;
          bellmanResidual[0] = trajectories.getRewards()(k, trajectoryIndex)[0] + gamma * values[timeIndex + 1] - values[timeIndex];
          advantage[k] = bellmanResidual[0] + lambda * gamma * advantage[k + 1];
        }
      }
      // Optionally normalize the advantage trajectory
      if (normalizeMode == Normalization::Trajectory) {
        auto stats = math::cwise_normalize(advantage);
        metrics_[AdvantageMean] = (stats.first + trajectoryIndex*metrics_[AdvantageMean])/(trajectoryIndex+1);
        metrics_[AdvantageStdDev] = (stats.second + trajectoryIndex*metrics_[AdvantageStdDev])/(trajectoryIndex+1);
      }
      // Append computed advantages to trajectory set
      advantages.asEigenMatrix().block(0, timeIndex, 1, trajectoryLength) = advantage;
      // Progress timeIndex to beyond the current trajectory
      timeIndex += trajectoryLength;
    }
    // Optionally normalize the advantage batch
    if (normalizeMode == Normalization::Batch) {
      auto stats = math::cwise_normalize(advantages);
      metrics_[AdvantageMean] = stats.first;
      metrics_[AdvantageStdDev] = stats.second;
    } else if (normalizeMode == Normalization::None) {
      auto moments = math::cwise_moments(advantages);
      metrics_[AdvantageMean] = moments.first;
      metrics_[AdvantageStdDev] = std::sqrt(moments.second);
    }
    // NOTE: these checks are only enabled in debug builds to reduce overhead
    DNFATAL_IF(advantages.hasNaN(), "[" << this->namescope() << "]: Advantages batch contains NaN!");
    DNFATAL_IF(advantages.hasInf(), "[" << this->namescope() << "]: Advantages batch contains Inf!");
  }

private:
  
  Normalization normalizeModeFromString(const std::string& mode_string) {
    Normalization mode = Normalization::None;
    if (mode_string == "none") {
      mode = Normalization::None;
    } else if (mode_string == "trajectory") {
      mode = Normalization::Trajectory;
    } else if (mode_string == "batch") {
      mode = Normalization::Batch;
    } else {
      NFATAL("[" << this->namescope() <<
        "]: Invalid value for normalization mode. Value must be one of {none, trajectory, batch}");
    }
    return mode;
  }

  std::string normalizeModeToString(Normalization mode) {
    std::string modeString;
    switch(mode) {
      case Normalization::None:
        modeString = "none";
        break;
      case Normalization::Trajectory:
        modeString = "trajectory";
        break;
      case Normalization::Batch:
        modeString = "batch";
        break;
    }
    return modeString;
  }

private:
  //! Defines indices for metrics collected by this class.
  enum Metric {
    AdvantageMean = 0,
    AdvantageStdDev
  };
  //! @brief Container of metrics recorded by this class.
  Metrics metrics_;
  //! @brief The reinforcement learning discount factor 'gamma',
  //! also referred to as the variance reduction parameter (Schulman 2016)
  hyperparam::HyperParameter<Scalar> discountFactor_;
  //! @brief The GAE trace-decay parameter 'lambda' (Sutton & Barto 2017)
  hyperparam::HyperParameter<Scalar> traceDecay_;
  //! @brief Sets the mode for internal normalization of each advantage trajectory
  hyperparam::HyperParameter<std::string> normalization_;
};

} // namespace algorithm
} // namespace noesis

#endif // NOESIS_RL_ALGORITHM_POLICY_EVALUATION_GENERALIZED_ADVANTAGE_ESTIMATION_HPP_

/* EOF */
