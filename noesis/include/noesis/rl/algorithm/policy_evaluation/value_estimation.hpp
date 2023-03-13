/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_ALGORITHM_POLICY_EVALUATION_VALUE_ESTIMATION_HPP_
#define NOESIS_RL_ALGORITHM_POLICY_EVALUATION_VALUE_ESTIMATION_HPP_

// Noesis
#include <noesis/mdp/types.hpp>

namespace noesis {
namespace algorithm {

/*!
 * @brief An implementation of the TD(1) algorithm used for computing value estimates of a given policy.
 *
 *  This implementation estimates V*() using the TD(1) algorithm, which combines Monte-Carlo and n-step
 *  TD methods. Trajectories collected over a set of episodes provide the sample transitions for learning,
 *  but these can either be non-terminated episodes (i.e. they timed-out using a maximum number of steps)
 *  or they can terminated with respective terminal state and reward.
 *
 *  It is a combination between Monte-Carlo and n-step TD learning because the full sequence of observed
 *  rewards is used to estimate the return at each step, but the terminal value depends on whether the
 *  episode had terminated. If an episode terminated, then the terminal reward r[T] is used to compute
 *  the target values V*[t], otherwise the value function approximator V^(s[t]) is used to bootstrap.
 *
 *  Adopting the conventions from [1] regarding a unified notation for episodic and continuing tasks, we
 *  denote all step rewards resulting from transition tuples {s[t], a[t], s[t+1]} as r[t], with t in
 *  {0, ..., T-1}, and all experienced terminal rewards as r[T]. r[T] depends on the task, and can vary
 *  depending on how the episode terminates, i.e. for each task, r[T] is drawn from the discrete set R_T.
 *  Therefore `td1_value_estimates()` conditionally performs the following computations:
 *
 *  1. If r[T] is given, then: V*(s[t]) = r[t] + gamma * r[t+1] + ... + gamma^T * r[T]
 *
 *  2. Otherwise: V*(s[t]) = r[t] + gamma * r[t+1] + ... + gamma^(T-t-1) * gamma^T * V^(s[T])
 *
 *  [1] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. Vol.1.
 *      No.1. Cambridge: MIT press, 1998.
 *
 * @note This is implementation is designed to operate as a batched method.
 * @tparam ScalarType_ The fundamental arithmetic scalar type used for all numerical operations.
 * @param gamma
 * @param rewards
 * @param terminal_values
 * @param terminations
 * @param target_values
 */
template<typename ScalarType_>
static inline void td1_value_estimates(
    const ScalarType_ gamma,
    const noesis::Tensor<ScalarType_>& rewards,
    const noesis::Tensor<ScalarType_>& terminal_values,
    const noesis::mdp::Terminations<ScalarType_>& terminations,
    Tensor<ScalarType_>& target_values) {
  NFATAL_IF(rewards.dimensions().size() < 3,
    "Rewards tensor does not have sufficient dimensions: must be {1, time_size, batch_size}!");
  NFATAL_IF(rewards.batches() != terminal_values.size(),
    "Rewards and terminal values tensors do not have the same batch size!");
  NFATAL_IF(rewards.batches() != terminations.size(),
    "Rewards tensor and terminations do not have the same batch size!");
  // Extract sample and batch counts
  auto& batchSizes = rewards.timesteps();
  size_t numberOfBatches = batchSizes.size();
  // Count the total number of rewards from the each batch
  size_t number_of_rewards = std::accumulate(batchSizes.begin(), batchSizes.end(), static_cast<size_t>(0));
  // target_values is allocated to hold the total number of rewards over all batches
  target_values.resize({1, number_of_rewards, 1}, true);
  // timeIndex keeps track of time-steps along all the trajectories in target_values
  size_t sampleIdx = 0;
  // Iterate over all batches
  for (size_t batch = 0; batch < numberOfBatches; batch++) {
    size_t batchSize = batchSizes[batch];
    if (batchSize < 1) { continue; }
    // Set the offset of the current trajectory
    sampleIdx += batchSize - 1;
    // Internal buffer
    Tensor<ScalarType_> terminalValue({1}, false);
    // Set the terminal values - there are two cases:
    // 1. Trajectories which terminated received a terminal reward which is the exact value into the infinite horizon.
    // 2. Trajectories which timed-out did not experience a terminal state. The respective value is approximated using the value function.
    const auto& termination = terminations[batch];
    if (termination.type == noesis::mdp::Termination<ScalarType_>::Type::TerminalState) {
      terminalValue[0] = termination.value; // r[T]
    } else {
      terminalValue[0] = terminal_values[batch]; // V^(s[T])
    }
    // Iterate back from the final reward to the initial reward in the trajectory and compute target values
    // NOTE: the final reward not the terminal reward, it is the one received from the final transition.
    // t = T-1
    target_values[sampleIdx] = rewards(batchSize - 1, batch)[0] + gamma * terminalValue[0];
    // t = 0:T-2
    if (batchSize > 1) {
      for (int k = static_cast<int>(batchSize) - 2; k > -1; k--) {
        sampleIdx--;
        target_values[sampleIdx] = rewards(k, batch)[0] + gamma * target_values[sampleIdx + 1];
      }
    }
    // Progress sampleIdx to beyond the current trajectory
    sampleIdx += batchSize;
  }
  // NOTE: these checks are only enabled in debug builds to reduce overhead
  DNFATAL_IF(target_values.hasNaN(), "Target values contain NaN!");
  DNFATAL_IF(target_values.hasInf(), "Target values contain Inf!");
}

} // namespace algorithm
} // namespace noesis

#endif // NOESIS_RL_ALGORITHM_POLICY_EVALUATION_VALUE_ESTIMATION_HPP_

/* EOF */
