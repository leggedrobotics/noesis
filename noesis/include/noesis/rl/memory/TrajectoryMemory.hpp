/*!
 * @author  Vassilios Tsounis
 * @email   tsounisv@ethz.ch
 * @author  David Hoeller
 * @email   dhoeller@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_MEMORY_TRAJECTORY_MEMORY_HPP_
#define NOESIS_RL_MEMORY_TRAJECTORY_MEMORY_HPP_

// C/C++
#include <algorithm>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"
#include "noesis/mdp/types.hpp"

namespace noesis {
namespace memory {

/*!
 * @brief Specialized configuration type for the TrajectoryMemory class.
 */
struct TrajectoryMemoryConfig {
  TensorsSpec observations_spec;
  TensorsSpec actions_spec;
  std::string name{"Memory"};
  std::string scope{"/"};
  size_t number_of_instances{1};
  size_t max_trajectory_length{1};
  size_t number_of_tasks{1};
  int initial_capacity{1};
  bool use_adaptive_capacity{false};
  bool verbose{false};
};

/*!
 * @brief
 * @tparam ScalarType_
 */
template<typename ScalarType_>
class TrajectoryMemory final: public core::Object
{
public:
  
  // Ensure that ScalarType_ is one of the supported types
  static_assert(
    std::is_arithmetic<ScalarType_>::value,
    "ScalarType_ must be an arithmetic type, e.g. {int, float, double, etc..}"
  );

  // Aliases
  using Object = ::noesis::core::Object;
  using Scalar = ScalarType_;
  using Observations = mdp::Observations<Scalar>;
  using Actions = mdp::Actions<Scalar>;
  using Rewards = mdp::Rewards<Scalar>;
  using Termination = mdp::Termination<Scalar>;
  using Terminations = mdp::Terminations<Scalar>;
  using DimensionsVectorType = std::vector<std::vector<size_t>>;
  
  // Constants
  static constexpr int MaxInt = std::numeric_limits<int>::max();
  
  /*
   * Instantiation
   */
  
  explicit TrajectoryMemory(const TrajectoryMemoryConfig& config):
    Object(config.name, config.scope, config.verbose),
    rewards_("rewards", {1,1,1}, true),
    flattenedRewards_("flat/rewards", {1,1,1}, true),
    observations_("observations", 1, 1),
    terminalObservations_("terminal_observations", 1, 1),
    flattenedObservations_("flat/observations", 1, 1),
    flattenedTerminalObservations_("flat/terminal_observations", 1, 1),
    actions_("actions", 1, 1),
    flattenedActions_("flat/actions", 1, 1),
    initialCapacity_(config.initial_capacity, utils::make_namescope({config.scope, config.name, "initial_capacity"}), {0, MaxInt}),
    useAdaptiveCapacity_(config.use_adaptive_capacity, utils::make_namescope({config.scope, config.name, "adaptive_capacity"})),
    observationsSpec_(config.observations_spec),
    actionsSpec_(config.actions_spec),
    instanceTrajectoryIndex_(config.number_of_instances, 0),
    maxTrajectoryLength_(config.max_trajectory_length),
    numberOfTasks_(config.number_of_tasks)
  {
    // Add hyper-parameters to the global manager
    noesis::hyperparam::manager->addParameter(initialCapacity_);
    noesis::hyperparam::manager->addParameter(useAdaptiveCapacity_);
  }
  
  ~TrajectoryMemory() override {
    // Remove hyper-parameters from the global manager
    noesis::hyperparam::manager->removeParameter(initialCapacity_);
    noesis::hyperparam::manager->removeParameter(useAdaptiveCapacity_);
  }

  /*
   * Configurations
   */

  void setObservationsSpec(const TensorsSpec& spec) {
    observationsSpec_ = spec;
  }
  
  void setActionsSpec(const TensorsSpec& spec) {
    actionsSpec_ = spec;
  }

  void setMaxTrajectoryLength(size_t max) {
    maxTrajectoryLength_ = max;
  }
  
  void setNumberOfTasks(size_t tasks) {
    numberOfTasks_ = tasks;
  }
  
  void setInitialCapacity(size_t capacity) {
    initialCapacity_ = static_cast<int>(capacity);
  }
  
  void setUseAdaptiveCapacity(bool enabled) {
    useAdaptiveCapacity_ = enabled;
  }
  
  /*
   * Properties
   */
  
  /*!
   * @brief Retrieves the maximum trajectory length supported by the memory instance.
   * @return Maximum length configured for trajectories.
   */
  size_t getMaxTrajectoryLength() const {
    return maxTrajectoryLength_;
  }
  
  /*!
   * @brief Retrieves the maximum number of trajectories the current memory allocation supports.
   * @return Trajectory capacity.
   */
  size_t getTrajectoryCapacity() const {
    return terminations_.capacity();
  }
  
  /*!
   * @brief Retrieves the number of trajectories currently in memory.
   * @return The number of trajectories currently in memory.
   */
  size_t getNumberOfTrajectories() const {
    return trajectoryCounter_;
  }
  
  /*!
   * @brief Retrieves the number of trajectories currently in memory.
   * @return The number of trajectories currently in memory.
   */
  size_t getTotalTransitions() const {
    return transitionsCounter_;
  }
  
  /*!
   * @brief Returns the number of simultaneous trajectory intances for which
   *        the memory container has been configured to support.
   * @return Number of simultaneous trajectories supported by the memory.
   */
  size_t getNumberOfInstances() const {
    return instanceTrajectoryIndex_.size();
  }
  
  /*!
   * @brief Retrieves the absolute trajectory index that the instance is currently assigned to.
   * @param instance The instance index. This must be less than the number of supported instances.
   * @return The absolute trajectory index for the target instance.
   */
  size_t getTrajectoryIndex(size_t instance) const {
    DNFATAL_IF(instance >= instanceTrajectoryIndex_.size(), "[" << this->namescope() <<
      "]: 'instance' argument must not exceed total number of instances! ('" << instance <<
      "' vs '" << instanceTrajectoryIndex_.size() << "')!");
    return instanceTrajectoryIndex_[instance];
  }
  
  /*!
   * @brief  Get the number of transitions of an existing trajectory at a specific index.
   * @param index The index of the target trajectory.
   * @return The length of the trajectory in terms of the total transitions.
   */
  size_t getTrajectoryTransitions(size_t index) const {
    DNFATAL_IF(index >= rewards_.batches(), "[" << namescope() << "]: Index '" << index
      << "' is invalid. Total number of trajectories is '" << rewards_.batches() << "'.");
    return rewards_.timesteps()[index];
  }
  
  /*!
   * @brief  Get the length of an existing trajectory at a specific index.
   * @param index The index of the target trajectory.
   * @return The length of the trajectory in terms of the total time-steps.
   */
  size_t getTrajectoryLength(size_t index) const {
    DNFATAL_IF(index >= observations_[0].batches(), "[" << namescope() << "]: Index '" << index
      << "' is invalid. Total number of trajectories is '" << observations_[0].batches() << "'.");
    return observations_[0].timesteps()[index];
  }
  
  /*!
  * @brief Retrieve the termination state of a trajectory.
  * @return Termination state of the trajectory.
  */
  Termination getTrajectoryTermination(size_t index) const {
    DNFATAL_IF(index >= observations_[0].batches(), "[" << namescope() << "]: Index '" << index
      << "' is invalid. Total number of trajectories is '" << observations_[0].batches() << "'.");
    return terminations_[index];
  }
  
  /*!
  * @brief Retrieve the batched trajectories of rewards.
  * @return Vector of Tensors type containing collected rewards.
  */
  const Rewards& getRewards() const {
    return rewards_;
  }
  
  /*!
   * @brief Retrieve the batched trajectories of observations. Does not include the terminal observations
   * @return Vector of Tensors type containing collected observations.
   */
  const Observations& getObservations() const {
    return observations_;
  }
  
  /*!
   * @brief Retrieve the terminal observations oh each trajectory.
   * @return Vector of Tensors type containing flattened observations.
   */
  const Observations& getTerminalObservations() const {
    return terminalObservations_;
  }
  
  /*!
   * @brief Retrieve the batched trajectories of actions.
   * @return Vector of Tensors type containing collected actions.
   */
  const Actions& getActions() const {
    return actions_;
  }
  
  /*!
   * @brief Retrieve the termination states of all trajectories
   * @return Termination state of the trajectory.
   */
  const Terminations& getTerminations() const {
    return terminations_;
  }
  
  /*!
   * @brief Retrieve the flattened trajectories of observations. Does not include the terminal observations
   * @return Vector of Tensors type containing flattened observations.
   */
  Rewards& getFlattenedRewards() {
    return flattenedRewards_;
  }
  
  /*!
   * @brief Retrieve the flattened trajectories of observations. Does not include the terminal observations
   * @return Vector of Tensors type containing flattened observations.
   */
  Observations& getFlattenedObservations() {
    return flattenedObservations_;
  }
  
  /*!
   * @brief Retrieve the terminal observations oh each trajectory.
   * @return Vector of Tensors type containing flattened observations.
   */
  Observations& getFlattenedTerminalObservations() {
    return flattenedTerminalObservations_;
  }
  
  /*!
   * @brief Retrieve the flattened trajectories of actions.
   * @return Vector of Tensors type containing flattened actions.
   */
  Actions& getFlattenedActions() {
    return flattenedActions_;
  }
  
  /*
   * Operations
   */

  /*!
   * @brief
   */
  void configure() {
    const auto ns = namescope();
    NINFO("[" << ns << "]: Configuring memory ...");
    NINFO("[" << ns << "]: Max trajectory length: " << maxTrajectoryLength_);
    NINFO("[" << ns << "]: Initial capacity: " << static_cast<int>(initialCapacity_));
    NINFO("[" << ns << "]: Using adaptive capacity: " << std::boolalpha << static_cast<bool>(useAdaptiveCapacity_));
    // Configure the internal containers
    actions_.setFromSpec(actionsSpec_);
    flattenedActions_.setFromSpec(actionsSpec_);
    observations_.setFromSpec(observationsSpec_);
    terminalObservations_.setFromSpec(observationsSpec_);
    flattenedObservations_.setFromSpec(observationsSpec_);
    flattenedTerminalObservations_.setFromSpec(observationsSpec_);
  }

  /*!
   * @brief Clears the contents of the memory.
   * @warning Memory is reset to the max time-steps and number of instances.
   */
  void reset() {
    const auto ns = namescope();
    NFATAL_IF(maxTrajectoryLength_ < 2, "[" << ns << "]: Maximum trajectory length has not been configured!");
    // Define local static variables
    static size_t prevInitialCapacity = 0;
    static size_t prevMaxNumberOfTimeSteps = 0;
    // Retrieve configurations
    auto initialCapacity = static_cast<size_t>(initialCapacity_);
    // Change memory allocation
    if ( (prevInitialCapacity != initialCapacity) || (prevMaxNumberOfTimeSteps != maxTrajectoryLength_) ) {
      // Reset trajectory management members
      terminations_.reserve(initialCapacity);
      // Reset scalar containers
      rewards_.reserve({numberOfTasks_, maxTrajectoryLength_, initialCapacity});
      // Reset observations and actions containers
      observations_.reserve(maxTrajectoryLength_+1, initialCapacity);
      terminalObservations_.reserve(1, initialCapacity);
      actions_.reserve(maxTrajectoryLength_, initialCapacity);
      // Update the local capacities
      prevInitialCapacity = initialCapacity;
      prevMaxNumberOfTimeSteps = maxTrajectoryLength_;
      // Verbose output (conditional)
      NINFO_IF(this->isVerbose(), "[" << ns << "]: Re-allocation of memory: New capacities are:"
        << "\n  Terminations: " << terminations_.capacity()
        << "\n  Rewards: " << utils::vector_to_string(rewards_.capacities())
        << "\n  Observations: " << utils::vector_to_string(observations_.capacities())
        << "\n  Terminal observations: " << utils::vector_to_string(terminalObservations_.capacities())
        << "\n  Actions: " << utils::vector_to_string(actions_.capacities()));
    }
    // Reset trajectory management members
    trajectoryCounter_ = 0;
    transitionsCounter_ = 0;
    terminations_.clear();
    isFlattened_ = false;
    // Clear scalar containers
    rewards_.clear();
    flattenedRewards_.clear();
    // Clear observations containers
    observations_.clear();
    terminalObservations_.clear();
    flattenedObservations_.clear();
    flattenedTerminalObservations_.clear();
    // Clear actions containers
    actions_.clear();
    flattenedActions_.clear();
  }
  
  void initializeTrajectory(size_t instance, const Observations& observations) {
    DNFATAL_IF(instance >= instanceTrajectoryIndex_.size(),
      "[" << this->namescope() << "]: 'instance' argument must not exceed total number of instances! ('"
      << instance << "' vs '" << instanceTrajectoryIndex_.size() << "')!");
    // Assign a new batch index to the instance for a subsequent trajectory
    instanceTrajectoryIndex_[instance] = trajectoryCounter_++;
    // Initialize the trajectory's termination state
    terminations_.push_back(Termination());
    // Set the initial observations (state) of the trajectory
    for (size_t k = 0; k < observations_.size(); k++) {
      observations_[k].pushBack(instanceTrajectoryIndex_[instance], observations[k](0, 0));
    }
    DNFATAL_IF(observations_[0].timesteps()[instanceTrajectoryIndex_[instance]] > 1,
      "[" << this->namescope() << "]: Cannot initialize non-empty trajectory at index '"
      << instanceTrajectoryIndex_[instance] << "'!");
  }
  
  void addTransition(
    size_t instance,
    const Actions& actions,
    const Observations& observations,
    const Rewards& rewards
  ) {
    DNFATAL_IF(instance >= instanceTrajectoryIndex_.size(), "[" << this->namescope() <<
      "]: 'instance' argument must not exceed total number of instances! ('" << instance <<
      "' vs '" << instanceTrajectoryIndex_.size() << "')!");
    // Sample actions a[t]
    for (size_t k = 0; k < actions_.size(); k++) {
      actions_[k].pushBack(instanceTrajectoryIndex_[instance], actions[k](0, 0));
    }
    // Next observations o[t+1]
    for (size_t k = 0; k < observations_.size(); k++) {
      observations_[k].pushBack(instanceTrajectoryIndex_[instance], observations[k](0, 0));
    }
    // Step rewards r[t]
    rewards_.pushBack(instanceTrajectoryIndex_[instance], rewards(0, 0));
    // Increment the transitions counter
    transitionsCounter_++;
  }
  
  void terminateTrajectory(size_t instance, const Observations& observations, const Termination& termination) {
    DNFATAL_IF(instance >= instanceTrajectoryIndex_.size(), "[" << this->namescope()
      << "]: 'instance' argument must not exceed total number of instances! ('"
      << instance << "' vs '" << instanceTrajectoryIndex_.size() << "')!");
    // Append terminal observations
    for (size_t k = 0; k < terminalObservations_.size(); k++) {
      terminalObservations_[k].pushBack(instanceTrajectoryIndex_[instance], observations[k](0, 0));
    }
    // Set the termination type
    terminations_[instanceTrajectoryIndex_[instance]] = termination;
  }
  
  void restartTrajectory(size_t instance, const Observations& observations) {
    DNFATAL_IF(instance >= instanceTrajectoryIndex_.size(),
      "[" << this->namescope() << "]: 'instance' argument must not exceed total number of instances! ('"
      << instance << "' vs '" << instanceTrajectoryIndex_.size() << "')!");
    NWARNING_IF(this->isVerbose(),
      "[" << namescope() << "]: Instance '" << instance << "': Restarting trajectory '"
      << instanceTrajectoryIndex_[instance] << "'");
    // Ger current length
    auto transitions = getTrajectoryTransitions(instanceTrajectoryIndex_[instance]);
    transitionsCounter_ -= transitions;
    // Set the termination type
    terminations_[instanceTrajectoryIndex_[instance]] = Termination();
    // Actions samples a[t]
    for (size_t k = 0; k < actions_.size(); k++) {
      actions_[k].popBackTimeSteps(instanceTrajectoryIndex_[instance], transitions);
    }
    // Next observations o[t+1]
    for (size_t k = 0; k < observations_.size(); k++) {
      observations_[k].popBackTimeSteps(instanceTrajectoryIndex_[instance], transitions+1);
    }
    // Step rewards r[t]
    rewards_.popBackTimeSteps(instanceTrajectoryIndex_[instance], transitions);
    // Set the initial observations (state) of the trajectory
    for (size_t k = 0; k < observations_.size(); k++) {
      observations_[k].pushBack(instanceTrajectoryIndex_[instance], observations[k](0, 0));
    }
  }
  
  /*!
   * @brief Flattens the memory along the batch dimension so that all trajectories are concatenated along contiguous memory regions.
   */
  void flatten() {
    NFATAL_IF(isFlattened_, "[" << this->namescope() << "]: Memory has already been flattened!");
    // Truncate the last observations samples as they are terminal
    for (size_t k = 0; k < observations_.size(); k++) {
      observations_[k].popBackTimeSteps(1);
    }
    // Flatten all trajectories into contiguous memory while retaining only valida data, i.e. truncate empty allocations.
    flattenedRewards_ = rewards_.getFlattenedBatches();
    for (size_t k = 0; k < flattenedObservations_.size(); k++) {
      flattenedObservations_[k] = observations_[k].getFlattenedBatches();
      flattenedTerminalObservations_[k] = terminalObservations_[k].getFlattenedBatches();
    }
    for (size_t k = 0; k < flattenedActions_.size(); k++) {
      flattenedActions_[k] = actions_[k].getFlattenedBatches();
    }
    // Set the internal flag to prevent multiple invocations to re-flatten the memory
    isFlattened_ = true;
  }
  
  /*!
   * @brief Helper function for verbose output.
   * @param os The target output stream to which the output is written.
   * @param memory The memory instance.
   * @return The modified output stream object.
   */
  friend std::ostream& operator<<(std::ostream& os, const TrajectoryMemory<Scalar>& memory) {
    if (memory.getNumberOfTrajectories() > 0) {
      os << "[" << memory.namescope() << "]: Memory has '" << memory.getNumberOfTrajectories() << "' trajectories:\n";
      for (size_t i=0; i<memory.getNumberOfTrajectories(); i++) {
        os << "  '" << i << "' has length: " << memory.getTrajectoryLength(i) << "\n";
      }
    } else {
      os << "[" << memory.namescope() << "]: Memory does not contain any trajectories.\n";
    }
    return os;
  }
  
private:
  //! Stores the rewards trajectory.
  Rewards rewards_;
  //! Stores the rewards trajectory.
  Rewards flattenedRewards_;
  //! Stores the trajectory for each observation.
  Observations observations_;
  //! Stores the flattened trajectory for each observation.
  Observations terminalObservations_;
  //! Stores the flattened trajectory for each observation.
  Observations flattenedObservations_;
  //! Stores the flattened reduced terminal observations.
  Observations flattenedTerminalObservations_;
  //! Stores the trajectory of each action.
  Actions actions_;
  //! Stores the flattened trajectory of each action.
  Actions flattenedActions_;
  //! Initial capacity in number of trajectories, is used to (re)initialize the memory allocation.
  noesis::hyperparam::HyperParameter<int> initialCapacity_;
  //! Enables adaptive capacity which optimizes the current capacity based on previous usage.
  noesis::hyperparam::HyperParameter<bool> useAdaptiveCapacity_;
  //! Stores the tensor specifications for observations
  TensorsSpec observationsSpec_;
  //! Stores the tensor specifications for actions
  TensorsSpec actionsSpec_;
  //! Stores the termination state at the end of a trajectory.
  Terminations terminations_;
  //! Stores the current batch index for each instance.
  std::vector<size_t> instanceTrajectoryIndex_;
  //! Thread-safe counter of the total number of trajectories currently held in the memory.
  std::atomic_size_t trajectoryCounter_{0};
  //! Thread-safe counter of the current total number of transition samples over all current trajectories held in the memory.
  std::atomic_size_t transitionsCounter_{0};
  //! The number of time steps to reserve for each trajectory.
  size_t maxTrajectoryLength_{0};
  //! The number of time steps to reserve for each trajectory.
  size_t numberOfTasks_{1};
  //! Flag to indicate if the memory has been flattened.
  bool isFlattened_{false};
};

} // namespace memory
} // namespace noesis

#endif // NOESIS_RL_MEMORY_TRAJECTORY_MEMORY_HPP_

/* EOF */
