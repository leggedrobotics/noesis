/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_WORLD_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_WORLD_HPP_

// C/C++
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

// Boost
#include <boost/thread/shared_mutex.hpp>

// Environments
#include "noesis/gym/envs/raisim/common/raisim.hpp"

namespace noesis {
namespace gym {

/*!
 * @brief A simple wrapper of the raisim::World class enabling asynchronous operation with the respective raisim::OgreVis wrapper.
 */
class RaiSimWorld
{
public:

  /*
   * Instantiation
   */
  
  RaiSimWorld() = default;

  virtual ~RaiSimWorld() = default;
  
  /*
   * Properties
   */

  /*!
   * @brief Retrieves the pointer to the raisim::World instance being wrapped.
   * @return Raw pointer to the raisim::World object.
   */
  const ::raisim::World* get() const { return world_.get(); }
  
  /*!
   * @brief Retrieves the pointer to the raisim::World instance being wrapped.
   * @return Raw pointer to the raisim::World object.
   */
  ::raisim::World* get() { return world_.get(); }
  
  /*!
   * @brief Retrieves the current simulation time.
   */
  double time() const { return world_->getWorldTime(); }
  
  /*!
   * @brief Retrieves the configured simulation time-step.
   */
  double timestep() const { return world_->getTimeStep(); }
  
  /*
   * Operations
   */

  /*!
   * @brief Overloading of the de-referencing operator for direct access of members of the underlying raisim::World instance.
   * @return Raw pointer to the raisim::World instance being wrapped.
   */
  const ::raisim::World* operator->() const {
    return world_.get();
  }
  
  /*!
   * @brief Overloading of the dereferencing operator for direct access of members of the underlying raisim::World instance.
   * @return Raw pointer to the raisim::World instance being wrapped.
   */
  ::raisim::World* operator->() {
    return world_.get();
  }
  
  /*!
   * @brief Constructs the instance
   */
  void create() { world_ = std::make_unique<::raisim::World>(); }
  
  /*!
   * @brief Destructs the instance
   */
  void destroy() { world_.reset(nullptr); }
  
  /*!
   * @brief Enables asynchronous operation.
   */
  void lock() {
    useLocking_ = true;
    shouldRun_.store(false);
  }
  
  /*!
   * @brief Disables asynchronous operation.
   */
  void unlock() {
    useLocking_ = false;
    shouldRun_.store(false);
    cv_.notify_one();
  }
  
  /*!
   * @brief Triggers execution of integrate() or integrate1() if blocked on waiting.
   * @note Use this to synchronize with the thread calling the integration methods.
   */
  void start() {
    cv_.notify_one();
  }
  
  /*!
   * @brief Waits until the synchronization call from integrate() or integrate1().
   * @note Use this to synchronize with the thread calling the integration methods.
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{return !this->shouldRun_.load();});
    shouldRun_.store(true);
  }
  
  /*!
   * @brief Performs collision detection and updates of the forward kinematics and dynamics properties.
   */
  void integrate() {
    // Wait for external synchronization if enabled.
    if (useLocking_) { sync(); }
    world_->integrate();
  }
  
  /*!
   * @brief Performs collision detection and updates of the forward kinematics and dynamics properties.
   */
  void integrate1() {
    // Wait for external synchronization if enabled.
    if (useLocking_) { sync(); }
    world_->integrate1();
  }
  
  /*!
   * @brief Computes solution of the physics for one time-step and performs forward integration.
   */
  inline void integrate2() { world_->integrate2(); }

private:

  inline void sync() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.notify_one();
    cv_.wait(lock, [this]{return this->shouldRun_.load();});
    shouldRun_.store(false);
  }

private:
  //! @brief A container for the raisim simulation world (i.e. physics engine).
  std::unique_ptr<::raisim::World> world_;
  //! @brief Condition variable synchronizing the renderer with integration of the world.
  std::condition_variable cv_;
  //! @brief Mutex to protect access to the condition variable member.
  std::mutex mutex_;
  //! @brief Internal flag indicating if external synchronization is enabled.
  std::atomic_bool useLocking_{false};
  //! @brief Internal flag signaling the state of the renderer-world synchronization.
  std::atomic_bool shouldRun_{false};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_WORLD_HPP_

/* EOF */
