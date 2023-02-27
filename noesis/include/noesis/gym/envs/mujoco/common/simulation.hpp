/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_MUJOCO_COMMON_SIMULATION_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_COMMON_SIMULATION_HPP_

// C/C++
#include <memory>
#include <vector>

// Boost
#include <boost/thread/shared_mutex.hpp>

// Mujoco
#include <mujoco_cpp/mujoco_cpp.hpp>

// Noesis
#include <noesis/framework/log/message.hpp>

namespace noesis {
namespace gym {

typedef ::mujoco::SimulationConfig MujocoSimulationConfig;

/*!
 * @brief A wrapper of the mujoco::Simulation class enabling asynchronous operation
 * with the respective mujoco::Visualization wrapper. It also provides methods to reset and
 * step through the environment.
 *
 * .. note::
 *      The simulation class assumes that the state of the system can be described using the
 *      generalized coorindates (q) and velocities (u).
 */
class MujocoSimulation
{
public:

  /*
   * Instantiation
   */

  explicit MujocoSimulation(const MujocoSimulationConfig &config);

  ~MujocoSimulation();

  /*
   * Properties
   */

  /*!
   * @brief Retrieve the timestep at which the simulator runs.
   * @return The discretization time of the simulator physics.
   */
  double timestep() const {
    return timeStep_;
  }

  /*!
   * @brief Retrieve the phyics state of the simulation.
   * @return An instance of the object containing the physical state information.
   */
  const ::mujoco::State &state() const {
    return *state_;
  }

  /*!
  * @brief Retrieve the actuation instance of the simulation.
  * @return An instance of the object containing the physical input information.
  */
  const ::mujoco::Actuation &actuation() const {
    return *actuation_;
  }

  /*!
   * @brief Retrieves the current simulation walltime.
   */
  double time() const { return simulation_->getTime(); }

  /*!
   * @brief Retrieves the pointer to the  mujoco::Simulation instance being wrapped.
   * @return Raw pointer to the  mujoco::Simulation object.
   */
  const ::mujoco::Simulation *get() const { return simulation_.get(); }

  /*!
   * @brief Retrieves the pointer to the  mujoco::Simulation instance being wrapped.
   * @return Raw pointer to the  mujoco::Simulation object.
   */
  ::mujoco::Simulation *get() { return simulation_.get(); }

  /*!
   * @brief Indicates if the advance function is currently blocked waiting for external synchronization.
   * @note The only way to unblock is by calling release().
   */
  inline bool isWaiting() const { return isWaiting_; }

  /*
   * Visualization Operations
   */

  /*!
   * @brief Enables locking-mode asynchronous operation.
   */
  inline void lock() { useLocking_ = true; }

  /*!
   * @brief Disables locking-mode asynchronous operation.
   */
  inline void unlock() { useLocking_ = false; }

  /*!
   * @brief Releases the advance() function from a blocking wait.
   * @note Use isWaiting() to check if the thread calling advance() is currently blocked.
   */
  inline void release() { isLocked_ = false; }

  /*
   * Environment Operations
   */

  /*!
   * Resets the state of the system to given generalized coordinates and velocities.
   * @param q The generalized coordinates of the system.
   * @param u The generealized velocities of the system.
   */
  virtual void reset(const Eigen::VectorXd &q, const Eigen::VectorXd &u);

  /*!
   * Resets the state of the system to given state.
   * @param state The state of the system comprising of generalized coordinates and velocities.
   */
  virtual void reset(const Eigen::VectorXd &state);

  /*!
   * @brief Step through the simulation environment using joint torques command.
   * @param joint_commands The actuation torques applied to the system.
   * @return Boolean variable which states whether the transition is valid or not.
   */
  virtual bool step(const Eigen::VectorXd &joint_commands);

protected:

  /*!
   * @brief  Advance the simulation by a single time-step while using locking mechanism.
   */
  inline void advance();

private:
  //! A container for the mujoco simulation world (i.e. simulation instance).
  std::unique_ptr<::mujoco::Simulation> simulation_;
  //! Local handles to the model internal state buffer
  ::mujoco::State *state_;
  //! Local handles to the model internal actuation buffer
  ::mujoco::Actuation *actuation_;
  //! @brief The time-step of integration of the dynamics.
  double timeStep_{0.0};
  ///! Internal flag to synchronize execution with external caller.
  std::atomic_bool isLocked_{false};
  //! Internal flag indicating if the advance() method is currently blocked.
  std::atomic_bool isWaiting_{false};
  //! Internal flag indicating if external synchronization is enabled.
  std::atomic_bool useLocking_{false};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_COMMON_SIMULATION_HPP_

/* EOF */
