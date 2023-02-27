/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Environments
#include "noesis/gym/envs/mujoco/common/simulation.hpp"

namespace noesis {
namespace gym {

MujocoSimulation::MujocoSimulation(const mujoco::SimulationConfig &config) :
  state_(),
  actuation_() {
  // Construct the simulation instance if starting-up for the first time
  if (simulation_ == nullptr) {
    simulation_ = std::make_unique<::mujoco::Simulation>(config);
  }
  // Allocate and set-up the mujoco model
  simulation_->create();
  simulation_->reset();
  // If a custom time-step has been set, propagate the the mujoco simulation
  // otherwise retrieve the default specified in the MJCF
  if (timeStep_ != 0.0) {
    simulation_->setTimeStep(timeStep_);
  } else {
    timeStep_ = simulation_->getTimeStep();
  }
  // Configure pointers to buffers
  state_ = &simulation_->getState();
  actuation_ = &simulation_->getActuation();
}

MujocoSimulation::~MujocoSimulation() {
  if (simulation_ != nullptr) {
    simulation_->cleanup();
  }
  simulation_.reset(nullptr);
}

void MujocoSimulation::reset(const Eigen::VectorXd &q, const Eigen::VectorXd &u) {
  NFATAL_IF(q.size() + u.size() != state_->state_dims(),
            "State dimensions mismatch. Expected " << state_->state_dims() << " but got " << q.size() + u.size() << ".")
  state_->setGeneralizedCoordinates(q);
  state_->setGeneralizedVelocities(u);
  simulation_->resetFromBuffer();
}

void MujocoSimulation::reset(const Eigen::VectorXd &state) {
  NFATAL_IF(state.size() != state_->state_dims(),
            "State dimensions mismatch. Expected " << state_->state_dims() << " but got " << state.size() << ".")
  state_->setGeneralizedCoordinates(state.head(state_->q_dims()));
  state_->setGeneralizedVelocities(state.tail(state_->u_dims()));
  simulation_->resetFromBuffer();
}

bool MujocoSimulation::step(const Eigen::VectorXd &joint_commands) {
  NFATAL_IF(joint_commands.size() != state_->input_dims(),
            "Input actuation command should be of the same size as number of joints in "
            "the system. Expected " << state_->input_dims() << " but got " << joint_commands.size() << ".");
  bool stepIsValid = true;
  // Set the actuation commands
  actuation_->setJointEffort(joint_commands);
  // Integrate physics
  this->advance();
  // Update state (inside simulation wrapper)
  state_->updateBase();
  state_->updateJoints();
  state_->updateBodies();
  state_->updateContacts();
  // Data validity checks
  if (!state_->getGeneralizedCoordinates().allFinite()) { stepIsValid = false; }
  if (!state_->getGeneralizedVelocities().allFinite()) { stepIsValid = false; }
  return stepIsValid;
}

/*!
 * @brief  Advance the simulation by a single time-step.
 */
void MujocoSimulation::advance() {
  // Block on waiting if external synchronization is enabled
  if (useLocking_) {
    isWaiting_ = true;
    while (isLocked_);
    isWaiting_ = false;
    isLocked_ = true;
  }
  simulation_->advance();
}

} // namespace gym
} // namesapce noesis

/* EOF */
