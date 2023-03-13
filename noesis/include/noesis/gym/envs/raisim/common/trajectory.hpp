/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Ennio Filicicchia
 * @email     efilicicc@student.ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_TRAJECTORY_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_TRAJECTORY_HPP_

// Noesis
#include "noesis/gym/envs/raisim/common/types.hpp"

namespace noesis {
namespace gym {

class Trajectory
{
public:
  
  //! Aliases
  using Pose = ::noesis::gym::Pose;
  using Position = ::noesis::gym::Position;
  using Positions = ::noesis::gym::Trajectory3;
  
  /*
   * Instantiation
   */
  
  explicit Trajectory(const Pose& origin, double dt=0.005, double duration=10.0, double scale=0.2) {
    generate(origin, dt, duration, scale);
  }
  
  Trajectory() = default;
  
  ~Trajectory() = default;
  
  /*
   * Configurations
   */
  
  /*
   * Properties
   */

  const Positions& positions() const {
    return positions_;
  }
  
  const Position& head() const {
    return *pit_;
  }
  
  
  /*
   * Operations
   */
  
  void generate(const Pose& origin, double dt=0.005, double duration=10.0, double scale=0.2) {
    // Generate a fixed figure-eight curve starting on the initial position.
    const auto& r_0 = origin.second;
    const auto number_of_samples = static_cast<size_t>(duration / dt);
    positions_.resize(number_of_samples);
    for (int i = 0; i < positions_.size(); ++i) {
      const auto phase = 2.0 * M_PI * (i * dt) / duration;
      const auto y = 1.0 * scale * std::cos(phase);
      const auto z = 0.5 * scale * std::sin(2.0 * phase);
      positions_[i] = r_0 + Position(0.0, y, z);
    }
    // Initialize the internal iterator
    pit_ = positions_.begin();
  }
  
  void reset() {
    pit_ = positions_.begin();
  }
  
  Position sample() {
    // Capture the current sample
    Position position = *pit_;
    // Advance the iterator of the circular buffer.
    if (pit_ == positions_.end() - 1) {
      pit_ = positions_.begin();
    } else {
      pit_++;
    }
    // Return the current sample pose
    return position;
  }

private:
  //! @brief Trajectory container representing a circular buffer.
  Positions positions_;
  //! @brief An iterator helper to carry the head of the circular buffer.
  Positions::iterator pit_;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_TRAJECTORY_HPP_

/* EOF */
