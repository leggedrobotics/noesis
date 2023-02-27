/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_TRAIN_VISUALIZER_INTERFACE_HPP_
#define NOESIS_GYM_TRAIN_VISUALIZER_INTERFACE_HPP_

// C/C++
#include <string>

namespace noesis {
namespace gym {

class VisualizerInterface
{
public:
  
  /*
   * Instantiation
   */
  
  VisualizerInterface() = default;

  virtual ~VisualizerInterface() = default;
  
  /*
   * Properties
   */
  
  virtual bool isActive() const = 0;
  
  virtual bool isEnabled() const = 0;
  
  /*
   * Operations
   */

  virtual void launch() = 0;
  
  virtual void enable() = 0;
  
  virtual void disable() = 0;
  
  virtual void render() = 0;
  
  virtual void startRecording(const std::string& directory, const std::string& filename) = 0;

  virtual void stopRecording() = 0;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_TRAIN_VISUALIZER_INTERFACE_HPP_

/* EOF */
