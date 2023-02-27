/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_MUJOCO_COMMON_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_MUJOCO_COMMON_VISUALIZER_HPP_

// C/C++
#include <unistd.h>
#include <atomic>
#include <memory>
#include <vector>
#include <thread>

// MuJoCo
#include <mujoco_cpp/mujoco_cpp.hpp>

// Noesis
#include <noesis/framework/log/message.hpp>
#include <noesis/framework/core/Object.hpp>

// Environments
#include "noesis/gym/envs/mujoco/common/simulation.hpp"
#include "noesis/gym/train/VisualizerInterface.hpp"

namespace noesis {
namespace gym {

/*!
 * @brief Helper struct for visualizer construction.
 */
struct MujocoVisualizerConfig : public mujoco::VisualizerConfig
{
  //! @brief The name-scope to which the dynamics instance belongs.
  std::string scope{""};
  //! @brief Sets how much faster or slower than real-time the visualization will render.
  double real_time_factor{1.0};
  //! @brief Set to true if rendering should be executed asynchronously.
  bool use_render_thread{false};
  //! @brief Set to the agent's body ID to track during visualization
  //! @note -1 indicates no body tracking i.e. the camera will free
  int body_id_for_cam_tracking{-1};
};

/*!
 * @brief Visualizer for MuJoCo environments and dynamics instances.
 *
 * This adapter class provides a front-end using the noesis::environment::VisualizerInterface, by
 * managing and wrapping the operations of the mujoco::Visualizer class.
 */
class MujocoVisualizer :
  public noesis::core::Object,
  public noesis::gym::VisualizerInterface
{
public:
  // Aliases
  using ObjectBase = noesis::core::Object;
  using VisualizerBase = noesis::gym::VisualizerInterface;
  using SimulationType = noesis::gym::MujocoSimulation;

  /*
   * Instantiation
   */

  explicit MujocoVisualizer(const MujocoVisualizerConfig &config);

  virtual ~MujocoVisualizer() override;

  /*
   * Configurations
   */

  void setSimulation(SimulationType *simulation) {
    NFATAL_IF(!simulation, "[" << namescope() << "]: 'simulation' argument is an invalid pointer (nullptr)!");
    simulation_ = simulation;
  }


  void setBodyForCameraTracking(const int &body_index, const double &elevation = -20, const double &distance = 4.0,
                                const std::array<double, 3> &location = {0.0, 0.0, 2.0});

  void
  setBodyForCameraTracking(const std::string &body_name, const double &elevation = -20, const double &distance = 4.0,
                           const std::array<double, 3> &location = {0.0, 0.0, 2.0});

  /*
   * Properties
   */

  double getFramesPerSecond() const;

  double getRealTimeFactor() const;

  bool isActive() const override;

  bool isEnabled() const override;

  /*
   * Visualizer Interface
   */

  void launch() override;

  void render() override;

  void enable() override;

  void disable() override;

  void startRecording(const std::string &directory, const std::string &filename) override;

  void stopRecording() override;

protected:

  /*
   * Implementation specifics
   */

  virtual void setup() {};

  virtual void update() {};

private:

  void init();

  void renderCallback();

private:
  //! @brief Interface wrapper that encapsulates a MuJoCo simulation instance.
  std::unique_ptr<mujoco::Visualizer> visualizer_;
  //! @brief Thread  worker used for asynchronous rendering of the visualization.
  std::unique_ptr<std::thread> renderThread_;
  //! @brief Pointer the simulation to be visualized.
  SimulationType *simulation_{nullptr};
  //! @brief Controls how much faster or slower than real-time the visualization will be running.
  double realTimeFactor_{1.0};
  //! @brief Stores the frame-rate supported by the active monitor.
  double fps_{0.0};
  //! @brief Determines if the asynchronous rendering thread is to be launched on startup.
  bool useRenderThread_{false};
  //! @brief Agent's body ID to track during visualization.
  int bodyIdForTrackingCam_{-1};
  //! @brief Indicates if the rendering thread is currently running. Evaluates to True only if it is running.
  std::atomic_bool threadIsRunning_{false};
  //! @brief Indicates if the visualization is updating the rendering.
  std::atomic_bool isEnabled_{false};
  //! @brief Indicates if an existing visualization has been set up.
  std::atomic_bool isActive_{false};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_MUJOCO_COMMON_VISUALIZER_HPP_

/* EOF */
