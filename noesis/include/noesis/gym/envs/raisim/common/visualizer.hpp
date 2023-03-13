/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_VISUALIZER_HPP_

// C/C++
#include <memory>
#include <vector>

// RaiSim
#include <raisim/OgreVis.hpp>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/log/message.hpp"
#include "noesis/gym/train/VisualizerInterface.hpp"
#include "noesis/gym/envs/raisim/common/math.hpp"
#include "noesis/gym/envs/raisim/common/world.hpp"

namespace noesis {
namespace gym {

namespace raisim {
class CoordinateFrame
{
public:
  
  /*
   * Instantiation
   */
  
  explicit CoordinateFrame(const std::string& name, ::raisim::OgreVis* visualizer) {
    create(name, visualizer);
  }
  
  CoordinateFrame() = default;
  
  ~CoordinateFrame() = default;
  
  /*
   * Operations
   */
  
  void create(const std::string& name, ::raisim::OgreVis* visualizer) {
    const std::vector<std::string> colors = {"black", "red", "green", "blue"};
    const auto group = ::raisim::OgreVis::RAISIM_OBJECT_GROUP;
    origin_ = visualizer->addVisualObject(name + "_origin", "sphereMesh", colors[0], {0.02, 0.02, 0.02}, false, group);
    xVec_ = visualizer->addVisualObject(name + "_x", "arrowMesh", colors[1], {0.1, 0.1, 0.2}, false, group);
    yVec_ = visualizer->addVisualObject(name + "_y", "arrowMesh", colors[2], {0.1, 0.1, 0.2}, false, group);
    zVec_ = visualizer->addVisualObject(name + "_z", "arrowMesh", colors[3], {0.1, 0.1, 0.2}, false, group);
  }
  
  void update(const Position& r, const RotationMatrix& R) {
    const RotationMatrix R0 = R * math::rotation_y(M_PI / 2.0);
    const RotationMatrix R1 = R * math::rotation_x(-M_PI / 2.0);
    origin_->setPosition(r);
    xVec_->setPosition(r);
    yVec_->setPosition(r);
    zVec_->setPosition(r);
    xVec_->setOrientation(R0);
    yVec_->setOrientation(R1);
    zVec_->setOrientation(R);
  }

private:
  ::raisim::VisualObject* origin_{nullptr};
  ::raisim::VisualObject* xVec_{nullptr};
  ::raisim::VisualObject* yVec_{nullptr};
  ::raisim::VisualObject* zVec_{nullptr};
};
} // namespace raisim

struct RaiSimVisualizerConfig {
  //! @brief The name of the visualizer instance.
  std::string name{"visualizer"};
  //! @brief The scope under which the visualizer is created.
  std::string scope{"raisim"};
  //! @brief The frame-rate (<= monitor refresh rate) with which to render.
  double frames_per_second{30.0};
  //! @brief Set to value >1 for slow-motion and <1 for fast-forward visualization.
  double real_time_factor{1.0};
  //! @brief The width of the visualization window in pixels
  uint32_t window_width{1024};
  //! @brief The height of the visualization window in pixels
  uint32_t window_height{768};
  //! @brief Anti-aliasing processing configuration.
  //! @note Valid values are: 1,2,4,8
  int anti_aliasing{1};
  //! @brief Enables the use of a concurrent thread for asynchronous rendering.
  //! @warning This uses atomic operations to lock access to calls of the raisim::World::integrate1() operation.
  bool use_rendering_thread{true};
  //!@brief Enables the Im GUI panel for adding visual controls and additional visualization elements.
  bool show_gui{true};
  //! @brief Set to 'true' for verbose console output.
  bool verbose{false};
};


class RaiSimVisualizer:
  public noesis::core::Object,
  public noesis::gym::VisualizerInterface
{
public:
  // Aliases
  using ObjectBase = noesis::core::Object;
  using VisualizerBase = noesis::gym::VisualizerInterface;
  
  /*
   * Instantiation
   */
  
  explicit RaiSimVisualizer(const RaiSimVisualizerConfig& config);
  
  virtual ~RaiSimVisualizer() override;
  
  /*
   * Configurations
   */
  
  void setWorld(RaiSimWorld* world) {
    NFATAL_IF(!world, "[" << namescope() << "]: 'world' argument is an invalid (nullptr)!");
    world_ = world;
  }
  
  void setWindowTitle(const std::string& title) { visualizer_->setWindowTitle(title); }
  
  void setFramesPerSecond(double fps) { config_.frames_per_second = fps; }
  
  void setRealTimeFactor(double rtf) { config_.real_time_factor = rtf; }
  
  /*
   * Properties
   */

  bool isActive() const override { return isActive_; }
  
  bool isEnabled() const override { return isEnabled_; }
  
  const ::raisim::OgreVis& get() const { return *visualizer_; }
  
  ::raisim::OgreVis& get() { return *visualizer_; }
  
  const ::raisim::World& world() const { return *world_->get(); }
  
  ::raisim::World& world() { return *world_->get(); }
  
  double getFramesPerSecond() const { return config_.frames_per_second; }
  
  double getRealTimeFactor() const { return config_.real_time_factor; }
  
  bool isPaused() const { return isPaused_; }
  
  /*
   * Operations
   */
  
  void launch() override;
  
  void enable() override;
  
  void disable() override;
  
  void render() override;
  
  void startRecording(const std::string& directory, const std::string& filename) override;
  
  void stopRecording() override;
  
  void pause();
  
  void resume();

protected:
  
  /*
   * Implementation specifics
   */
  
  virtual void setup() {};
  
  virtual void update() {};

private:
  
  void init();
  
  void setupCallback();
  
  void renderCallback();
  
  void guiSetupCallback();
  
  void guiRenderCallback();
  
  bool guiKeyboardCallback(const ::OgreBites::KeyboardEvent &evt);

private:
  //! @brief Configurations for the visualizer.
  RaiSimVisualizerConfig config_;
  //! @brief Thread  worker used for asynchronous rendering of the visualization.
  std::unique_ptr<std::thread> renderThread_;
  //! @brief Visualization interface.
  ::raisim::OgreVis* visualizer_{nullptr};
  //! @brief World interface.
  RaiSimWorld* world_{nullptr};
  //! @brief Configures the font of the visualizer's text fields.
  ImFont* font_{nullptr};
  // @brief Indicates the true FPS achieved by the rendering loop
  std::atomic<double> framesPerSecond_{0.0};
  //! @brief Indicates if the rendering thread is currently running. Evaluates to True only if it is running.
  std::atomic_bool threadIsRunning_{false};
  //! @brief Indicates if the visualization is updating the rendering.
  std::atomic_bool isEnabled_{false};
  //! @brief Indicates of the visualization and world stepping is halted or running.
  std::atomic_bool isPaused_{false};
  //! @brief Indicates if an existing visualization has been set up.
  std::atomic_bool isActive_{false};
  //! @brief Enables showing visual bodies in the scene.
  bool showVisualBodies_{true};
  //! @brief Enables showing collision bodies in the scene.
  bool showCollisionBodies_{false};
  //! @brief Enables visualizing coordinate frames.
  bool showCoordinateFrames_{false};
  //! @brief Enables showing active contact points.
  bool showContactPoints_{false};
  //! @brief Enables showing active contact forces.
  bool showContactForces_{false};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_VISUALIZER_HPP_

/* EOF */
