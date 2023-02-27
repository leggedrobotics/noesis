/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Environments
#include "noesis/gym/envs/mujoco/common/visualizer.hpp"

namespace noesis {
namespace gym {

MujocoVisualizer::MujocoVisualizer(const MujocoVisualizerConfig& config):
  ObjectBase(config.name, config.scope, config.verbose),
  VisualizerBase(),
  realTimeFactor_(config.real_time_factor),
  useRenderThread_(config.use_render_thread),
  bodyIdForTrackingCam_(config.body_id_for_cam_tracking)
{
}

MujocoVisualizer::~MujocoVisualizer() {
  if (isActive_) {
    NFATAL_IF(!simulation_, "[" << namescope() << "]: Simulation instance used for visualization has not been set!");
    NINFO("[" << namescope() << "]: Shutting-down visualization ...");
    if (renderThread_) { // Threaded execution
      simulation_->unlock();
      threadIsRunning_ = false;
      isEnabled_ = false;
      renderThread_->join();
    } else { // Execution managed by main thread
      visualizer_->stop();
      visualizer_->shutdown();
      isActive_ = false;
      isEnabled_ = false;
    }
  }
}

void MujocoVisualizer::setBodyForCameraTracking(const int& body_index, const double& elevation, const double& distance,
                                                const std::array<double, 3>& location) {
  auto camera = visualizer_->getCamera();
  if (body_index >= 0) {
    camera.elevation = elevation;
    camera.distance = distance;
    for (int i = 0; i < 3; i++) {
      camera.lookat[i] = location[i];
    }
    // set to track camera
    camera.trackbodyid = body_index;
    camera.type = mjCAMERA_TRACKING;
  } else {
    camera.trackbodyid = -1;
    camera.type = mjCAMERA_FREE;
  }
  visualizer_->setCamera(camera);
  // update internal parameter
  bodyIdForTrackingCam_ = body_index;
}

void MujocoVisualizer::setBodyForCameraTracking(const std::string& body_name, const double& elevation, const double& distance,
                                                const std::array<double, 3>& location) {
  auto camera = visualizer_->getCamera();
  camera.elevation = elevation;
  camera.distance = distance;
  for (int i = 0; i < 3; i++){
    camera.lookat[i] = location[i];
  }
  // set to track camera
  int body_index = simulation_->get()->getBodyIndexFromName(body_name);
  camera.trackbodyid = body_index;
  camera.type = mjCAMERA_TRACKING;
  visualizer_->setCamera(camera);
  // update internal parameter
  bodyIdForTrackingCam_ = body_index;
}

double MujocoVisualizer::getFramesPerSecond() const {
  return fps_;
}

double MujocoVisualizer::getRealTimeFactor() const {
  return realTimeFactor_;
}


bool MujocoVisualizer::isActive() const {
  return isActive_;
}

bool MujocoVisualizer::isEnabled() const {
  return isEnabled_;
}

void MujocoVisualizer::launch() {
  NFATAL_IF(!simulation_, "[" << this->namescope() << "]: Simulation instance used for visualization has not been set!");
  NINFO("[" << this->namescope() << "]: Starting-up visualization ...");
  if (useRenderThread_) { // Threaded execution
    simulation_->lock();
    threadIsRunning_ = true;
    renderThread_ = std::make_unique<std::thread>(&MujocoVisualizer::renderCallback, this);
    // Wait until the rendering thread is ready.
    while (!isActive_);
  } else { // Execution managed by main thread
    init();
    update();
    visualizer_->render();
    isActive_ = true;
  }
  isEnabled_ = true;
}

void MujocoVisualizer::enable() {
  isEnabled_ = true;
  if (renderThread_) {
    simulation_->lock();
  }
}

void MujocoVisualizer::disable() {
  isEnabled_ = false;
  if (renderThread_) {
    simulation_->unlock();
  }
}

void MujocoVisualizer::render() {
  if (renderThread_) {
    NFATAL("[" << namescope() << "]: Cannot render manually while using async rendering thread!");
  }
  if (isEnabled_) {
    update();
    visualizer_->render();
  }
}

void MujocoVisualizer::startRecording(const std::string& directory, const std::string& filename) {
  visualizer_->startRecordingVideo(noesis::utils::make_namescope({directory, filename+".mp4"}));
}

void MujocoVisualizer::stopRecording() {
  visualizer_->stopRecordingVideoAndSave();
}

void MujocoVisualizer::init() {
  mujoco::VisualizerConfig visConf;
  visConf.name = namescope();
  visConf.verbose = isVerbose();
  visualizer_ = std::make_unique<mujoco::Visualizer>(visConf);
  visualizer_->startup(simulation_->get());
  fps_ = static_cast<double>(glfwGetVideoMode(glfwGetPrimaryMonitor())->refreshRate);
  // Set to tracking body camera
  if(bodyIdForTrackingCam_ >= 0) {
    setBodyForCameraTracking(bodyIdForTrackingCam_);
  }
}

void MujocoVisualizer::renderCallback() {
  // Construct visualization resources in the worker thread
  init();
  update();
  visualizer_->render();
  // Start threaded visualization
  isActive_ = true;
  double time = 0;
  while (threadIsRunning_) {
    if (isEnabled_) {
      // Step the physics by the number of simulation time-steps necessary for a single rendering cycle
      while(time < 0.0 && isEnabled_) {
        simulation_->release();
        time += simulation_->timestep();
        while (!simulation_->isWaiting() && isEnabled_);
      }
      time -= realTimeFactor_ / fps_;
      update();
      visualizer_->render();
    }
  }
  // Shutdown visualization resources
  visualizer_->stop();
  visualizer_->shutdown();
  isActive_ = false;
}

} // namespace gym
} // namespace noesis

/* EOF */
