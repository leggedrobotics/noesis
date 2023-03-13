/*!
 *
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    JoonHo Lee
 * @email     junja94@gmail.com
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    HaoChih Lin
 * @email     hlin@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_COMMON_SFML_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_COMMON_SFML_VISUALIZER_HPP_

// C/C++
#include <unistd.h>
#include <cmath>
#include <chrono>
#include <mutex>
#include <thread>

// SFML
#include <SFML/Graphics.hpp>

// Noesis
#include "noesis/framework/system/time.hpp"
#include "noesis/framework/log/message.hpp"
#include "noesis/framework/core/Object.hpp"
#include "noesis/gym/core/Synchronizer.hpp"
#include "noesis/gym/train/VisualizerInterface.hpp"

namespace noesis {
namespace gym {

struct SfmlVisualizerConfig {
  //! @brief The name of the visualizer instance.
  std::string name{"visualizer"};
  //! @brief The scope under which the visualizer is created.
  std::string scope{"classic"};
  //! @brief The width of the visualization window in pixels
  uint32_t window_width{500};
  //! @brief The height of the visualization window in pixels
  uint32_t window_height{500};
  //! @brief The scale of the visualization window
  float window_scale = 1.0;
  //! @brief Set to value >1 for slow-motion and <1 for fast-forward visualization.
  float real_time_factor{1.0};
  //! @brief Enables the use of a concurrent thread for asynchronous rendering.
  bool use_rendering_thread{true};
  //! @brief Set to 'true' for verbose console output.
  bool verbose{false};
};

/*!
 * @brief Visualizers for custom environments and dynamic instances. Built using Simple and Fast Multimedia Library (SFML).
 * @note Allows usage of concurrent thread for asynchronous rendering
 */
template <typename ScalarType_>
class SfmlVisualizer :
  public ::noesis::core::Object,
  public ::noesis::gym::VisualizerInterface
{
public:
  
  // Aliases
  using Object = ::noesis::core::Object;
  using Interface = ::noesis::gym::VisualizerInterface;
  using Environment = ::noesis::gym::Synchronizer<ScalarType_>;
  
  /*
   * Instantiation
   */
  
  explicit SfmlVisualizer(
      Environment* environment,
      std::function<void(void)> setup,
      std::function<void(void)> update,
      const SfmlVisualizerConfig& config):
    Object(config.name, config.scope, config.verbose),
    Interface(),
    window_(),
    texture_(),
    view_(),
    windowEvent_(),
    windowScale_(config.window_scale),
    setupCallback_(std::move(setup)),
    updateCallback_(std::move(update)),
    realTimeFactor_(config.real_time_factor),
    windowWidth_(config.window_width),
    windowHeight_(config.window_height),
    useRenderingThread_(config.use_rendering_thread),
    renderThread_(),
    environment_(environment)
  {
    NFATAL_IF(!environment_, "[" << namescope() << "]: 'environment' pointer argument is invalid (nullptr).");
    NFATAL_IF(!setupCallback_, "[" << namescope() << "]: 'setupCallback' is invalid!");
    NFATAL_IF(!updateCallback_, "[" << namescope() << "]: 'updateCallback' is invalid!");
    NNOTIFY_IF(config.verbose, "[SfmlVisualizer]: New instance at: " << std::hex << this);
  }
  
  virtual ~SfmlVisualizer() {
    if (isActive_) {
      if (renderThread_) {
        environment_->unlock();
        threadIsRunning_ = false;
        isEnabled_ = false;
        renderThread_->join();
      } else {
        isActive_ = false;
        isEnabled_ = false;
      }
      window_.close();
    }
  }
  
  /*
   * Configurations
   */

  void setFramesPerSecond(float fps) {
    framesPerSecond_ = fps;
  }
  
  void setRealTimeFactor(float factor) {
    realTimeFactor_ = factor;
  }
  
  /*
   * Properties
   */

  bool isActive() const override { return isActive_; }
  
  bool isEnabled() const override { return isEnabled_; }
  
  typename Environment::Interface* env() const { return environment_->env(); }
  
  sf::RenderWindow& getWindow() { return window_; }
  
  sf::RenderTexture& getTexture() { return texture_; }
  
  float getWindowScale() const { return windowScale_; }
  
  uint32_t getWindowWidth() const { return windowWidth_; }
  
  uint32_t getWindowHeight() const { return windowHeight_; }
  
  double getFramesPerSecond() const { return framesPerSecond_; }
  
  float getRealTimeFactor() const { return realTimeFactor_; }
  
  sf::Vector2f getMouseInput() const {
    hasMouseInput_ = false;
    return mouseInput_;
  }
  
  bool hasMouseInput() const { return hasMouseInput_; }
  
  /*
   * Operations
   */

  void launch() override {
    NINFO_IF(!environment_, "[" << namescope() << "]: The environment to visualize has not been set!");
    NINFO("[" << this->namescope() << "]: Starting-up visualization ...");
    if (useRenderingThread_) { // Threaded execution
      environment_->lock();
      isEnabled_ = true;
      threadIsRunning_ = true;
      renderThread_ = std::make_unique<std::thread>(&SfmlVisualizer::renderCallback, this);
      // Wait until the rendering thread is ready.
      while (!isActive_);
    } else { // Execution managed by main thread
      // Create SFML window with defined entities
      createWindow();
      window_.setActive(true);
      if (updateCallback_) updateCallback_();
      window_.display();
      isEnabled_ = true;
      isActive_ = true;
    }
  }
  
  void enable() override {
    if (renderThread_) { environment_->lock(); }
    isEnabled_ = true;
  }
  
  void disable() override {
    isEnabled_ = false;
    if (renderThread_) { environment_->unlock(); }
  }
  
  void render() override {
    NFATAL_IF(renderThread_, "[" << namescope() << "]: Cannot render manually while using async rendering thread!");
    if (isEnabled_) {
      // poll events to check GUI input
      windowEventWorker();
      // activate the window's context
      window_.setActive(true);
      // update the context of window
      if (updateCallback_) updateCallback_();
      // end the current frame
      window_.display();
      // store rendered context if video recording enabled
      if (isVideoRecording_) {
        frameCounter_++;
        // write into output file
        fwrite(texture_.getTexture().copyToImage().getPixelsPtr(), 4, windowWidth_ * windowHeight_, ffmpegFile_);
      }
    }
  }
  
  void startRecording(const std::string& directory, const std::string& filename) override {
    if (!boost::filesystem::exists(directory)) { boost::filesystem::create_directories(directory); }
    videoFileName_ = noesis::utils::make_namescope({directory, filename +".mp4"});
    isVideoRecording_ = true;
    frameCounter_ = 0;
    auto desiredFPS = static_cast<int>(framesPerSecond_);
    // define shell command for creating codec video using FFMPEG
    std::string command =
      "ffmpeg -loglevel panic -r " + std::to_string(desiredFPS) + " -f rawvideo -pix_fmt rgba -s "
      + std::to_string(windowWidth_) + "x" + std::to_string(windowHeight_)
      + " -i - -map_metadata 0 -metadata:s:v rotate=\"0\" -threads 0 -preset fast -y -pix_fmt yuv420p -crf 23 "
      + videoFileName_;
    ffmpegFile_ = popen(command.c_str(), "w");
    NINFO("[" << namescope() << "]: Started recording video: " << videoFileName_);
  }
  
  void stopRecording() override {
    if (isVideoRecording_) {
      isVideoRecording_ = false;
      fflush(ffmpegFile_);
      pclose(ffmpegFile_);
      frameCounter_ = 0;
      NINFO("[" << namescope() << "]: Video saved at: " << videoFileName_);
    }
  }
  
  void pause() {
    isPaused_ = true;
  }
  
  void resume() {
    isPaused_ = false;
  }
  
private:
  
  void createWindow() {
    // Create the visualization window
    auto windowStyle = sf::Style::Titlebar | sf::Style::Close;
    window_.create(sf::VideoMode(windowWidth_, windowHeight_), name(), windowStyle, sf::ContextSettings(24, 8, 8));
    // Create the render texture
    texture_.create(windowWidth_, windowHeight_);
    texture_.setSmooth(true);
    // Fill color for mouse click point
    mouseCursor_.setFillColor(sf::Color::Blue);
    // Create visualizer view
    view_ = window_.getDefaultView();
    // Setup implementation-specifics
    if (setupCallback_) setupCallback_();
    // Configure the visualization
    window_.setFramerateLimit(static_cast<unsigned int>(framesPerSecond_));
    // Set window activity as false in the main thread
    window_.setActive(false);
  }
  
  void renderCallback() {
    // Create SFML window with defined entities
    createWindow();
    window_.setActive(true);
    if (updateCallback_) updateCallback_();
    window_.display();
    double t_sim = 0.0;
    auto t_start = std::chrono::system_clock::now();
    auto t_end = t_start;
    while (threadIsRunning_) {
      // poll events to check GUI input
      windowEventWorker();
      // Update the dynamics only if enabled in the GUI
      if (isEnabled_) {
        // Step the physics by the number of simulation time-steps necessary for a single rendering cycle
        // NOTE: The world is progressed only if enabled in the GUI or the visualization is enabled
        while(t_sim < 0.0 && !isPaused_ && isEnabled_) {
          environment_->start();
          if (t_sim < 0.0 && isEnabled_) {
            environment_->wait();
            t_sim += environment_->time_step();
          }
        }
        // Adjust simulation time-progress according to the visualization rate
        auto t_vis = realTimeFactor_ / framesPerSecond_;
        if (t_sim > -0.1 * t_vis) { t_sim -= t_vis; }
        // activate the window's context
        window_.setActive(true);
        // update the context of window
        if (updateCallback_) updateCallback_(); // TODO: FIX THIS BECAUSE IT STILL THROWS WHEN DESTRUCTOR IS CALLED
        // end the current frame
        window_.display();
        // Update measured FPS
        t_end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;
        // Perform a sleep if necessary in order to realize the target FPS.
        auto t_diff = 1.0 / framesPerSecond_ - elapsed.count();
        if (t_diff > 1e-6) { usleep(static_cast<uint32_t>(t_diff*1.0e+6)); }
        // Capture the new start time
        t_start = std::chrono::system_clock::now();
        // store rendered context if video recording enabled
        if (isVideoRecording_) {
          frameCounter_++;
          // write into output file
          fwrite(texture_.getTexture().copyToImage().getPixelsPtr(), 4, windowWidth_ * windowHeight_, ffmpegFile_);
          fflush(ffmpegFile_);
        }
      }
      // NOTE: We deliberately indicate that the thread is activate after we perform one round of rendering.
      isActive_ = true;
    }
    // deactivate the window's context
    window_.setActive(false);
    isActive_ = false;
  }
  
  void windowEventWorker() {
    // check the window's events that were triggered since the last iteration of the loop
    while (window_.pollEvent(windowEvent_)) {
      switch (windowEvent_.type) {
        // "mouse scrolling" event: we zoom in/out the window
        case sf::Event::MouseWheelScrolled:
          if (windowEvent_.mouseWheelScroll.delta < 0) { windowScale_ *= 1.05; }
          else if (windowEvent_.mouseWheelScroll.delta > 0) {  windowScale_ *= 0.95; }
          // Set zoom of the window
          view_ = window_.getDefaultView();
          view_.zoom(windowScale_);
          window_.setView(view_);
          break;
          // "mouse clicked" event: mark the mouse position
        case sf::Event::MouseButtonReleased:
          mouseInput_ = sf::Vector2f((float)sf::Mouse::getPosition(window_).x, (float)sf::Mouse::getPosition(window_).y);
          mouseCursor_.setPosition(mouseInput_.x, mouseInput_.y);
          window_.draw(mouseCursor_);
          window_.display();
          hasMouseInput_ = true;
          NINFO_IF(this->isVerbose(),
                   "[" << namescope() << "]: Mouse Input (x,y): (" << mouseInput_.x << ", " << mouseInput_.y << ")");
          break;
        default:
          break;
      }
    }
  }
  
private:
  //! @brief TODO
  std::function<void(void)> setupCallback_;
  //! @brief TODO
  std::function<void(void)> updateCallback_;
  //! @brief TODO
  sf::RenderWindow window_;
  //! @brief TODO
  sf::RenderTexture texture_;
  //! @brief TODO
  sf::View view_;
  //! @brief SFML event worker and synchronization mechanisms.
  sf::Event windowEvent_;
  //! @brief TODO
  sf::CircleShape mouseCursor_{sf::CircleShape(5.0)};
  //! @brief TODO
  sf::Vector2f mouseInput_{sf::Vector2f(0,0)};
  //! @brief Filename for recording video
  std::string videoFileName_{""};
  //! @brief Stores the frame-rate supported by the active monitor.
  double framesPerSecond_{30.0};
  //! @brief Real-time factor for rendering videos.
  float realTimeFactor_{1.0f};
  //! @brief TODO
  float windowScale_;
  //! @brief TODO
  mutable std::atomic_bool hasMouseInput_{false};
  //! @brief Indicates if the rendering thread is currently running. Evaluates to True only if it is running.
  std::atomic_bool threadIsRunning_{false};
  //! @brief Indicates if the visualization is updating the rendering.
  std::atomic_bool isEnabled_{false};
  //! @brief Indicates of the visualization and world stepping is halted or running.
  std::atomic_bool isPaused_{false};
  //! @brief Indicates if an existing visualization has been set up.
  std::atomic_bool isActive_{false};
  //! @brief The width of the rendering window.
  uint32_t windowWidth_{500};
  //! @brief The height of the rendering window.
  uint32_t windowHeight_{500};
  //! @brief Variable for counting total number of frames rendered
  int frameCounter_{0};
  //! @brief Enables the use of a concurrent thread for asynchronous rendering.
  bool useRenderingThread_{false};
  //! @brief Flag indicating if video is to be recorded
  bool isVideoRecording_{false};
  //! @brief Instance of output file where data will be written to
  FILE* ffmpegFile_{nullptr};
  //! @brief Thread worker used for asynchronous rendering of the visualization.
  std::unique_ptr<std::thread> renderThread_;
  //! @brief A pointer to the visualized environment to be wrapped.
  Environment* environment_{nullptr};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_COMMON_SFML_VISUALIZER_HPP_

/* EOF */
