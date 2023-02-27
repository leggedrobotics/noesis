/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Environments
#include "noesis/gym/envs/raisim/common/visualizer.hpp"

namespace noesis {
namespace gym {

RaiSimVisualizer::RaiSimVisualizer(const RaiSimVisualizerConfig& config):
  ObjectBase(config.name, config.scope, config.verbose),
  VisualizerBase(),
  visualizer_(::raisim::OgreVis::get()),
  config_(config),
  framesPerSecond_(config.frames_per_second)
{
}

RaiSimVisualizer::~RaiSimVisualizer() {
  if (isActive_) {
    NINFO_IF(this->isVerbose(), "[" << this->namescope() << "]: Stopping visualizer ...");
    if (renderThread_) { // Threaded execution
      isEnabled_ = false;
      threadIsRunning_ = false;
      world_->unlock();
      renderThread_->join();
      // NOTE: We must wait until the visualizer is inactive
      // to ensure proper termination.
      while(isActive_);
    } else { // Execution managed by main thread
      visualizer_->closeApp();
      isActive_ = false;
      isEnabled_ = false;
    }
  }
}

void RaiSimVisualizer::launch() {
  NINFO_IF(!world_, "[" << this->namescope() << "]: The world to visualize has not been set!");
  NINFO("[" << this->namescope() << "]: Starting-up visualization ...");
  if (config_.use_rendering_thread) { // Threaded execution
    world_->lock();
    threadIsRunning_ = true;
    renderThread_ = std::make_unique<std::thread>(&RaiSimVisualizer::renderCallback, this);
    // Wait until the rendering thread is ready.
    while (!isActive_);
  } else { // Execution managed by main thread
    init();
    visualizer_->initApp();
    update();
    visualizer_->renderOneFrame();
    isActive_ = true;
  }
  isEnabled_ = true;
}

void RaiSimVisualizer::enable() {
  if (renderThread_) { world_->lock(); }
  isEnabled_ = true;
}

void RaiSimVisualizer::disable() {
  isEnabled_ = false;
  if (renderThread_) { world_->unlock(); }
}

void RaiSimVisualizer::render() {
  NFATAL_IF(renderThread_, "[" << namescope() << "]: Cannot render manually while using async rendering thread!");
  if (isEnabled_) {
    update();
    visualizer_->renderOneFrame();
  }
}

void RaiSimVisualizer::startRecording(const std::string& directory, const std::string& filename) {
  boost::filesystem::create_directories(directory);
  visualizer_->startRecordingVideo(noesis::utils::make_namescope({directory, filename+".mp4"}));
}

void RaiSimVisualizer::stopRecording() {
  visualizer_->stopRecordingVideoAndSave();
}

void RaiSimVisualizer::pause() {
  isPaused_ = true;
}

void RaiSimVisualizer::resume() {
  isPaused_ = false;
}

void RaiSimVisualizer::init() {
  // Configure the visualization
  visualizer_->setWorld(world_->get());
  visualizer_->setDesiredFPS(config_.frames_per_second);
  visualizer_->setAntiAliasing(config_.anti_aliasing);
  visualizer_->setWindowSize(config_.window_width, config_.window_height);
  visualizer_->setSetUpCallback(std::bind(&RaiSimVisualizer::setupCallback, this));
  if (config_.show_gui) {
    visualizer_->setImguiSetupCallback(std::bind(&RaiSimVisualizer::guiSetupCallback, this));
    visualizer_->setImguiRenderCallback(std::bind(&RaiSimVisualizer::guiRenderCallback, this));
  }
  visualizer_->setKeyboardCallback(std::bind(&RaiSimVisualizer::guiKeyboardCallback, this, std::placeholders::_1));
}

void RaiSimVisualizer::setupCallback() {
  auto* sm = visualizer_->getSceneManager();
  // Set default camera pose
  visualizer_->getCameraMan()->setStyle(::raisim::CameraStyle::CS_ORBIT);
  visualizer_->getCameraMan()->setYawPitchDist(Ogre::Radian(0.0), Ogre::Radian(-1.2f), Ogre::Real(3.0f), false);
  // Create a default skybox for the scene
  Ogre::Quaternion quat;
  quat.FromAngleAxis(Ogre::Radian(M_PI_2), {1, 0, 0}); // NOTE: we rotate the sky because OGRE does not use z-axis as vertical direction
  sm->setSkyBox(true, "BaseWhiteNoLighting", 500, true, quat, Ogre::ResourceGroupManager::AUTODETECT_RESOURCE_GROUP_NAME);
  // Setup implementation-specifics
  setup();
}

void RaiSimVisualizer::renderCallback() {
  init();
  visualizer_->initApp();
  update();
  visualizer_->renderOneFrame();
  isActive_ = true;
  double t_sim = 0.0;
  auto t_start = std::chrono::system_clock::now();
  auto t_end = t_start;
  while (threadIsRunning_) {
    if (isEnabled_) {
      // Step the physics by the number of simulation time-steps necessary for a single rendering cycle
      // NOTE: The world is progressed only if enabled in the GUI or the visualization is enabled
      while(t_sim < 0.0 && !isPaused_ && isEnabled_) {
        world_->start();
        t_sim += world_->timestep();
        if (t_sim < 0.0 && isEnabled_) { world_->wait(); }
      }
      // Adjust simulation time-progress according to the visualization rate
      auto t_vis = config_.real_time_factor / config_.frames_per_second;
      if (t_sim > -0.1 * t_vis) { t_sim -= t_vis; }
      // Update and render scene
      update();
      visualizer_->renderOneFrame();
      // Update measured FPS
      t_end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = t_end - t_start;
      framesPerSecond_ = 1.0 / elapsed.count();
      t_start = t_end;
    }
  }
  visualizer_->closeApp();
  isActive_ = false;
}

void RaiSimVisualizer::guiSetupCallback() {
  #define HI(v)   ImVec4(0.502f, 0.075f, 0.256f, v)
  #define MED(v)  ImVec4(0.455f, 0.198f, 0.301f, v)
  #define LOW(v)  ImVec4(0.232f, 0.201f, 0.271f, v)
  #define BG(v)   ImVec4(0.200f, 0.220f, 0.270f, v)
  #define TEXT(v) ImVec4(0.860f, 0.930f, 0.890f, v)
  auto &style = ImGui::GetStyle();
  style.Alpha = 0.8;
  style.Colors[ImGuiCol_Text]                  = TEXT(0.78f);
  style.Colors[ImGuiCol_TextDisabled]          = TEXT(0.28f);
  style.Colors[ImGuiCol_WindowBg]              = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
  style.Colors[ImGuiCol_ChildWindowBg]         = BG( 0.58f);
  style.Colors[ImGuiCol_PopupBg]               = BG( 0.9f);
  style.Colors[ImGuiCol_Border]                = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
  style.Colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  style.Colors[ImGuiCol_FrameBg]               = BG( 1.00f);
  style.Colors[ImGuiCol_FrameBgHovered]        = MED( 0.78f);
  style.Colors[ImGuiCol_FrameBgActive]         = MED( 1.00f);
  style.Colors[ImGuiCol_TitleBg]               = LOW( 1.00f);
  style.Colors[ImGuiCol_TitleBgActive]         = HI( 1.00f);
  style.Colors[ImGuiCol_TitleBgCollapsed]      = BG( 0.75f);
  style.Colors[ImGuiCol_MenuBarBg]             = BG( 0.47f);
  style.Colors[ImGuiCol_ScrollbarBg]           = BG( 1.00f);
  style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered]  = MED( 0.78f);
  style.Colors[ImGuiCol_ScrollbarGrabActive]   = MED( 1.00f);
  style.Colors[ImGuiCol_CheckMark]             = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
  style.Colors[ImGuiCol_SliderGrab]            = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
  style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
  style.Colors[ImGuiCol_Button]                = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
  style.Colors[ImGuiCol_ButtonHovered]         = MED( 0.86f);
  style.Colors[ImGuiCol_ButtonActive]          = MED( 1.00f);
  style.Colors[ImGuiCol_Header]                = MED( 0.76f);
  style.Colors[ImGuiCol_HeaderHovered]         = MED( 0.86f);
  style.Colors[ImGuiCol_HeaderActive]          = HI( 1.00f);
  style.Colors[ImGuiCol_Column]                = ImVec4(0.14f, 0.16f, 0.19f, 1.00f);
  style.Colors[ImGuiCol_ColumnHovered]         = MED( 0.78f);
  style.Colors[ImGuiCol_ColumnActive]          = MED( 1.00f);
  style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
  style.Colors[ImGuiCol_ResizeGripHovered]     = MED( 0.78f);
  style.Colors[ImGuiCol_ResizeGripActive]      = MED( 1.00f);
  style.Colors[ImGuiCol_PlotLines]             = TEXT(0.63f);
  style.Colors[ImGuiCol_PlotLinesHovered]      = MED( 1.00f);
  style.Colors[ImGuiCol_PlotHistogram]         = TEXT(0.63f);
  style.Colors[ImGuiCol_PlotHistogramHovered]  = MED( 1.00f);
  style.Colors[ImGuiCol_TextSelectedBg]        = MED( 0.43f);
  style.Colors[ImGuiCol_ModalWindowDarkening]  = BG( 0.73f);
  style.WindowPadding            = ImVec2(6, 4);
  style.WindowRounding           = 0.0f;
  style.FramePadding             = ImVec2(5, 2);
  style.FrameRounding            = 3.0f;
  style.ItemSpacing              = ImVec2(7, 1);
  style.ItemInnerSpacing         = ImVec2(1, 1);
  style.TouchExtraPadding        = ImVec2(0, 0);
  style.IndentSpacing            = 6.0f;
  style.ScrollbarSize            = 12.0f;
  style.ScrollbarRounding        = 16.0f;
  style.GrabMinSize              = 20.0f;
  style.GrabRounding             = 2.0f;
  style.WindowTitleAlign.x = 0.50f;
  style.Colors[ImGuiCol_Border] = ImVec4(0.539f, 0.479f, 0.255f, 0.162f);
  style.FrameBorderSize = 0.0f;
  style.WindowBorderSize = 1.0f;
  ImGuiIO &io = ImGui::GetIO();
  font_ = io.Fonts->AddFontFromFileTTF((::raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 18.0f);
}

void RaiSimVisualizer::guiRenderCallback() {
  
  ImGui::SetNextWindowPos({0, 0});
  if (!ImGui::Begin("Simulation")) {
    ImGui::End();
    return;
  }
  
  auto* vis = ::raisim::OgreVis::get();
  auto* world = vis->getWorld();
  unsigned long mask = 0;
  ImGui::PushFont(font_);
  
  // 1
  if (isPaused_) {
    if(ImGui::Button("Resume")) { isPaused_ = false; }
  } else {
    if(ImGui::Button("Pause")) { isPaused_ = true; }
  }
  ImGui::Text("Time: %10.4f", world->getWorldTime());
  ImGui::Text("FPS: %3.2f", (double)framesPerSecond_);
  
  // 2
  if (ImGui::CollapsingHeader("Real-Time Factor")) {
    auto rtf = static_cast<float>(config_.real_time_factor);
    ImGui::SliderFloat("", &rtf, 1e-4, 10.0, "Value: %5.4f", 3);
    ImGui::SameLine();
    if(ImGui::Button("Reset")) { rtf = 1.0; }
    config_.real_time_factor = static_cast<double>(rtf);
  }
  
  // 3
  if (ImGui::CollapsingHeader("Solver")) {
    ImGui::Text("Iterations: %d", world->getContactSolver().getLoopCounter());
    ImGui::Text("Contacts: %lu", world->getContactProblem()->size());
    std::vector<float> error;
    error.reserve(static_cast<size_t>(world->getContactSolver().getLoopCounter()));
    for(int i=0; i<world->getContactSolver().getLoopCounter(); i++) {
      error.push_back(float(std::log(world->getContactSolver().getErrorHistory()[i])));
    }
  }
  
  // 4
  if (ImGui::CollapsingHeader("Options")) {
    ImGui::Checkbox("Visual Bodies", &showVisualBodies_);
    ImGui::Checkbox("Collision Bodies", &showCollisionBodies_);
    ImGui::Checkbox("Contact Points", &showContactPoints_);
    ImGui::Checkbox("Contact Forces", &showContactForces_);
    if(showVisualBodies_) mask |= ::raisim::OgreVis::RAISIM_OBJECT_GROUP;
    if(showCollisionBodies_) mask |= ::raisim::OgreVis::RAISIM_COLLISION_BODY_GROUP;
    if(showContactPoints_) mask |= ::raisim::OgreVis::RAISIM_CONTACT_POINT_GROUP;
    if(showContactForces_) mask |= ::raisim::OgreVis::RAISIM_CONTACT_FORCE_GROUP;
    vis->setVisibilityMask(mask);
  }
  
  // 5
  if (ImGui::CollapsingHeader("Objects")) {
    auto selected = vis->getSelected();
    auto ro = std::get<0>(selected);
    auto li = std::get<1>(selected);
    if(ro) {
      if(!ro->getName().empty()){
        ImGui::Text("%s", ("name: " + ro->getName() + "/" + vis->getSelectedGraphicalObject()->name).c_str());
      } else {
        ImGui::Text("Unnamed object");
      }
      ::raisim::Vec<3> pos; ro->getPosition(li, pos);
      ::raisim::Vec<3> vel; ro->getVelocity(li, vel);
      ::raisim::Vec<4> ori; ::raisim::Mat<3,3> mat; ro->getOrientation(li, mat); ::raisim::rotMatToQuat(mat, ori);
      ImGui::Text("Position");
      ImGui::Text("x = %2.2f, y = %2.2f, z = %2.2f", pos[0], pos[1], pos[2]);
      ImGui::Text("Velocity");
      ImGui::Text("x = %2.2f, y = %2.2f, z = %2.2f", vel[0], vel[1], vel[2]);
      ImGui::Text("Orientation");
      ImGui::Text("x = %2.2f, x = %2.2f, y = %2.2f, z = %2.2f", ori[0], ori[1], ori[2], ori[3]);
      ImGui::Text("Contacts: %lu", ro->getContacts().size());
    }
  }
  
  ImGui::PopFont();
  ImGui::End();
}

bool RaiSimVisualizer::guiKeyboardCallback(const ::OgreBites::KeyboardEvent &evt) {
  auto& key = evt.keysym.sym;
  switch (key) {
    case '1':
      showVisualBodies_ = !showVisualBodies_;
      break;
    case '2':
      showCollisionBodies_ = !showCollisionBodies_;
      break;
    case '3':
      showCoordinateFrames_ = !showCoordinateFrames_;
      break;
    case '4':
      showContactPoints_ = !showContactPoints_;
      break;
    case '5':
      showContactForces_ = !showContactForces_;
      break;
    case SDLK_SPACE:
      isPaused_ = !isPaused_;
      break;
    default:
      break;
  }
  return false;
}

} // namespace gym
} // namespace noesis

/* EOF */
